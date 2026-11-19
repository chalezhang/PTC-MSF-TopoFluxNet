"""Supervised multi-objective loss; gradients and flux are explicitly discrete proxies."""
import torch
import torch.nn.functional as F
import config as c


def mean_weighted(error,weight):
    # weight broadcasts over channels; normalize by the number of contributing
    # elements (not by the entire grid including solids).
    expanded=weight.expand_as(error)
    return (error*expanded).sum()/expanded.sum().clamp_min(1.0)


def components(pred,true,x):
    m=x[:,0:1].clamp(0.,1.)
    throat=x[:,7:8].clamp(0.,1.)*m
    wall=(x[:,1:2]<0.26).to(pred.dtype)*m
    w=m*(1+2*wall)
    p=mean_weighted(F.smooth_l1_loss(pred[:,:1],true[:,:1],beta=.2,reduction='none'),w)
    v=mean_weighted(F.smooth_l1_loss(pred[:,1:3],true[:,1:3],beta=.2,reduction='none'),w)
    local=mean_weighted(F.smooth_l1_loss(pred[:,1:3],true[:,1:3],beta=.2,reduction='none'),
                        throat)
    def face_gradient_error(a,b,axis,focus=None):
        da=torch.diff(a,dim=axis)-torch.diff(b,dim=axis)
        if axis==-1:
            valid=m[:,:,:,:-1]*m[:,:,:,1:]
            if focus is not None:
                valid=valid*torch.maximum(focus[:,:,:,:-1],focus[:,:,:,1:])
        else:
            valid=m[:,:,:-1,:]*m[:,:,1:,:]
            if focus is not None:
                valid=valid*torch.maximum(focus[:,:,:-1,:],focus[:,:,1:,:])
        return mean_weighted(da.abs(),valid)
    grad=(face_gradient_error(pred[:,1:3],true[:,1:3],-1)+
          face_gradient_error(pred[:,1:3],true[:,1:3],-2))/2
    pgrad=(face_gradient_error(pred[:,:1],true[:,:1],-1)+
           face_gradient_error(pred[:,:1],true[:,:1],-2))/2
    throatgrad=(face_gradient_error(pred[:,1:3],true[:,1:3],-1,throat)+
                face_gradient_error(pred[:,1:3],true[:,1:3],-2,throat))/2
    # Velocity integrated along every vertical cross section. Comparison is a
    # *cell-centre flux proxy*, not an OpenFOAM face-phi conservation test.
    qtrue=(true[:,1:2]*m).sum(dim=2)
    qpred=(pred[:,1:2]*m).sum(dim=2)
    qscale=(true[:,1:2].abs()*m).sum(dim=(2,3),keepdim=False).unsqueeze(-1)/m.shape[-1]
    flux=F.smooth_l1_loss((qpred-qtrue)/qscale.clamp_min(.1),torch.zeros_like(qtrue),beta=.25)
    # Interior cells with all four fluid neighbours only (does not assert wall
    # boundary flux conservation or substitute for actual OpenFOAM continuity).
    support=m[:,:,1:-1,1:-1]*m[:,:,1:-1,:-2]*m[:,:,1:-1,2:]*m[:,:,:-2,1:-1]*m[:,:,2:,1:-1]
    def div(v):
        return ((v[:,0:1,1:-1,2:]-v[:,0:1,1:-1,:-2])+
                (v[:,1:2,2:,1:-1]-v[:,1:2,:-2,1:-1]))*.5
    div_error=mean_weighted((div(pred[:,1:3])-div(true[:,1:3])).square(),support)
    # Ignore nearly stationary cells for unit-vector direction metric.
    truthmag=torch.linalg.vector_norm(true[:,1:3],dim=1,keepdim=True)
    predmag=torch.linalg.vector_norm(pred[:,1:3],dim=1,keepdim=True)
    cosine=(pred[:,1:3]*true[:,1:3]).sum(dim=1,keepdim=True)/(predmag*truthmag).clamp_min(1e-5)
    direction=mean_weighted((1-cosine.clamp(-1,1)),m*(truthmag>.05).to(m.dtype))
    return {'pressure':p,'velocity':v,'throat_velocity':local,
            'velocity_gradient':grad,'pressure_gradient':pgrad,
            'throat_gradient':throatgrad,'cross_section_flux':flux,
            'interior_divergence':div_error,'velocity_direction':direction}


def loss(pred,true,x,return_components=False):
    terms=components(pred,true,x)
    weighted=(c.LOSS_W_P*terms['pressure']+c.LOSS_W_U*terms['velocity']+
              c.LOSS_W_GRAD*terms['velocity_gradient']+c.LOSS_W_PGRAD*terms['pressure_gradient']+
              c.LOSS_W_THROAT*(terms['throat_velocity']+.5*terms['throat_gradient'])+
              c.LOSS_W_FLUX*terms['cross_section_flux']+
              c.LOSS_W_DIV*terms['interior_divergence']+
              c.LOSS_W_DIRECTION*terms['velocity_direction'])
    return (weighted,terms) if return_components else weighted
