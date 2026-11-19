import numpy as np
import config as c


def _l2(a,b):
    a=np.asarray(a,dtype=np.float64);b=np.asarray(b,dtype=np.float64)
    denom=np.linalg.norm(b.ravel())
    return float(np.linalg.norm((a-b).ravel())/denom) if denom>1e-20 else None


def _r2(a,b):
    a=np.asarray(a,dtype=np.float64);b=np.asarray(b,dtype=np.float64)
    denom=np.square(b-b.mean()).sum()
    return float(1-np.square(a-b).sum()/denom) if denom>1e-30 else None


def _rmse(err):
    return float(np.sqrt(np.mean(np.square(err)))) if np.size(err) else None


def _meanabs(err):
    return float(np.mean(np.abs(err))) if np.size(err) else None


def _edge_gradient_errors(a,b,mask,channel,dx,dy):
    estimates=[];refs=[]
    for axis,spacing in ((1,dx),(0,dy)):
        both=(np.take(mask,np.arange(mask.shape[axis]-1),axis=axis)&
              np.take(mask,np.arange(1,mask.shape[axis]),axis=axis))
        estimates.append((np.diff(a[channel],axis=axis)/spacing)[both])
        refs.append((np.diff(b[channel],axis=axis)/spacing)[both])
    e=np.concatenate(estimates);t=np.concatenate(refs)
    return _l2(e,t),_rmse(e-t)


def case_metrics(pred,true,x):
    """Both fields are in normalized (p,Ux,Uy) units; convert to physical outputs."""
    pred=np.asarray(pred,dtype=np.float64)
    true=np.asarray(true,dtype=np.float64)
    x=np.asarray(x,dtype=np.float64)
    if pred.shape!=true.shape or pred.shape[0]!=3 or x.shape[1:]!=true.shape[1:]:
        raise ValueError('Model output/reference dimension mismatch')
    if not (np.isfinite(pred).all() and np.isfinite(true).all()):
        raise ValueError('Non-finite model prediction/CFD reference')
    mask=x[0]>.5
    if not mask.any():raise ValueError('No fluid cells')
    deltaP=c.pressure_scale();Uscale=c.velocity_scale()
    pp=pred[0]*deltaP;pt=true[0]*deltaP
    up=pred[1:3]*Uscale;ut=true[1:3]*Uscale
    ep=(pp-pt)[mask]
    eu=(up-ut)[:,mask]
    norm_e=np.sqrt(np.sum(eu**2,axis=0))
    upmag=np.linalg.norm(up[:,mask],axis=0)
    utmag=np.linalg.norm(ut[:,mask],axis=0)
    reference_flow=np.mean(utmag)
    # Raw pressure error matters for fixed inlet/outlet pressure BC, but also
    # report a gauge-aligned variant separately for spatial-pattern diagnosis.
    error_gauge=ep-ep.mean()
    result={
        'fluid_fraction':float(mask.mean()),
        'pressure_rmse_m2_s2':_rmse(ep),
        'pressure_mae_m2_s2':_meanabs(ep),
        'pressure_rmse_over_deltaP':_rmse(ep)/deltaP,
        'pressure_gauge_aligned_rmse_over_deltaP':_rmse(error_gauge)/deltaP,
        'pressure_relative_l2':_l2(pp[mask],pt[mask]),
        'pressure_r2':_r2(pp[mask],pt[mask]),
        'velocity_vector_rmse_m_s':_rmse(norm_e),
        'velocity_vector_mae_m_s':_meanabs(norm_e),
        'velocity_rmse_over_scale':_rmse(norm_e)/Uscale,
        'velocity_relative_l2':_l2(up[:,mask],ut[:,mask]),
        'velocity_r2':_r2(up[:,mask],ut[:,mask]),
        'velocity_magnitude_r2':_r2(upmag,utmag),
        'speed_mae_m_s':_meanabs(upmag-utmag),
        'velocity_magnitude_mean_true_m_s':float(reference_flow),
        'velocity_magnitude_mean_pred_m_s':float(np.mean(upmag)),
        'velocity_ux_relative_l2':_l2(up[0,mask],ut[0,mask]),
        'velocity_uy_relative_l2':_l2(up[1,mask],ut[1,mask]),
        'pressure_prior_relative_l2':_l2((x[5]*deltaP)[mask],pt[mask]),
        'zero_velocity_baseline_relative_l2':1.0 if np.linalg.norm(ut[:,mask])>1e-20 else None,
        'throat_fraction_of_fluid':float(np.mean(x[7,mask]>.1)),
    }
    valid_velocity=utmag>max(Uscale*.05,1e-14)
    if valid_velocity.any():
        cos=np.sum(up[:,mask][:,valid_velocity]*ut[:,mask][:,valid_velocity],axis=0)/np.maximum(
            upmag[valid_velocity]*utmag[valid_velocity],1e-20)
        result['velocity_cosine_mean']=float(np.mean(np.clip(cos,-1,1)))
    else:result['velocity_cosine_mean']=None
    for name,sel in [('throat',(x[7]>.1)&mask),
                     ('near_wall',(x[1]<.26)&mask),
                     ('wide_open',(x[11]>=.8)&(x[7]<.1)&mask)]:
        result[f'{name}_fluid_cells']=int(np.sum(sel))
        result[f'{name}_velocity_rmse_m_s']=_rmse(np.linalg.norm((up-ut)[:,sel],axis=0))
        result[f'{name}_velocity_relative_l2']=_l2(up[:,sel],ut[:,sel]) if np.any(sel) else None
    dx=c.LX/c.NX;dy=c.LY/c.NY
    p_grad_l2,p_grad_rmse=_edge_gradient_errors(np.stack([pp]),np.stack([pt]),mask,0,dx,dy)
    result['pressure_gradient_relative_l2']=p_grad_l2
    result['pressure_gradient_rmse_units_per_m']=p_grad_rmse
    grad_e=[];grad_t=[]
    for channel in (0,1):
        rel,rmse=_edge_gradient_errors(up,ut,mask,channel,dx,dy)
        if rel is not None:grad_e.append(rel)
        if rmse is not None:grad_t.append(rmse)
    result['velocity_gradient_mean_component_relative_l2']=float(np.mean(grad_e)) if grad_e else None
    result['velocity_gradient_mean_component_rmse_s_inv']=float(np.mean(grad_t)) if grad_t else None
    # Vertical cross-section flow proxy, using cell-centre Ux and geometric cell area.
    area=dy*c.THICKNESS
    qtrue=(ut[0]*mask).sum(axis=0)*area
    qpred=(up[0]*mask).sum(axis=0)*area
    result['cross_section_flux_proxy_relative_l2']=_l2(qpred,qtrue)
    result['cross_section_flux_proxy_mae_m3_s']=_meanabs(qpred-qtrue)
    result['cross_section_flux_proxy_reference_mean_m3_s']=float(np.mean(qtrue))
    result['cross_section_flux_proxy_prediction_mean_m3_s']=float(np.mean(qpred))
    # Divergence defined only for centres with 4 fluid neighbours; absolute
    # dimensional values are not full control-volume conservative face fluxes.
    valid=(mask[1:-1,1:-1]&mask[1:-1,2:]&mask[1:-1,:-2]&
           mask[2:,1:-1]&mask[:-2,1:-1])
    def div(u):
        return ((u[0,1:-1,2:]-u[0,1:-1,:-2])/(2*dx)+
                (u[1,2:,1:-1]-u[1,:-2,1:-1])/(2*dy))[valid]
    dp=div(up);dt=div(ut)
    result['interior_divergence_rmse_s_inv']=_rmse(dp)
    result['interior_divergence_reference_rmse_s_inv']=_rmse(dt)
    result['interior_divergence_error_rmse_s_inv']=_rmse(dp-dt)
    prior=result['pressure_prior_relative_l2']
    current=result['pressure_relative_l2']
    result['pressure_improvement_over_harmonic_prior_pct']=(100*(1-current/prior)
        if prior is not None and current is not None and prior>1e-12 else None)
    rel=result['velocity_relative_l2']
    result['velocity_improvement_over_zero_baseline_pct']=(100*(1-rel)
        if rel is not None else None)
    return result


def summarise_cases(records,bootstrap=2000):
    if not records:raise ValueError('No evaluated geometries')
    rng=np.random.default_rng(2027)
    numeric=sorted({k for record in records for k,v in record.items()
                    if isinstance(v,(float,int,np.floating,np.integer)) and
                    not isinstance(v,(bool,np.bool_))})
    output={'geometries':len(records),'bootstrap_resamples':bootstrap,
            'statistics':{}}
    for k in numeric:
        vals=np.asarray([r[k] for r in records if r.get(k) is not None and np.isfinite(r[k])],dtype=float)
        if vals.size==0:continue
        item={'n':int(len(vals)),'mean':float(np.mean(vals)),
              'median':float(np.median(vals)),'min':float(np.min(vals)),
              'max':float(np.max(vals))}
        if vals.size>1 and bootstrap>0:
            boot=vals[rng.integers(0,len(vals),(bootstrap,len(vals)))].mean(axis=1)
            item['bootstrap_95ci_mean']=[float(v) for v in np.percentile(boot,[2.5,97.5])]
        else:item['bootstrap_95ci_mean']=None
        output['statistics'][k]=item
    return output
