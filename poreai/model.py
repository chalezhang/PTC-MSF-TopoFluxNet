"""Plain U-Net and geometry-conditioned graph/multi-scale flow surrogate."""
import torch
from torch import nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self,cin,cout):
        super().__init__()
        self.seq=nn.Sequential(
            nn.Conv2d(cin,cout,3,padding=1),nn.GroupNorm(min(8,cout),cout),nn.SiLU(),
            nn.Conv2d(cout,cout,3,padding=1),nn.GroupNorm(min(8,cout),cout),nn.SiLU())
    def forward(self,x):return self.seq(x)


class ResidualBlock(nn.Module):
    def __init__(self,cin,cout):
        super().__init__()
        self.conv1=nn.Conv2d(cin,cout,3,padding=1)
        self.norm1=nn.GroupNorm(min(8,cout),cout)
        self.conv2=nn.Conv2d(cout,cout,3,padding=1)
        self.norm2=nn.GroupNorm(min(8,cout),cout)
        self.skip=nn.Conv2d(cin,cout,1) if cin!=cout else nn.Identity()
        self.act=nn.SiLU()
    def forward(self,x):
        v=self.act(self.norm1(self.conv1(x)))
        v=self.norm2(self.conv2(v))
        return self.act((v+self.skip(x))*0.70710678)


class SEBlock(nn.Module):
    def __init__(self,channels,reduction=8):
        super().__init__()
        hidden=max(channels//reduction,4)
        self.pool=nn.AdaptiveAvgPool2d(1)
        self.mlp=nn.Sequential(nn.Conv2d(channels,hidden,1),nn.SiLU(),nn.Conv2d(hidden,channels,1),nn.Sigmoid())
    def forward(self,x):return x*self.mlp(self.pool(x))


class GlobalContext(nn.Module):
    def __init__(self,channels):
        super().__init__()
        self.norm=nn.LayerNorm(channels)
        self.attn=nn.MultiheadAttention(channels,4,batch_first=True,dropout=0.)
        self.scale=nn.Parameter(torch.tensor(0.05))
    def forward(self,x):
        b,c,h,w=x.shape
        tokens=x.flatten(2).transpose(1,2)
        q=self.norm(tokens)
        a,_=self.attn(q,q,q,need_weights=False)
        return x+self.scale*a.transpose(1,2).reshape(b,c,h,w)


class LocalThroatBranch(nn.Module):
    """High-resolution branch emphasizing throat and wall-adjacent geometry cues."""
    def __init__(self,base):
        super().__init__()
        self.seq=nn.Sequential(
            nn.Conv2d(6,base,3,padding=1),nn.GroupNorm(min(8,base),base),nn.SiLU(),
            nn.Conv2d(base,base,3,padding=2,dilation=2),nn.GroupNorm(min(8,base),base),nn.SiLU(),
            SEBlock(base))
    def forward(self,x):
        focus=torch.cat([x[:,1:2],x[:,6:8],x[:,9:12]],dim=1)
        return self.seq(focus)


class PlainUNetPrior(nn.Module):
    def __init__(self,base=32,in_channels=12):
        super().__init__()
        if base<8 or base%8:raise ValueError('base must be a multiple of 8')
        if in_channels!=12:raise ValueError('Expected 12 channels')
        self.in_channels=in_channels
        self.enc1=ConvBlock(in_channels,base)
        self.enc2=ConvBlock(base,base*2)
        self.enc3=ConvBlock(base*2,base*4)
        self.enc4=ConvBlock(base*4,base*8)
        self.center=ConvBlock(base*8,base*8)
        self.dec4=ConvBlock(base*16,base*4)
        self.dec3=ConvBlock(base*8,base*2)
        self.dec2=ConvBlock(base*4,base)
        self.dec1=ConvBlock(base*2,base)
        self.p_head=nn.Conv2d(base,1,1)
        self.u_head=nn.Conv2d(base,2,1)
        nn.init.zeros_(self.p_head.weight);nn.init.zeros_(self.p_head.bias)
        nn.init.zeros_(self.u_head.weight);nn.init.zeros_(self.u_head.bias)
    @staticmethod
    def _pool(x):return F.avg_pool2d(x,2)
    @staticmethod
    def _up(x,size):return F.interpolate(x,size=size,mode='bilinear',align_corners=False)
    def forward(self,x):
        if x.ndim!=4 or x.shape[1]!=12 or min(x.shape[-2:])<16 or x.shape[-1]%16 or x.shape[-2]%16:
            raise ValueError('Expected N x 12 x H x W, H/W divisible by 16')
        m=x[:,0:1].clamp(0.,1.)
        prior=x[:,5:6]*m
        e1=self.enc1(x)
        e2=self.enc2(self._pool(e1))
        e3=self.enc3(self._pool(e2))
        e4=self.enc4(self._pool(e3))
        z=self.center(self._pool(e4))
        d4=self.dec4(torch.cat([self._up(z,e4.shape[-2:]),e4],dim=1))
        d3=self.dec3(torch.cat([self._up(d4,e3.shape[-2:]),e3],dim=1))
        d2=self.dec2(torch.cat([self._up(d3,e2.shape[-2:]),e2],dim=1))
        d1=self.dec1(torch.cat([self._up(d2,e1.shape[-2:]),e1],dim=1))
        p=(prior+self.p_head(d1))*m
        u=self.u_head(d1)*m
        return torch.cat([p,u],dim=1)


class TopoFluxNet(nn.Module):
    """Geometry-conditioned local/multi-scale/topology fusion surrogate.

    Outputs normalized kinematic pressure + 2D velocity, with a geometry-only
    harmonic-pressure residual. All learned gates are non-causal diagnostics.
    """
    def __init__(self,base=32,in_channels=12,graph_depth=6):
        super().__init__()
        if base<8 or base%8 or in_channels!=12:raise ValueError('Invalid model dimensions')
        from poreai.topology import HydraulicGraphPath
        from poreai.innovations import PoreThroatConditionedAttention,ThreeWayCrossScaleFusion,ConditionalDenseBridge
        self.local=LocalThroatBranch(base)
        self.graph=HydraulicGraphPath(base,depth=graph_depth)
        self.enc1=ResidualBlock(12,base)
        self.enc2=ResidualBlock(base,base*2)
        self.enc3=ResidualBlock(base*2,base*4)
        self.enc4=ResidualBlock(base*4,base*8)
        self.center=nn.Sequential(ResidualBlock(base*8,base*8),GlobalContext(base*8))
        self.condition=nn.ModuleList([PoreThroatConditionedAttention(c) for c in (base,base*2,base*4,base*8)])
        self.graph_gate=nn.Conv2d(base*3+1,base,1)
        self.graph_scale=nn.Parameter(torch.tensor(.2))
        self.fuse4=ThreeWayCrossScaleFusion(base*8,base*8,base*4,base*4)
        self.fuse3=ThreeWayCrossScaleFusion(base*4,base*4,base*2,base*2)
        self.fuse2=ThreeWayCrossScaleFusion(base*2,base*2,base,base)
        self.fuse1=ThreeWayCrossScaleFusion(base,base,base,base)
        self.dense=ConditionalDenseBridge(base)
        self.refine=ResidualBlock(base*2,base)
        self.p_head=nn.Sequential(ResidualBlock(base,base),nn.Conv2d(base,1,1))
        self.u_head=nn.Sequential(ResidualBlock(base,base),nn.Conv2d(base,2,1))
        nn.init.zeros_(self.p_head[-1].weight);nn.init.zeros_(self.p_head[-1].bias)
        nn.init.zeros_(self.u_head[-1].weight);nn.init.zeros_(self.u_head[-1].bias)

    @staticmethod
    def _pool(x):return F.avg_pool2d(x,2)

    def _forward(self,x,explain=False):
        if x.ndim!=4 or x.shape[1]!=12 or min(x.shape[-2:])<16 or x.shape[-1]%16 or x.shape[-2]%16:
            raise ValueError('Expected N x 12 x H x W divisible by 16')
        mask=(x[:,0:1]>.5).to(x.dtype)
        p0=x[:,5:6]*mask
        # Geometry-only: clearance, throat, connectivity, harmonic-driving vector.
        cond=torch.cat((x[:,1:2],x[:,6:9],x[:,9:11]),dim=1)
        throat=x[:,7:8]
        local=self.local(x)
        if explain:
            graph,importance,edges=self.graph(x,explain=True)
        else:graph=self.graph(x)
        e1=self.enc1(x)
        gate=torch.sigmoid(self.graph_gate(torch.cat((e1,local,graph,throat),1)))
        e1=e1+self.graph_scale.tanh()*gate*(local+graph)
        e1,a1=self.condition[0](e1,cond,mask,True)
        e2,a2=self.condition[1](self.enc2(self._pool(e1)),cond,mask,True)
        e3,a3=self.condition[2](self.enc3(self._pool(e2)),cond,mask,True)
        e4,a4=self.condition[3](self.enc4(self._pool(e3)),cond,mask,True)
        z=self.center(self._pool(e4))
        d4,w4=self.fuse4(e4,z,e3,throat,True)
        d3,w3=self.fuse3(e3,d4,e2,throat,True)
        d2,w2=self.fuse2(e2,d3,e1,throat,True)
        d1,w1=self.fuse1(e1,d2,local,throat,True)
        d1=self.dense(d1,local,e1,throat,mask)
        feat=self.refine(torch.cat((d1,local),1))
        out=torch.cat(((p0+self.p_head(feat))*mask,self.u_head(feat)*mask),1)
        if not explain:return out
        diagnostics={'graph_attention':importance,'graph_gate':gate.mean(1,keepdim=True)*mask,
                     'condition_attention':a1,'scale_weights':w1,
                     'throat_weight':throat*mask,'edge_right':edges[0],'edge_down':edges[1]}
        return out,diagnostics

    def forward(self,x):return self._forward(x)
    def explain(self,x):return self._forward(x,explain=True)


MODEL_REGISTRY={'plain_unet':PlainUNetPrior,'conditioned_multiscale_topoflux':TopoFluxNet}


def build_model(architecture,base=32,in_channels=12,graph_depth=6):
    if architecture not in MODEL_REGISTRY:raise ValueError(f'Unknown architecture: {architecture}')
    if architecture=='conditioned_multiscale_topoflux':
        return TopoFluxNet(base=base,in_channels=in_channels,graph_depth=graph_depth)
    return PlainUNetPrior(base=base,in_channels=in_channels)
