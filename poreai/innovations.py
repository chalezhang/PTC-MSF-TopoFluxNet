"""Pore-throat conditioning, cross-scale selective fusion, dense skip reuse."""
import torch
from torch import nn
import torch.nn.functional as F
from poreai.model import ResidualBlock


class PoreThroatConditionedAttention(nn.Module):
    """Geometry-only throat/clearance/harmonic-gradient conditions gate features.

    Residual gate 1+λA preserves information from large fluid pores. Attention
    indicates network allocation, *not* a causal physical attribution.
    """
    def __init__(self,channels,condition_channels=6):
        super().__init__()
        hidden=max(8,channels//4)
        self.feature=nn.Conv2d(channels,hidden,1)
        self.geometry=nn.Conv2d(condition_channels,hidden,1)
        self.gate=nn.Conv2d(hidden,1,1)
        self.strength=nn.Parameter(torch.tensor(.12))
    def forward(self,feature,condition,mask,explain=False):
        cond=F.interpolate(condition,size=feature.shape[-2:],mode='bilinear',align_corners=False)
        support=F.interpolate(mask,size=feature.shape[-2:],mode='nearest')
        attn=torch.sigmoid(self.gate(F.silu(self.feature(feature)+self.geometry(cond))))*support
        result=feature*(1+self.strength.tanh()*attn)
        return (result,attn) if explain else result


class ThreeWayCrossScaleFusion(nn.Module):
    """Learnable spatial softmax between decoder, same-scale, and extra-scale features."""
    def __init__(self,skip_ch,up_ch,extra_ch,out_ch):
        super().__init__()
        self.transforms=nn.ModuleList([nn.Conv2d(ch,out_ch,1) for ch in (skip_ch,up_ch,extra_ch)])
        self.gate=nn.Conv2d(out_ch*3+1,3,1)
        self.refine=ResidualBlock(out_ch,out_ch)
    def forward(self,skip,up,extra,throat,explain=False):
        size=skip.shape[-2:]
        vs=[skip,up,extra]
        vs=[layer(F.interpolate(v,size=size,mode='bilinear',align_corners=False) if v.shape[-2:]!=size else v)
            for layer,v in zip(self.transforms,vs)]
        t=F.interpolate(throat,size=size,mode='bilinear',align_corners=False)
        weights=torch.softmax(self.gate(torch.cat([*vs,t],1)),dim=1)
        fused=sum(weights[:,j:j+1]*v for j,v in enumerate(vs))
        output=self.refine(fused)
        return (output,weights) if explain else output


class ConditionalDenseBridge(nn.Module):
    """Selective dense information from full-res local encoder and coarser decoder."""
    def __init__(self,channels):
        super().__init__()
        self.gate=nn.Sequential(nn.Conv2d(channels*3+1,channels,1),nn.Sigmoid())
        self.mix=nn.Conv2d(channels*3,channels,1)
        self.strength=nn.Parameter(torch.tensor(.1))
    def forward(self,current,local,encoder,throat,mask):
        data=torch.cat([current,local,encoder],1)
        weight=self.gate(torch.cat([data,throat],1))
        return (current+self.strength.tanh()*weight*self.mix(data))*mask
