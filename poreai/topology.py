"""Geometry-only, exactly connected four-neighbour fluid-cell message passing."""
import torch
from torch import nn
import torch.nn.functional as F


def hydraulic_edges(x):
    if x.ndim!=4 or x.shape[1]!=12:raise ValueError('12 geometry channels are required')
    m=(x[:,0:1]>.5).to(x.dtype)
    clearance=(.25+x[:,11:12].clamp(0,4)) / 4.25
    right=torch.zeros_like(m);down=torch.zeros_like(m)
    right[:,:,:,:-1]=m[:,:,:,:-1]*m[:,:,:,1:]*torch.minimum(clearance[:,:,:,:-1],clearance[:,:,:,1:]).pow(3)
    down[:,:,:-1,:]=m[:,:,:-1,:]*m[:,:,1:,:]*torch.minimum(clearance[:,:,:-1,:],clearance[:,:,1:,:]).pow(3)
    return right,down


def neighbour_average(z,edges):
    r,d=edges
    if z.shape[-2:]!=r.shape[-2:]:raise ValueError('Edge and node grids differ')
    sums=torch.zeros_like(z)
    sums[:,:,:,:-1]+=z[:,:,:,1:]*r[:,:,:,:-1]
    sums[:,:,:,1:]+=z[:,:,:,:-1]*r[:,:,:,:-1]
    sums[:,:,:-1,:]+=z[:,:,1:,:]*d[:,:,:-1,:]
    sums[:,:,1:,:]+=z[:,:,:-1,:]*d[:,:,:-1,:]
    weights=r+F.pad(r[:,:,:,:-1],(1,0))+d+F.pad(d[:,:,:-1,:],(0,0,1,0))
    return torch.where(weights>0,sums/weights.clamp_min(1e-12),z)


class GraphBlock(nn.Module):
    def __init__(self,width):
        super().__init__()
        self.net=nn.Sequential(nn.Conv2d(width*2,width,1),nn.GroupNorm(min(8,width),width),
                               nn.SiLU(),nn.Conv2d(width,width,1))
        self.scale=nn.Parameter(torch.tensor(.12))
    def forward(self,z,edges,m):
        v=neighbour_average(z,edges)
        return (z+self.scale.tanh()*self.net(torch.cat((z,v-z),1)))*m


class HydraulicGraphPath(nn.Module):
    def __init__(self,base=32,depth=6):
        super().__init__()
        self.stem=nn.Conv2d(12,base,1)
        self.blocks=nn.ModuleList(GraphBlock(base) for _ in range(depth))
        self.importance=nn.Conv2d(base,1,1)
    def forward(self,x,explain=False):
        m=(x[:,0:1]>.5).to(x.dtype)
        edges=hydraulic_edges(x)
        z=F.silu(self.stem(x))*m
        for layer in self.blocks:z=layer(z,edges,m)
        if explain:return z,torch.sigmoid(self.importance(z))*m,edges
        return z
