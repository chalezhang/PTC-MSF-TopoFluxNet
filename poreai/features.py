"""Geometry-only (no label leakage) physics features and dimensional targets."""
import json
from pathlib import Path
import numpy as np
from scipy.ndimage import distance_transform_edt,label,binary_dilation,maximum_filter
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

import config as c
from poreai.fields import field_to_grid


def harmonic_pressure(mask):
    """Solve graph Laplace pressure with left/right Dirichlet and solid Neumann faces.

    Dimensionless pressure (inlet=1, outlet=0). Used as a geometry-only feature
    and base prediction, without viewing any training or testing CFD labels.
    Fluid components not connected to either open face receive a linear fallback.
    """
    mask=np.asarray(mask,dtype=bool)
    if mask.ndim!=2 or not mask.any():raise ValueError('Empty hydraulic domain')
    ny,nx=mask.shape
    ids=np.full(mask.shape,-1,dtype=np.int32)
    ids[mask]=np.arange(mask.sum(),dtype=np.int32)
    nn=int(mask.sum());rr=[];cc=[];vv=[];b=np.zeros(nn,dtype=np.float64)
    comps,ncomp=label(mask)
    detached=[]
    for comp in range(1,ncomp+1):
        sel=comps==comp
        if not (sel[:,0].any() or sel[:,-1].any()):
            detached.append(int(ids[sel][0]))
    for j,i in np.argwhere(mask):
        cur=int(ids[j,i]);degree=0
        for jj,ii in ((j-1,i),(j+1,i),(j,i-1),(j,i+1)):
            if 0<=jj<ny and 0<=ii<nx and mask[jj,ii]:
                rr.append(cur);cc.append(int(ids[jj,ii]));vv.append(-1.)
                degree+=1
        if i==0:
            degree+=2;b[cur]+=2
        if i==nx-1:degree+=2
        if cur in detached:
            degree+=1;b[cur]+=1.-(i+.5)/nx
        rr.append(cur);cc.append(cur);vv.append(float(degree))
    A=coo_matrix((vv,(rr,cc)),shape=(nn,nn)).tocsr()
    values=spsolve(A,b)
    if not np.isfinite(values).all():raise ValueError('Graph harmonic pressure failed')
    grid=np.zeros(mask.shape,dtype=np.float64);grid[mask]=values
    return np.clip(grid,0.,1.).astype(np.float32)


def throat_geometry(mask,harmonic=None):
    """Pure geometry-based flow-through bottleneck proxy.

    The 4-connected component must touch both inlet and outlet. A medial-axis
    ridge with low local clearance is dilated to include the throat neighbourhood.
    This is a *proxy*, not a hydraulic constriction proved by Navier-Stokes.
    It never consumes p/U target labels or any validation/test output.
    """
    mask=np.asarray(mask,dtype=bool)
    if mask.ndim!=2 or not mask.any():raise ValueError('Empty 2D mask')
    components,n=label(mask)
    through=np.zeros_like(mask)
    for i in range(1,n+1):
        comp=components==i
        if np.any(comp[:,0]) and np.any(comp[:,-1]):through|=comp
    # EDT extends the inlet/outlet so those open faces are not counted as walls.
    e=np.pad(mask,((1,1),(0,0)),mode='constant')
    e=np.pad(e,((0,0),(8,8)),mode='edge')
    dist=distance_transform_edt(e)[1:-1,8:-8]
    # Choose the local throat cross-section perpendicular to the geometry-only
    # harmonic driving direction. Unlike an unconditional x/y ridge union,
    # this does not mark every near-wall row of a wide straight channel.
    if harmonic is None:harmonic=harmonic_pressure(mask)
    py,px=np.gradient(np.asarray(harmonic,dtype=np.float64))
    flow_along_x=np.abs(px)>=np.abs(py)
    ridge_y=dist>=maximum_filter(dist,size=(3,1),mode='nearest')-1e-6
    ridge_x=dist>=maximum_filter(dist,size=(1,3),mode='nearest')-1e-6
    ridge=((ridge_y&flow_along_x)|(ridge_x&~flow_along_x)) & \
          (dist<=c.THROAT_MAX_RADIUS)&through
    focus=binary_dilation(ridge,iterations=2)&through
    # Weight the narrowest passable cells; saturate at 1 for stable training.
    weights=np.clip((c.THROAT_MAX_RADIUS+.5-dist)/max(c.THROAT_MAX_RADIUS,1.),0.,1.)*focus
    return weights.astype(np.float32),through.astype(np.float32),dist.astype(np.float32)


def throat_score(mask):
    """Higher scores contain more narrow flow-through medial-axis pixels."""
    t,through,_=throat_geometry(mask)
    if not through.any():return 0.
    return float(np.sum(t)/np.sum(through))


def features(mask):
    mask=np.asarray(mask,dtype=bool)
    if mask.ndim!=2 or not mask.any():raise ValueError('Empty or non-2D CFD mask')
    ny,nx=mask.shape
    y,x=np.mgrid[:ny,:nx]
    # Left/right boundaries are open; top/bottom are walls. Extend lateral
    # fluid cells so EDT does not invent a solid inlet or outlet boundary.
    extended=np.pad(mask,((1,1),(0,0)),mode='constant',constant_values=False)
    extended=np.pad(extended,((0,0),(16,16)),mode='edge')
    dist=np.clip(distance_transform_edt(extended)[1:-1,16:-16]/8.0,0.,1.)
    xx=(x+0.5)/nx
    yy=(y+0.5)/ny
    porosity=np.full(mask.shape,mask.mean())
    ph=harmonic_pressure(mask)
    gy,gx=np.gradient(ph.astype(float))
    # Local pressure-gradient proxy; masked faces and obstacle boundaries can
    # create edge jumps, hence clipping to a finite fixed range.
    drive=np.clip(np.hypot(gx*nx,gy*ny),0.,4.)*mask
    throat,through,_=throat_geometry(mask,ph)
    # Extra directional harmonic features and untruncated clearance information.
    # All inputs are computed from geometry and specified boundary conditions only.
    dx=np.clip(-gx*nx,-4.,4.)*mask
    dy=np.clip(-gy*ny,-4.,4.)*mask
    clearance=np.clip(dist*8./max(c.THROAT_MAX_RADIUS,1e-6),0.,4.)*mask
    return np.stack([mask,dist,xx,yy,porosity,ph,drive,throat,through,
                     dx,dy,clearance],axis=0).astype(np.float32)


def target(p,U,mask):
    pg=field_to_grid(p,mask)
    ug=field_to_grid(U,mask)
    arr=np.stack([np.nan_to_num(pg)/c.pressure_scale(),
                  np.nan_to_num(ug[...,0])/c.velocity_scale(),
                  np.nan_to_num(ug[...,1])/c.velocity_scale()],axis=0)
    if not np.isfinite(arr).all():raise ValueError('Nonfinite target')
    return arr.astype(np.float32)


def physical_prediction(y,mask):
    mask=np.asarray(mask,dtype=bool)
    y=np.asarray(y,dtype=np.float64)
    if y.shape!=(3,*mask.shape) or not np.isfinite(y).all():
        raise ValueError('Invalid AI pressure/velocity shape or nonfinite values')
    p=y[0,mask]*c.pressure_scale()
    U=np.column_stack([y[1,mask]*c.velocity_scale(),y[2,mask]*c.velocity_scale(),np.zeros(int(mask.sum()))])
    if np.max(np.abs(y[0,mask]))>4.0 or np.max(np.abs(y[1:3,mask]))>8.01:
        raise ValueError('Out-of-range AI normalized field')
    return p,U


def case_features(case):
    case=Path(case)
    meta=json.loads((case/'geometry.json').read_text())
    mask=np.load(case/'mask.npy').astype(bool)
    if mask.shape!=(meta['grid'][1],meta['grid'][0]) or mask.sum()!=meta['n_cells']:
        raise ValueError('Mask mismatch with geometry metadata')
    return features(mask),mask,meta
