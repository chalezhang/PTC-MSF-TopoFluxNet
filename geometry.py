from __future__ import annotations
from dataclasses import dataclass
from math import pi
import numpy as np
from scipy.ndimage import label

STRUCT_4 = np.array([[0,1,0],[1,1,1],[0,1,0]],dtype=np.uint8)

@dataclass
class Geometry:
    material_void: np.ndarray
    cfd_fluid: np.ndarray
    isolated_void: np.ndarray
    diagnostics: dict


def _rasterize_one(mask,cx,cy,r):
    ny,nx=mask.shape
    x0=max(0,int(np.floor(cx-r)));x1=min(nx,int(np.ceil(cx+r))+1)
    y0=max(0,int(np.floor(cy-r)));y1=min(ny,int(np.ceil(cy+r))+1)
    if x1<=x0 or y1<=y0:return 0
    yy,xx=np.ogrid[y0:y1,x0:x1]
    patch=mask[y0:y1,x0:x1]
    before=int(patch.sum())
    patch |= ((xx+.5-cx)**2+(yy+.5-cy)**2<=r*r)
    return int(patch.sum())-before


def _candidate_circles(nx,ny,seed,rmin,rmax,gap,max_attempts,allow_overlap,overlap_ratio,max_nominal_factor=11.):
    """随机圆心/随机半径采样，支持有限度重叠。"""
    rng=np.random.default_rng(seed)
    circles=[]; area=0.; rejected=0
    max_area=max_nominal_factor*nx*ny
    fail_streak=0
    bins={}
    step=float(max(rmax+gap,1.0))
    min_center_factor=(1.0-float(overlap_ratio)) if allow_overlap else 1.0
    for _ in range(max_attempts):
        if area>=max_area:break
        r=float(rng.uniform(rmin,rmax))
        cx=float(rng.uniform(-0.5*r,nx+0.5*r));cy=float(rng.uniform(-0.5*r,ny+0.5*r))
        key=(int(cx/step),int(cy/step))
        valid=True
        for ix in range(key[0]-2,key[0]+3):
            for iy in range(key[1]-2,key[1]+3):
                for px,py,pr in bins.get((ix,iy),()):
                    required=max(0.0,min_center_factor*(r+pr)+gap)
                    if (cx-px)**2+(cy-py)**2 < required*required:
                        valid=False;break
                if not valid:break
            if not valid:break
        if valid:
            circle=(cx,cy,r)
            circles.append(circle);area+=pi*r*r;fail_streak=0
            bins.setdefault(key,[]).append(circle)
        else:
            rejected+=1;fail_streak+=1
            if fail_streak>7000:break
    return circles,rejected


def _rasterize_prefix(circles,nx,ny,n):
    mask=np.zeros((ny,nx),bool)
    for cx,cy,r in circles[:n]:
        _rasterize_one(mask,cx,cy,r)
    return mask


def random_circle_mask(nx,ny,seed,porosity=.12,r_min=3.,r_max=10.,gap=0.,
                       allow_overlap=True,overlap_ratio=.35,tol=.005,soft_tol=.015,
                       max_iter=22,max_attempts=12000,circle_phase='solid'):
    """返回 (solid, circles, actual_porosity, bisection_steps, abs_error, rejected, soft_accepted)。"""
    if nx<8 or ny<8 or not 0<porosity<1:raise ValueError('无效尺寸或孔隙率')
    if not 0<r_min<=r_max<min(nx,ny)/2:raise ValueError('圆半径范围无效')
    if circle_phase not in ('solid','void'):raise ValueError('circle_phase 必须为 solid 或 void')
    if gap<0 or max_iter<1 or max_attempts<1 or tol<=0 or soft_tol<tol:raise ValueError('圆生成参数无效')
    candidates,rejected=_candidate_circles(nx,ny,seed,r_min,r_max,gap,max_attempts,allow_overlap,overlap_ratio)
    if not candidates:raise RuntimeError('没有可放置的圆，检查半径和间隙参数')
    target_coverage=(1-porosity) if circle_phase=='solid' else porosity
    cumulative=np.zeros(len(candidates)+1,dtype=np.int64)
    scan=np.zeros((ny,nx),dtype=bool)
    for i,(cx,cy,r) in enumerate(candidates,1):
        cumulative[i]=cumulative[i-1]+_rasterize_one(scan,cx,cy,r)
    target=target_coverage*nx*ny
    lo,hi=0,len(candidates)
    iters=0
    for it in range(max_iter):
        mid=(lo+hi)//2;iters=it+1
        if cumulative[mid] < target:lo=mid+1
        else:hi=mid
        if lo>=hi:break
    search_lo=max(0,lo-4); search_hi=min(len(candidates),lo+4)
    n=min(range(search_lo,search_hi+1), key=lambda k:abs(int(cumulative[k])-target))
    filled=_rasterize_prefix(candidates,nx,ny,n)
    err=abs(float(filled.mean())-target_coverage)
    solid=filled if circle_phase=='solid' else ~filled
    actual=float((~solid).mean())
    soft_accepted=bool(err>tol and err<=soft_tol)
    if err>soft_tol:
        raise RuntimeError(f'目标孔隙率 {porosity:.4f} 无法在当前圆形约束下达到；'
                           f'最接近 {actual:.4f}（误差 {err:.4f}），'
                           f'已接受 {n} 圆，拒绝 {rejected} 圆。'
                           '尝试减小 gap、增大允许重叠比例、缩小 r_min，或调整目标孔隙率。')
    return solid,candidates[:n],actual,iters,err,rejected,soft_accepted


def _local_metrics(mask):
    ys=np.array_split(np.arange(mask.shape[0]),4)
    xs=np.array_split(np.arange(mask.shape[1]),4)
    tile=np.array([[mask[np.ix_(y,x)].mean() for x in xs] for y in ys],float)
    return {'tile_porosity_4x4':np.round(tile,6).tolist(),
            'min_tile_porosity':float(tile.min()),'max_tile_porosity':float(tile.max()),
            'std_tile_porosity':float(tile.std())}


def _classify(void):
    comps,n=label(void,structure=STRUCT_4)
    inlet=set(np.unique(comps[:,0]).tolist())-{0}
    outlet=set(np.unique(comps[:,-1]).tolist())-{0}
    spanning=inlet & outlet
    if spanning:
        sizes=np.bincount(comps.ravel())
        cid=max(spanning,key=lambda i:sizes[i])
        flow=comps==cid
    else:
        flow=np.zeros_like(void)
    isolated=void & ~flow
    return flow,isolated,int(n),len(spanning)


def _mesh_corner_cleanup(raw_flow):
    work=raw_flow.copy(); removed=0
    for _ in range(12):
        a=work[:-1,:-1];b=work[:-1,1:];c=work[1:,:-1];d=work[1:,1:]
        pattern=(a&d&~b&~c)|(~a&~d&b&c)
        ys,xs=np.where(pattern)
        if not len(xs):break
        for y,x in zip(ys,xs):
            if work[y,x] and work[y+1,x+1] and not work[y,x+1] and not work[y+1,x]:
                work[y+1,x+1]=False;removed+=1
            elif work[y,x+1] and work[y+1,x] and not work[y,x] and not work[y+1,x+1]:
                work[y+1,x]=False;removed+=1
    return work,int(removed)



def count_underconstrained_cells(mask):
    """预估 Foundation OpenFOAM 13 的二维 cell determinant=0 单元数量。

    二维 voxel 的面法向仅沿 x/y，cellDeterminant 只统计内部面；
    每个单元至少需要一个 x 向内部面、一个 y 向内部面，
    否则面积张量退化，checkMesh -allGeometry 将报 underdeterminedCells。
    """
    mask=np.asarray(mask,dtype=bool)
    horizontal=np.zeros_like(mask)
    horizontal[:,1:] |= mask[:,:-1]
    horizontal[:,:-1] |= mask[:,1:]
    vertical=np.zeros_like(mask)
    vertical[1:,:] |= mask[:-1,:]
    vertical[:-1,:] |= mask[1:,:]
    return mask & ~(horizontal & vertical)


def _prune_underdetermined_cells(raw_flow):
    """仅从 CFD 掩膜移除二维行列式退化的单元；材料原图完全保持不变。

    删除后会影响相邻单元，需反复迭代到固定点，
    同时保持唯一的自然左右贯通连通域；绝不通过填孔/连线制造人工通路。
    """
    flow=np.asarray(raw_flow,dtype=bool).copy()
    original=int(flow.sum())
    if original==0:
        return flow,{'cfd_rank_removed_cells':0,'cfd_rank_iterations':0,
                     'cfd_rank_defects_before':0,'cfd_rank_defects_after':0}
    initial_bad=int(count_underconstrained_cells(flow).sum())
    loops=0
    while True:
        loops+=1
        proposed=flow & ~count_underconstrained_cells(flow)
        proposed,_=_mesh_corner_cleanup(proposed)
        proposed,_,_,_=_classify(proposed)
        if np.array_equal(proposed,flow):
            break
        if int(proposed.sum()) >= int(flow.sum()):
            raise RuntimeError('二维网格删减算法出现非单调变化，已停止')
        flow=proposed
        if loops>original:
            raise RuntimeError('二维网格秩清理未在规定次数内收敛')
    after_bad=int(count_underconstrained_cells(flow).sum())
    return flow,{'cfd_rank_removed_cells':original-int(flow.sum()),
                 'cfd_rank_iterations':loops,'cfd_rank_defects_before':initial_bad,
                 'cfd_rank_defects_after':after_bad}


def generate(nx,ny,seed,porosity=.12,feature_size=.001,domain=(.04,.08),
             isolated_share=None,site_pitch=None, r_min=3.,r_max=10., gap=0.,
             allow_overlap=True,overlap_ratio=.35,tol=.005,soft_tol=.015,max_iter=22,max_attempts=12000,
             circle_phase='solid',require_percolation=False,
             max_cfd_removed_fraction=0.10,**_):
    if nx<32 or ny<32 or nx%8 or ny%8:raise ValueError('网格必须>=32且为8的倍数')
    solid,circles,actual,iters,err,rejected,soft_accepted=random_circle_mask(
        nx,ny,seed,porosity,r_min,r_max,gap,allow_overlap,overlap_ratio,tol,soft_tol,max_iter,max_attempts,circle_phase)
    material=~solid
    natural_flow,isolated,ncomps,nspan=_classify(material)
    cleaned,corner_removed=_mesh_corner_cleanup(natural_flow)
    flow,_,_,_=_classify(cleaned)
    flow,rank_diagnostics=_prune_underdetermined_cells(flow)
    excluded_cfd=int(natural_flow.sum()-flow.sum())
    # 不允许靠删掉大部分流体“修好”网格。超过上限由套件生成器换随机种子。
    loss_fraction=excluded_cfd/max(1,int(natural_flow.sum()))
    ready=bool(nspan and flow.any() and flow[:,0].any() and flow[:,-1].any()
               and loss_fraction<=max_cfd_removed_fraction and rank_diagnostics['cfd_rank_defects_after']==0)
    if require_percolation and not ready:
        raise RuntimeError(f'seed={seed}: 自然圆形几何没有左右贯通流道；不能人为添加孔喉。'
                           '请提高孔隙率或改变圆的尺寸、密度/位置。')
    diag={'seed':int(seed),'nx':nx,'ny':ny,'total_porosity':float(actual),
          'connected_porosity':float(flow.mean()),'isolated_porosity':float(isolated.mean()),
          'isolated_fraction_of_pores':float(isolated.sum()/max(material.sum(),1)),
          'n_random_circles':len(circles),'n_throat_edges':0,'rejected_circles':int(rejected),
          'porosity_error':float(err),'calibration_iterations':iters,
          'min_radius':float(min((c[2] for c in circles),default=0.)),
          'max_radius':float(max((c[2] for c in circles),default=0.)),
          'target_total_porosity':float(porosity),'n_void_components':int(ncomps),
          'n_spanning_components':int(nspan),'cfd_ready':ready,
          'natural_spanning_porosity':float(natural_flow.mean()),
          'cfd_excluded_corner_pixels':excluded_cfd,
          'cfd_removed_fraction':float(loss_fraction),
          'max_cfd_removed_fraction':float(max_cfd_removed_fraction),
          **rank_diagnostics,
          'corner_contact_deletions':int(corner_removed),
          'inlet_cells':int(flow[:,0].sum()),'outlet_cells':int(flow[:,-1].sum()),
          'allow_overlap':bool(allow_overlap),'overlap_ratio':float(overlap_ratio),'gap_px':float(gap),
          'circle_phase':circle_phase,
          'porosity_soft_accepted':bool(soft_accepted),'porosity_soft_tolerance':float(soft_tol),
          **_local_metrics(material)}
    return Geometry(material,flow,isolated,diag)
