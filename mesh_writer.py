from pathlib import Path
import numpy as np

def _write_foam_list(path, cls, data):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w') as f:
        f.write(f'FoamFile\n{{ version 2.0; format ascii; class {cls}; object {path.name}; }}\n')
        f.write(f'{len(data)}\n(\n')
        for line in data:
            f.write(str(line)+'\n')
        f.write(')\n')


def write_mesh(case, mask, domain, thickness, dim):
    if dim!=2:
        raise ValueError('2D only')
    # 直接调用本函数时也不允许写出二维几何秩欠约束网格。
    from geometry import count_underconstrained_cells, STRUCT_4
    from scipy.ndimage import label
    if not isinstance(mask,np.ndarray) or mask.ndim!=2 or mask.dtype!=bool or not mask.any():
        raise ValueError('CFD 掩膜必须是非空二维 bool 数组')
    flawed=int(count_underconstrained_cells(mask).sum())
    if flawed:
        raise ValueError(f'CFD 网格有 {flawed} 个缺少二维 x/y 内部面支持的单元；'
                         '禁止绕过几何秩预检')
    if int(label(mask,structure=STRUCT_4)[1])!=1 or not (mask[:,0].any() and mask[:,-1].any()):
        raise ValueError('CFD 掩膜必须是唯一自然左右贯通流体区域')
    ny,nx=mask.shape; lx,ly=domain; lz=thickness
    verts=[]; vmap={}; face_keys={}; faces=[]; owner=[]; neighbour=[]; patches=[]; cells=[]
    def point(i,j,k):
        key=(i,j,k)
        if key not in vmap:
            vmap[key]=len(verts); verts.append((i*lx/nx, j*ly/ny, k*lz))
        return vmap[key]
    for y,x in np.argwhere(mask):
        idx=len(cells); cells.append((int(x),int(y),0))
        v000=point(x,y,0);v100=point(x+1,y,0);v110=point(x+1,y+1,0);v010=point(x,y+1,0)
        v001=point(x,y,1);v101=point(x+1,y,1);v111=point(x+1,y+1,1);v011=point(x,y+1,1)
        candidates=[([v000,v001,v011,v010],'inlet' if x==0 else 'walls'),
                    ([v100,v110,v111,v101],'outlet' if x==nx-1 else 'walls'),
                    ([v000,v100,v101,v001],'walls'),
                    ([v010,v011,v111,v110],'walls'),
                    ([v000,v010,v110,v100],'frontAndBack'),
                    ([v001,v101,v111,v011],'frontAndBack')]
        for v,tag in candidates:
            key=tuple(sorted(v))
            if key in face_keys:
                k=face_keys.pop(key)
                if neighbour[k]>=0: raise ValueError('nonmanifold face')
                neighbour[k]=idx; patches[k]=None
            else:
                face_keys[key]=len(faces)
                faces.append(v); owner.append(idx); neighbour.append(-1); patches.append(tag)
    internal=[i for i,t in enumerate(patches) if t is None]
    boundary_groups=['inlet','outlet','walls','frontAndBack']
    order=internal[:]
    for patch in boundary_groups:
        order += [i for i,t in enumerate(patches) if t==patch]
    poly=Path(case)/'constant/polyMesh'
    _write_foam_list(poly/'points','vectorField',[f'({x:.12g} {y:.12g} {z:.12g})' for x,y,z in verts])
    _write_foam_list(poly/'faces','faceList',[f'4({" ".join(str(v) for v in faces[i])})' for i in order])
    _write_foam_list(poly/'owner','labelList',[owner[i] for i in order])
    _write_foam_list(poly/'neighbour','labelList',[neighbour[i] for i in internal])
    lines=['FoamFile','{ version 2.0; format ascii; class polyBoundaryMesh; object boundary; }',str(len(boundary_groups)),'(']
    start=len(internal)
    for patch in boundary_groups:
        count=patches.count(patch)
        kind='patch' if patch in ('inlet','outlet') else 'empty' if patch=='frontAndBack' else 'wall'
        lines.extend([patch,'{',f'    type {kind};',f'    nFaces {count};',f'    startFace {start};','}'])
        start+=count
    lines.append(')')
    (poly/'boundary').write_text('\n'.join(lines)+'\n')
    return len(cells), len(verts)



