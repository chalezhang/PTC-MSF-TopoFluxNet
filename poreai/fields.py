"""OpenFOAM 13 ASCII volume-field I/O with a strict row-major cell-order contract."""
from pathlib import Path
import json
import re
import numpy as np

_BLOCK=re.compile(r'internalField\s+nonuniform\s+List<(scalar|vector)>\s*(\d+)\s*\((.*?)\)\s*;',re.S)
_UNIFORM=re.compile(r'internalField\s+uniform\s+(\([^()]+\)|[^;\s]+)\s*;')


def parse_internal(text,field,n):
    if field not in ('p','U') or n<1:
        raise ValueError('Unsupported field or cell count')
    m=_BLOCK.search(text)
    if m:
        typ,num,body=m.groups()
        if int(num)!=n or typ!=('scalar' if field=='p' else 'vector'):
            raise ValueError(f'OpenFOAM {field}: count/type mismatch')
        if field=='p':
            arr=np.fromstring(body,sep=' ')
        else:
            values=re.findall(r'\(([^()]*)\)',body)
            arr=np.fromstring(' '.join(values),sep=' ').reshape(-1) if values else np.empty(0)
        arr=arr.reshape((-1,3)) if field=='U' and arr.size==n*3 else arr
        if arr.shape!=((n,) if field=='p' else (n,3)) or not np.isfinite(arr).all():
            raise ValueError(f'OpenFOAM {field}: invalid field values')
        return arr.astype(np.float64)
    m=_UNIFORM.search(text)
    if m:
        vals=np.fromstring(m.group(1).strip('()'),sep=' ')
        if not np.isfinite(vals).all():raise ValueError('Nonfinite uniform value')
        if field=='p' and vals.size==1:return np.full(n,vals.item())
        if field=='U' and vals.size==3:return np.tile(vals,(n,1))
    raise ValueError(f'Cannot parse {field} internalField')


def read_internal(path,field,n):
    return parse_internal(Path(path).read_text(errors='replace'),field,n)


def final_time_directory(case,fields=('p','U')):
    dirs=[]
    for child in Path(case).iterdir():
        if not child.is_dir():continue
        try:t=float(child.name)
        except ValueError:continue
        if t>0 and np.isfinite(t) and all((child/f).is_file() for f in fields):
            dirs.append((t,child))
    if not dirs:raise FileNotFoundError(f'No final OpenFOAM fields in {case}')
    return max(dirs,key=lambda row:row[0])[1]


def read_final(case):
    meta=json.loads((Path(case)/'geometry.json').read_text())
    n=meta['n_cells']
    final=final_time_directory(case)
    return read_internal(final/'p','p',n),read_internal(final/'U','U',n)


def field_to_grid(values,mask):
    mask=np.asarray(mask,dtype=bool)
    values=np.asarray(values)
    if values.shape[0]!=mask.sum():raise ValueError('Cell count mismatch')
    grid=np.full((mask.shape[0],mask.shape[1],*values.shape[1:]),np.nan,dtype=np.float64)
    grid[mask]=values
    return grid


