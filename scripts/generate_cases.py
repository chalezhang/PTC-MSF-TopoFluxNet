"""Generate fresh independent OpenFOAM CFD geometries."""
from pathlib import Path
import json,shutil,os,multiprocessing
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from scipy.ndimage import label
import config as c
from geometry import generate as generate_geometry
from generate_porous_media import build_case
from poreai.features import throat_score


def atomic_json(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    tmp.replace(path)


def _throughflow(mask):
    # Four-connected pore channel linking both pressure Dirichlet faces.
    components,_=label(np.asarray(mask,dtype=bool))
    inlet=set(int(i) for i in np.unique(components[:,0]) if i)
    outlet=set(int(i) for i in np.unique(components[:,-1]) if i)
    return bool(inlet & outlet)


def _candidate(seed):
    for retry in range(c.GENERATE_MAX_RETRIES_PER_CASE):
        actual=seed+retry*c.GENERATE_SEED_RETRY_STRIDE
        try:
            geom=generate_geometry(c.NX,c.NY,actual,porosity=c.POROSITY,
                feature_size=c.FEATURE_SIZE,domain=c.DOMAIN,
                r_min=c.CIRCLE_R_MIN,r_max=c.CIRCLE_R_MAX,gap=c.CIRCLE_GAP,
                allow_overlap=c.CIRCLE_ALLOW_OVERLAP,overlap_ratio=c.CIRCLE_OVERLAP_RATIO,
                tol=c.POROSITY_TOLERANCE,soft_tol=c.POROSITY_SOFT_TOLERANCE,
                max_iter=c.CIRCLE_MAX_ITER,max_attempts=c.CIRCLE_MAX_ATTEMPTS,
                circle_phase=c.CIRCLE_PHASE,require_percolation=False,
                max_cfd_removed_fraction=c.MAX_CFD_REMOVAL_FRACTION)
        except (ValueError,RuntimeError):continue
        d=geom.diagnostics
        if (d['cfd_ready'] and d['inlet_cells'] and d['outlet_cells'] and
                d['connected_porosity']>0 and _throughflow(geom.cfd_fluid) and
                abs(d['total_porosity']-c.POROSITY)<=c.POROSITY_SOFT_TOLERANCE):
            return actual,geom
    raise RuntimeError(f'Cannot generate CFD-ready geometry from base seed {seed}')


def _generate_one(split,base,root):
    location=root/f'2d/{split}/geo_{base}'
    if location.exists():shutil.rmtree(location)
    selected=None
    # Multiple independent candidate geometries, *only* for throat-focused
    # train/validation, never for the test set. No OpenFOAM target is inspected.
    for trial in range(c.THROAT_CANDIDATES if split.startswith('throat_') else 1):
        seed,geom=_candidate(base+trial*100*c.GENERATE_SEED_RETRY_STRIDE)
        score=throat_score(geom.cfd_fluid) if split.startswith('throat_') else 0.
        if selected is None or score>selected[0]:selected=(score,seed,geom)
    score,actual,geometry=selected
    staging=root/'.staging'/split/f'geo_{base}'
    staging.parent.mkdir(parents=True,exist_ok=True)
    if staging.exists():shutil.rmtree(staging)
    try:
        build_case(staging,actual,(c.NX,c.NY),c.DOMAIN,c,precomputed_geometry=geometry)
        if not (staging/'constant/polyMesh/boundary').is_file():
            raise RuntimeError('Generated geometry has no OpenFOAM mesh')
        location.parent.mkdir(parents=True,exist_ok=True)
        staging.rename(location)
    finally:
        if staging.exists():shutil.rmtree(staging)
    return {'split':split,'seed':base,'actual_seed':actual,'throat_selection_score':score,'cached':False}


def generate(selected=None):
    c.validate()
    selected=list(selected if selected is not None else c.DEFAULT_SPLITS)
    if any(split not in c.SPLITS for split in selected):raise ValueError('Invalid split')
    c.CASE_ROOT.mkdir(parents=True,exist_ok=True)
    jobs=[(split,seed,c.CASE_ROOT) for split in selected for seed in c.SPLITS[split]]
    completed={};errors=[]
    # Each geometry is deterministic for its own base seed; process completion
    # order does not change generated pore configurations or split assignment.
    if c.GENERATE_WORKERS==1:
        for split,seed,root in jobs:
            row=_generate_one(split,seed,root)
            completed[f'{split}/{seed}']=row
            print(f'[02] {split}/{seed} seed={row["actual_seed"]}',flush=True)
    else:
        # Spawn before SciPy is imported in workers, each worker uses one BLAS thread.
        ctx=multiprocessing.get_context('spawn')
        original={k:os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')}
        try:
            for k in original:os.environ[k]='1'
            with ProcessPoolExecutor(max_workers=c.GENERATE_WORKERS,mp_context=ctx) as pool:
                tasks={pool.submit(_generate_one,*job):job[:2] for job in jobs}
                for future in as_completed(tasks):
                    split,seed=tasks[future]
                    try:
                        row=future.result();completed[f'{split}/{seed}']=row
                        print(f'[02] {split}/{seed} seed={row["actual_seed"]}',flush=True)
                    except Exception as e:
                        errors.append({'split':split,'seed':seed,'error':repr(e)})
                        print(f'[02] FAILED {split}/{seed}: {e}',flush=True)
        finally:
            for k,value in original.items():
                if value is None:os.environ.pop(k,None)
                else:os.environ[k]=value
    atomic_json(c.RESULT_ROOT/'generation_summary.json',{'records':completed,'errors':errors,
        'expected':len(jobs),'generated':len(completed)})
    if errors or len(completed)!=len(jobs):raise RuntimeError('Some CFD geometries failed to generate')
    return completed
