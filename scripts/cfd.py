"""Unmodified OpenFOAM Foundation 13 produces *only* supervised reference fields."""
import json,os,shutil,subprocess,time
from pathlib import Path
import numpy as np
import config as c
from poreai.features import case_features
from poreai.fields import read_final
from generate_porous_media import write_solver_controls
from scripts.convergence import analyze_log,save_convergence_trace
from scripts.resources import cpu_reservation
from scripts.generate_cases import atomic_json


def source_case(split,seed):return c.CASE_ROOT/f'2d/{split}/geo_{seed}'
def cfd_case(split,seed):return c.RESULT_ROOT/'cfd'/split/f'geo_{seed}'

def _call(cmd,log,env=None):
    with Path(log).open('w') as fp:
        return subprocess.run(cmd,stdout=fp,stderr=subprocess.STDOUT,env=env,check=False).returncode


def run_native(source,output,check_mesh=True):
    source=Path(source).resolve();output=Path(output).resolve()
    if source==output or source in output.parents or output in source.parents:
        raise ValueError('Source/output overlap')
    _,mask,meta=case_features(source)
    if not meta.get('cfd_ready') or meta.get('openfoam_version')!='13':
        raise ValueError('Invalid OpenFOAM 13 CFD case')
    if output.exists():shutil.rmtree(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    shutil.copytree(source,output,ignore=shutil.ignore_patterns('renders','*.png'))
    write_solver_controls(output,c)
    if check_mesh:
        rc=_call(['checkMesh','-case',str(output),'-allGeometry','-allTopology'],output/'checkMesh.log')
        log=(output/'checkMesh.log').read_text(errors='replace')
        if rc or 'Mesh OK.' not in log or 'Failed ' in log:
            raise RuntimeError(f'checkMesh failed: {output}/checkMesh.log')
    env=os.environ.copy()
    for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
        env[key]=str(c.COLLECT_THREADS_PER_CASE)
    with cpu_reservation(c.COLLECT_THREADS_PER_CASE):
        start=time.perf_counter()
        rc=_call(['foamRun','-case',str(output),'-solver','incompressibleFluid'],output/'solver.log',env)
        elapsed=time.perf_counter()-start
    log=(output/'solver.log').read_text(errors='replace')
    if rc:
        atomic_json(output/'error.json',{'stage':'foamRun','returncode':rc,'tail':log[-3000:]})
        raise RuntimeError(f'foamRun failed: {output}/solver.log; {log[-800:]}')
    metrics=analyze_log(log,c.SIMPLE_MAX_STEPS)
    save_convergence_trace(log,output/'iteration_trace.csv')
    residuals=metrics['final_field_residuals']
    continuity=metrics.get('final_continuity_errors')
    if not continuity or any(not np.isfinite(v) or abs(v)>c.MAX_CONTINUITY_ERROR for v in continuity.values()):
        metrics.update(converged=False,termination_reason='continuity_residual_invalid')
    if ('p' not in residuals or not ('Ux' in residuals or 'U' in residuals)):
        metrics.update(converged=False,termination_reason='missing_p_or_U_residual')
    for field,entry in residuals.items():
        if field in ('p','Ux','Uy','U') and (entry['initial_residual'] is None or
                entry['initial_residual']>(c.CONVERGENCE_P if field=='p' else c.CONVERGENCE_U)):
            metrics.update(converged=False,termination_reason=f'final_{field}_residual_too_large')
    valid=False;field_error=None
    if metrics['converged']:
        try:
            p,U=read_final(output)
            valid=(p.shape==(int(mask.sum()),) and U.shape==(int(mask.sum()),3)
                   and np.isfinite(p).all() and np.isfinite(U).all())
        except (OSError,ValueError,KeyError,FileNotFoundError) as e:
            field_error=repr(e)
    record={**metrics,'final_fields_valid':bool(valid),'final_fields_error':field_error,
            'solver_wall_seconds':elapsed,'output':str(output),
            'solver':'Foundation13/incompressibleFluid'}
    atomic_json(output/'run.json',record)
    if not metrics['converged'] or not valid:
        raise RuntimeError(f'CFD not converged/invalid fields: {output}/run.json')
    print(f'[03] CFD {source.name}: converged, outer={metrics["simple_outer_iterations"]}, seconds={elapsed:.1f}',flush=True)
    return record
