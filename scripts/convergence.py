"""OpenFOAM Foundation 13 SIMPLE convergence evidence and iteration counters."""
import math
import re

CONVERGENCE=re.compile(r'SIMPLE\s+solution\s+converged\s+in\s+(\d+)\s+iterations',re.I)
PRESSURE=re.compile(r'Solving for p,.*?No Iterations\s+(\d+)',re.I)
FIELD=re.compile(r'Solving for (p|Ux|Uy|Uz|U),\s*Initial residual\s*=\s*([^,\s]+),\s*Final residual\s*=\s*([^,\s]+)',re.I)
CONTINUITY=re.compile(r'time step continuity errors\s*:\s*sum local\s*=\s*([^,\s]+),\s*global\s*=\s*([^,\s]+)',re.I)


def analyze_log(log,max_steps):
    values=[int(x) for x in PRESSURE.findall(log)]
    if not values:raise RuntimeError('求解器日志缺少压力迭代次数，不可评价')
    marks=[int(x) for x in CONVERGENCE.findall(log)]
    outer=len(values)
    # No marker => cannot call completed, even if OpenFOAM exits with status 0.
    converged=(len(marks)==1 and marks[0]==outer and outer<=max_steps)
    fields={}
    for m in FIELD.finditer(log):
        try:
            v1=float(m.group(2));v2=float(m.group(3))
        except ValueError:continue
        fields[m.group(1)]={'initial_residual':v1 if math.isfinite(v1) else None,
                            'final_residual':v2 if math.isfinite(v2) else None}
    if any(v is None for f in fields.values() for v in f.values()):converged=False
    if 'p' not in fields or not ('Ux' in fields or 'U' in fields):converged=False
    continuity=list(CONTINUITY.finditer(log))
    last_error=None
    if continuity:
        m=continuity[-1]
        try:
            last_error={'sum_local':float(m.group(1)),'global':float(m.group(2))}
        except ValueError: pass
    return {'converged':converged,
            'termination_reason':('converged' if converged else 'missing_or_mismatched_convergence_marker'),
            'simple_outer_iterations':outer,'pressure_inner_iterations':sum(values),
            'max_simple_outer_iterations':max_steps,
            'reported_convergence_iteration':marks[-1] if marks else None,
            'final_field_residuals':fields,'final_continuity_errors':last_error}


def save_convergence_trace(log,path):
    """Preserve time-resolved SIMPLE residuals and linear-iteration counts.

    One CSV row per pressure solve, including Ux/Uy and continuity records from
    the corresponding outer step. Missing tokens are written blank, not guessed.
    """
    import csv
    from pathlib import Path
    rows=[];step=0;record={}
    for line in log.splitlines():
        t=re.search(r'^Time\s*=\s*([0-9.eE+-]+)',line)
        if t:
            if record.get('pressure_iterations') is not None:rows.append(record)
            step+=1
            record={'step':step,'pseudo_time':t.group(1)}
            continue
        m=re.search(r'Solving for (p|Ux|Uy),\s*Initial residual\s*=\s*([^,\s]+),\s*Final residual\s*=\s*([^,\s]+),\s*No Iterations\s+(\d+)',line)
        if m:
            name=m.group(1)
            for key,value in [('initial_residual',m.group(2)),('final_residual',m.group(3))]:
                val=float(value)
                if not math.isfinite(val):raise ValueError(f'Nonfinite log residual: step={step} field={name}')
                record[f'{name}_{key}']=val
            record[f'{name}_iterations']=int(m.group(4))
            if name=='p':record['pressure_iterations']=int(m.group(4))
        m=CONTINUITY.search(line)
        if m:
            record['continuity_sum_local']=float(m.group(1))
            record['continuity_global']=float(m.group(2))
    if record.get('pressure_iterations') is not None:rows.append(record)
    if not rows:raise ValueError('No pressure-iteration trace found in CFD output')
    dest=Path(path);dest.parent.mkdir(parents=True,exist_ok=True)
    columns=['step','pseudo_time','Ux_initial_residual','Uy_initial_residual','p_initial_residual',
             'Ux_final_residual','Uy_final_residual','p_final_residual',
             'Ux_iterations','Uy_iterations','p_iterations','pressure_iterations',
             'continuity_sum_local','continuity_global']
    with dest.open('w',newline='') as fp:
        writer=csv.DictWriter(fp,fieldnames=columns)
        writer.writeheader();writer.writerows(rows)
    return rows
