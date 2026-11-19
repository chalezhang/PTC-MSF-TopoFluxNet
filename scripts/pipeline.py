"""Clean sequential workflow with independent concurrent OpenFOAM collection."""
import argparse,json,os
from concurrent.futures import ThreadPoolExecutor,as_completed
from shutil import which
import config as c
from poreai.data import save_label,source_path
from poreai.fields import read_final
from scripts.cfd import run_native,cfd_case
from scripts.generate_cases import atomic_json


def health(require_foam=True):
    c.validate()
    import torch
    status={'torch':torch.__version__,'cpu_budget':c.CPU_BUDGET,
            'geometry_processes':c.GENERATE_WORKERS,'collect_workers':c.COLLECT_WORKERS,'threads_per_cfd':c.COLLECT_THREADS_PER_CASE,
            'figure_processes':c.FIGURE_WORKERS,'models':list(c.EXPERIMENT_MODELS),
            'splits':{name:len(items) for name,items in c.SPLITS.items()}}
    if require_foam and (os.environ.get('WM_PROJECT_VERSION')!='13' or not all(which(x) for x in ('foamRun','checkMesh'))):
        raise RuntimeError('OpenFOAM Foundation 13 is required for CFD stages')
    print('[01] '+json.dumps(status,ensure_ascii=False),flush=True)
    return status


def _collect_one(split,seed):
    source=source_path(split,seed)
    # Unique case paths per worker. No global OpenFOAM model state is shared.
    target=cfd_case(split,seed)
    run=run_native(source,target)
    p,U=read_final(target)
    return save_label(split,seed,source,run,p,U)


def collect(splits=None):
    selected=list(splits if splits is not None else c.DEFAULT_SPLITS)
    if any(s not in c.SPLITS for s in selected):raise ValueError('Unknown split')
    jobs=[(split,seed) for split in selected for seed in c.SPLITS[split]]
    rows=[];errors=[]
    print(f'[03] CFD cases={len(jobs)} concurrent_cases={c.COLLECT_WORKERS}',flush=True)
    with ThreadPoolExecutor(max_workers=c.COLLECT_WORKERS) as pool:
        future_map={pool.submit(_collect_one,split,seed):(split,seed) for split,seed in jobs}
        for future in as_completed(future_map):
            split,seed=future_map[future]
            try:
                rows.append(future.result());print(f'[03] {split}/{seed} complete',flush=True)
            except Exception as exc:
                errors.append({'split':split,'seed':seed,'error':repr(exc)})
                print(f'[03] {split}/{seed} FAIL {exc}',flush=True)
            atomic_json(c.RESULT_ROOT/'collect_progress.json',
                {'completed':len(rows),'expected':len(jobs),'failed':len(errors)})
    rows.sort(key=lambda r:(selected.index(r['split']),r['seed']))
    atomic_json(c.RESULT_ROOT/'collect_summary.json',
                {'completed':len(rows),'expected':len(jobs),'records':rows,'errors':errors})
    if errors or len(rows)!=len(jobs):raise RuntimeError(f'{len(errors)} CFD cases failed')
    return rows


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=('health','generate','collect','train','evaluate','visualize','report','predict'))
    parser.add_argument('--splits',nargs='+',choices=list(c.SPLITS))
    parser.add_argument('--case');parser.add_argument('--output')
    args=parser.parse_args()
    if args.stage=='health':return health()
    if args.stage=='generate':
        from scripts.generate_cases import generate
        return generate(args.splits)
    if args.stage=='collect':return collect(args.splits)
    if args.stage=='train':
        from poreai.train import train
        return train()
    if args.stage=='evaluate':
        from scripts.evaluate import evaluate
        return evaluate(args.splits or ['test'])
    if args.stage=='visualize':
        from scripts.visualize import visualize_all
        return visualize_all()
    if args.stage=='report':
        from scripts.evaluate import report
        return report()
    if args.stage=='predict':
        if not args.case or not args.output:parser.error('predict requires --case --output')
        from poreai.predict import save_prediction
        return save_prediction(args.case,args.output)

if __name__=='__main__':main()
