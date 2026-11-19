"""Full held-out p/U evaluation, two models, numeric outputs only."""
from pathlib import Path
import csv,json
import numpy as np
import torch
import config as c
from poreai.data import FlowDataset
from poreai.train import load_model
from poreai.metrics import case_metrics,summarise_cases
from scripts.generate_cases import atomic_json


def csv_write(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if not rows:return
    with path.open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def evaluate(splits=('test',)):
    if isinstance(splits,str):splits=(splits,)
    if not splits or any(s not in c.SPLITS for s in splits):raise ValueError('Unknown evaluation split')
    torch.set_num_threads(c.TORCH_THREADS)
    ds=FlowDataset(splits)
    result={}
    for name in c.EXPERIMENT_MODELS:
        model,_=load_model(name)
        rows=[]
        for entry in ds.entries:
            with np.load(entry['path'],allow_pickle=False) as f:
                x=f['x'].copy();truth=f['y'].copy()
            with torch.inference_mode():prediction=model(torch.from_numpy(x[None]))[0].numpy()
            values=case_metrics(prediction,truth,x)
            record={'model':name,'split':entry['split'],'seed':entry['seed'],**values}
            rows.append(record)
            out=c.RESULT_ROOT/'evaluation'/name/entry['split']/f'geo_{entry["seed"]}'
            out.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(out/'prediction.npz',x=x,y_true=truth,y_pred=prediction)
            atomic_json(out/'metrics.json',record)
            print(f'[05] {name}/{entry["split"]}/{entry["seed"]}: U={values["velocity_relative_l2"]:.5f} p={values["pressure_rmse_over_deltaP"]:.5f}',flush=True)
        folder=c.RESULT_ROOT/'evaluation'/name
        tag='_'.join(splits)
        csv_write(folder/('per_case.csv' if tag=='test' else f'per_case_{tag}.csv'),rows)
        summary={'model':name,'cases':len(rows),'metrics':summarise_cases(rows)}
        atomic_json(folder/('summary.json' if tag=='test' else f'summary_{tag}.json'),summary)
        result[name]=summary
    tag='_'.join(splits)
    atomic_json(c.RESULT_ROOT/'evaluation'/('comparison_summary.json' if tag=='test' else f'comparison_{tag}.json'),
                {'splits':list(splits),'models':result})
    return result


def report():
    source=c.RESULT_ROOT/'evaluation/comparison_summary.json'
    if not source.exists():raise FileNotFoundError('Run 05_evaluate before 07_report')
    data=json.loads(source.read_text())
    out=c.RESULT_ROOT/'reports';out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for name in c.EXPERIMENT_MODELS:
        for metric,stats in data['models'][name]['metrics']['statistics'].items():
            ci=stats.get('bootstrap_95ci_mean')
            rows.append({'model':name,'metric':metric,'n':stats['n'],
                         'mean':stats['mean'],'median':stats['median'],
                         'minimum':stats['min'],'maximum':stats['max'],
                         'ci95_lower':ci[0] if ci else None,'ci95_upper':ci[1] if ci else None})
    csv_write(out/'model_metrics.csv',rows)
    paired=[]
    if 'unet' in data['models'] and 'topoflux' in data['models']:
        a=c.RESULT_ROOT/'evaluation/unet/per_case.csv';b=c.RESULT_ROOT/'evaluation/topoflux/per_case.csv'
        if a.exists() and b.exists():
            with a.open(newline='') as f:unet={(r['split'],r['seed']):r for r in csv.DictReader(f)}
            with b.open(newline='') as f:adv={(r['split'],r['seed']):r for r in csv.DictReader(f)}
            common=sorted(set(unet)&set(adv))
            rng=np.random.default_rng(2027)
            keys=('pressure_rmse_over_deltaP','velocity_relative_l2','throat_velocity_relative_l2',
                  'near_wall_velocity_relative_l2','pressure_gradient_relative_l2',
                  'cross_section_flux_proxy_relative_l2')
            for key in keys:
                ab=np.array([(float(unet[k][key]),float(adv[k][key])) for k in common
                     if unet[k].get(key) not in ('None','') and adv[k].get(key) not in ('None','')],float)
                if not len(ab):continue
                mean_a,mean_b=ab.mean(axis=0)
                ids=rng.integers(0,len(ab),(2000,len(ab)))
                boot=ab[ids].mean(axis=1)
                improv=100*(boot[:,0]-boot[:,1])/np.maximum(boot[:,0],1e-30)
                lo,hi=np.percentile(improv,[2.5,97.5])
                paired.append({'metric':key,'cases':len(ab),'unet_mean':float(mean_a),
                    'topoflux_mean':float(mean_b),
                    'relative_improvement_pct':float(100*(mean_a-mean_b)/max(mean_a,1e-30)),
                    'topoflux_better_cases':int((ab[:,1]<ab[:,0]).sum()),
                    'ci95_lower':float(lo),'ci95_upper':float(hi)})
    csv_write(out/'paired_comparison.csv',paired)
    atomic_json(out/'summary.json',{'models':data['models'],'paired':paired})
    return [out/'model_metrics.csv',out/'paired_comparison.csv',out/'summary.json']
