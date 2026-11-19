"""All-case figure generation with spawn-safe multiprocess Matplotlib workers.

Main process performs all PyTorch inference. Worker processes render only
immutable NPZ files, preventing duplicated model memory and Matplotlib races.
"""
import csv,json,multiprocessing,os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import torch
import config as c
from poreai.data import label_path
from poreai.train import load_model
from scripts.generate_cases import atomic_json


def _imshow(ax,field,title,extent,cmap='viridis',bounds=None):
    import matplotlib.pyplot as plt
    kw={'vmin':bounds[0],'vmax':bounds[1]} if bounds is not None else {}
    image=ax.imshow(field,extent=extent,origin='lower',interpolation='nearest',cmap=cmap,**kw)
    ax.set(title=title,xlabel='x (mm)',ylabel='y (mm)')
    ax.figure.colorbar(image,ax=ax,fraction=.045,pad=.03)


def _render(task):
    """Each process consumes one geometry label and its two saved predictions."""
    import matplotlib
    matplotlib.use('Agg',force=True)
    from matplotlib import pyplot as plt
    from matplotlib.collections import LineCollection
    label_file, model_files, destination, split, seed, physics, aug=task
    path=Path(destination)
    dpi=physics['dpi'];scale_p=physics['pressure_scale'];scale_u=physics['velocity_scale']
    lx,ly=physics['lx'],physics['ly'];extent=(0,lx*1000,0,ly*1000)
    with np.load(label_file,allow_pickle=False) as pack:
        x=pack['x'].copy();truth=pack['y'].copy()
    mask=x[0]>.5
    p=np.where(mask,truth[0]*scale_p,np.nan)
    ux=np.where(mask,truth[1]*scale_u,np.nan)
    uy=np.where(mask,truth[2]*scale_u,np.nan)
    speed=np.hypot(ux,uy)
    produced=[]
    folder=path/'datasets'/split/f'geo_{seed}';folder.mkdir(parents=True,exist_ok=True)
    fig,axes=plt.subplots(2,3,figsize=(13,8),constrained_layout=True)
    panels=((mask.astype(float),'Fluid geometry','Greys'),(x[7],'Throat weight','YlOrRd'),
            (np.where(mask,x[11],np.nan),'Clearance','cividis'),
            (p,'OpenFOAM pressure (m²/s²)','viridis'),
            (speed,'OpenFOAM |U| (m/s)','plasma'),
            (np.where(mask,x[5],np.nan),'Harmonic prior','viridis'))
    for ax,(data,title,cmap) in zip(axes.flat,panels):_imshow(ax,data,title,extent,cmap)
    fig.savefig(folder/'reference_fields.png',dpi=dpi);plt.close(fig);produced.append(folder/'reference_fields.png')
    fig,ax=plt.subplots(figsize=(7,6),layout='constrained')
    ax.imshow(mask.astype(float),extent=extent,origin='lower',cmap='Greys')
    from itertools import chain
    segments=[[(v,0),(v,ly*1000)] for v in np.linspace(0,lx*1000,mask.shape[1]+1)]
    segments +=[[(0,v),(lx*1000,v)] for v in np.linspace(0,ly*1000,mask.shape[0]+1)]
    ax.add_collection(LineCollection(segments,colors='black',linewidths=.10,alpha=.25))
    ax.set(xlabel='x (mm)',ylabel='y (mm)',title='CFD grid')
    fig.savefig(folder/'mesh_grid.png',dpi=dpi);plt.close(fig);produced.append(folder/'mesh_grid.png')
    fig,ax=plt.subplots(figsize=(7,6),layout='constrained')
    image=ax.imshow(speed,extent=extent,origin='lower',cmap='plasma')
    j,i=np.mgrid[:mask.shape[0],:mask.shape[1]]
    step=max(1,max(mask.shape)//24)
    ax.quiver((i[::step,::step]+.5)*lx*1000/mask.shape[1],
              (j[::step,::step]+.5)*ly*1000/mask.shape[0],
              np.nan_to_num(ux[::step,::step]),np.nan_to_num(uy[::step,::step]),
              color='white',width=.003)
    ax.set(xlabel='x (mm)',ylabel='y (mm)',title='OpenFOAM velocity vectors')
    fig.colorbar(image,ax=ax)
    fig.savefig(folder/'velocity_vectors.png',dpi=dpi);plt.close(fig)
    produced.append(folder/'velocity_vectors.png')
    if aug and split in ('train','throat_train'):
        mirrored=np.flip(x,axis=-2).copy();ym=np.flip(truth,axis=-2).copy()
        mirrored[3]=1-mirrored[3];mirrored[10]=-mirrored[10];ym[2]=-ym[2]
        af=path/'datasets'/'augmented_mirror'/split/f'geo_{seed}';af.mkdir(parents=True,exist_ok=True)
        fig,axes=plt.subplots(2,3,figsize=(13,8),constrained_layout=True)
        plots=((x[0],'Original mask','Greys'),(truth[0],'Original p','viridis'),
               (np.hypot(truth[1],truth[2]),'Original speed','plasma'),
               (mirrored[0],'Mirrored mask','Greys'),(ym[0],'Mirrored p','viridis'),
               (np.hypot(ym[1],ym[2]),'Mirrored speed','plasma'))
        for ax,(arr,title,colormap) in zip(axes.flat,plots):_imshow(ax,arr,title,extent,colormap)
        fig.savefig(af/'mirror_y.png',dpi=dpi);plt.close(fig);produced.append(af/'mirror_y.png')
    for model,predict_path in model_files.items():
        with np.load(predict_path,allow_pickle=False) as pack:
            pred=pack['y_pred'].copy()
            info={k:pack[k].copy() for k in pack.files if k.startswith('explain_')}
        folder=path/'predictions'/model/split/f'geo_{seed}'
        folder.mkdir(parents=True,exist_ok=True)
        pt=truth[0]*scale_p;pp=pred[0]*scale_p
        vt=np.linalg.norm(truth[1:3],axis=0)*scale_u
        vp=np.linalg.norm(pred[1:3],axis=0)*scale_u
        ve=np.linalg.norm(pred[1:3]-truth[1:3],axis=0)*scale_u
        lo=min(pt[mask].min(),pp[mask].min());hi=max(pt[mask].max(),pp[mask].max())
        hi_u=max(vt[mask].max(),vp[mask].max(),1e-15)
        def fluid(z):return np.where(mask,z,np.nan)
        fig,axes=plt.subplots(2,3,figsize=(13,8),constrained_layout=True)
        fields=((fluid(pt),'OpenFOAM p','viridis',(lo,hi)),
                (fluid(pp),'AI p','viridis',(lo,hi)),
                (fluid(np.abs(pp-pt)),'|Δp|','inferno',None),
                (fluid(vt),'OpenFOAM |U|','plasma',(0,hi_u)),
                (fluid(vp),'AI |U|','plasma',(0,hi_u)),
                (fluid(ve),'|ΔU vector|','inferno',None))
        for ax,(arr,title,cmap,bounds) in zip(axes.flat,fields):_imshow(ax,arr,title,extent,cmap,bounds)
        fig.savefig(folder/'prediction_and_error.png',dpi=dpi);plt.close(fig)
        produced.append(folder/'prediction_and_error.png')
        fig,axes=plt.subplots(2,3,figsize=(13,8),constrained_layout=True)
        for row,ch in enumerate((1,2)):
            a=truth[ch]*scale_u;b=pred[ch]*scale_u
            vmax=max(abs(a[mask]).max(),abs(b[mask]).max(),1e-15)
            delta=max(abs((b-a)[mask]).max(),1e-15)
            for j,(plane,title,v) in enumerate(((a,'OF',vmax),(b,'AI',vmax),(b-a,'AI−OF',delta))):
                _imshow(axes[row,j],fluid(plane),f'U{("x" if ch==1 else "y")} {title}',extent,'coolwarm',(-v,v))
        fig.savefig(folder/'velocity_components.png',dpi=dpi);plt.close(fig)
        produced.append(folder/'velocity_components.png')
        if info:
            fig,axes=plt.subplots(2,3,figsize=(12,8),constrained_layout=True)
            diagnostics=((info['explain_condition_attention'][0],'Pore-throat condition attention'),
                (info['explain_graph_attention'][0],'Connected graph attention'),
                (info['explain_graph_gate'][0],'Graph-CNN gate'),
                (info['explain_scale_weights'][0],'Fine-scale selection'),
                (info['explain_scale_weights'][1],'Decoder selection'),
                (info['explain_scale_weights'][2],'Cross-scale selection'))
            for ax,(arr,title) in zip(axes.flat,diagnostics):
                _imshow(ax,fluid(arr),title,extent,'viridis',(0,1))
            fig.savefig(folder/'conditioned_fusion_maps.png',dpi=dpi);plt.close(fig)
            produced.append(folder/'conditioned_fusion_maps.png')
    return [str(p) for p in produced]


def _summary_figures():
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    directory=c.RESULT_ROOT/'figures/summary';directory.mkdir(parents=True,exist_ok=True)
    outputs=[]
    for model in c.EXPERIMENT_MODELS:
        file=c.MODEL_ROOT/model/'history.json'
        if not file.exists():continue
        history=json.loads(file.read_text())
        fig,ax=plt.subplots(figsize=(8,5))
        ax.plot([e['epoch'] for e in history],[e['train']['total'] for e in history],label='train')
        ax.plot([e['epoch'] for e in history],[e['validation']['total'] for e in history],label='validation')
        ax.legend();ax.set(xlabel='Epoch',ylabel='Loss',title=model);ax.grid(alpha=.2)
        fig.tight_layout();name=directory/f'training_{model}.png';fig.savefig(name,dpi=c.RENDER_DPI)
        plt.close(fig);outputs.append(name)
    path=c.RESULT_ROOT/'evaluation/comparison_summary.json'
    if path.exists():
        comp=json.loads(path.read_text())
        metrics=('pressure_rmse_over_deltaP','velocity_relative_l2','pressure_gradient_relative_l2','cross_section_flux_proxy_relative_l2')
        fig,axs=plt.subplots(2,2,figsize=(10,8),constrained_layout=True)
        for ax,key in zip(axs.flat,metrics):
            names=list(c.EXPERIMENT_MODELS)
            vals=[comp['models'][n]['metrics']['statistics'][key]['mean'] for n in names]
            ax.bar(names,vals);ax.set_title(key);ax.grid(axis='y',alpha=.2)
        name=directory/'model_metrics.png';fig.savefig(name,dpi=c.RENDER_DPI);plt.close(fig);outputs.append(name)
    return outputs


def _figure_index():
    root=c.RESULT_ROOT/'figures';root.mkdir(parents=True,exist_ok=True)
    rows=[]
    for file in sorted(root.rglob('*.png')):
        parts=file.relative_to(root).parts
        rows.append({'category':parts[0],'partition':'/'.join(parts[1:-1]),
                     'image':parts[-1],'relative_path':file.relative_to(root).as_posix()})
    with (root/'index.csv').open('w',newline='') as fp:
        writer=csv.DictWriter(fp,fieldnames=('category','partition','image','relative_path'))
        writer.writeheader();writer.writerows(rows)
    atomic_json(root/'summary.json',{'cases':sum((label_path(s,k).is_file()) for s in c.SPLITS for k in c.SPLITS[s]),
                                     'figures':len(rows),'workers':c.FIGURE_WORKERS})
    return len(rows)


def visualize_all():
    """Serial inference + spawn multiprocessing drawing of each CFD geometry."""
    torch.set_num_threads(c.TORCH_THREADS)
    models={name:load_model(name)[0] for name in c.EXPERIMENT_MODELS}
    tasks=[];root=c.RESULT_ROOT/'figures'
    physics={'dpi':c.RENDER_DPI,'pressure_scale':c.pressure_scale(),
             'velocity_scale':c.velocity_scale(),'lx':c.LX,'ly':c.LY}
    for split,keys in c.SPLITS.items():
        for seed in keys:
            lab=label_path(split,seed)
            if not lab.is_file():continue
            with np.load(lab,allow_pickle=False) as fp:
                x=fp['x'].copy();truth=fp['y'].copy()
            files={}
            for name,model in models.items():
                if name=='topoflux':
                    with torch.inference_mode():estimate,explain=model.explain(torch.from_numpy(x[None]))
                    additional={f'explain_{key}':v.cpu().numpy()[0] for key,v in explain.items()}
                else:
                    with torch.inference_mode():estimate=model(torch.from_numpy(x[None]))
                    additional={}
                pred=estimate.cpu().numpy()[0]
                out=c.RESULT_ROOT/'predictions'/name/split/f'geo_{seed}.npz'
                out.parent.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(out,y_pred=pred,**additional)
                files[name]=str(out)
            tasks.append((str(lab),files,str(root),split,seed,physics,c.PLOT_TRAIN_AUGMENTATION))
            print(f'[06] predicted {split}/{seed}',flush=True)
    if not tasks:raise FileNotFoundError('No OpenFOAM labels for visualization')
    # Spawn avoids inherited torch/OpenMP state and unsafe Matplotlib after fork.
    if c.FIGURE_WORKERS==1:
        for task in tasks:_render(task)
    else:
        ctx=multiprocessing.get_context('spawn')
        vars=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')
        old={k:os.environ.get(k) for k in vars}
        try:
            for key in vars:os.environ[key]='1'
            with ProcessPoolExecutor(max_workers=c.FIGURE_WORKERS,mp_context=ctx) as pool:
                for idx,_ in enumerate(pool.map(_render,tasks,chunksize=c.FIGURE_CHUNK_SIZE),1):
                    print(f'[06] figures {idx}/{len(tasks)}',flush=True)
        finally:
            for key,value in old.items():
                if value is None:os.environ.pop(key,None)
                else:os.environ[key]=value
    _summary_figures()
    return {'cases':len(tasks),'figures':_figure_index()}

if __name__=='__main__':print(visualize_all())
