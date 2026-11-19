"""From-scratch paired training: plain U-Net versus PTC MultiScale TopoFlux."""
import json,random,time
import numpy as np
import torch
from torch.utils.data import DataLoader,ConcatDataset,WeightedRandomSampler,Dataset
import config as c
from poreai.data import FlowDataset
from poreai.model import build_model
from poreai.losses import loss
from scripts.generate_cases import atomic_json


class ReflectYDataset(Dataset):
    def __init__(self,base):self.base=base
    def __len__(self):return len(self.base)
    def __getitem__(self,index):
        x,y=self.base[index]
        if torch.rand(())<.5:
            x=torch.flip(x,(-2,)).clone();y=torch.flip(y,(-2,)).clone()
            x[3]=1-x[3];x[10]=-x[10];y[2]=-y[2]
        return x,y


def _run_epoch(model,loader,optimizer=None):
    train=optimizer is not None
    model.train(train)
    total=0.;count=0;components={}
    with torch.enable_grad() if train else torch.inference_mode():
        for x,y in loader:
            if train:optimizer.zero_grad(set_to_none=True)
            pred=model(x);value,parts=loss(pred,y,x,True)
            if not torch.isfinite(value):raise FloatingPointError('Nonfinite loss')
            if train:
                value.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                optimizer.step()
            n=len(x);count+=n;total+=float(value.detach())*n
            for k,v in parts.items():components[k]=components.get(k,0.)+float(v.detach())*n
    if not count:raise RuntimeError('Empty dataset')
    return {'total':total/count,'components':{k:v/count for k,v in components.items()}}


def train_one(name):
    if name not in c.EXPERIMENT_MODELS:raise ValueError('Unexpected model')
    torch.set_num_threads(c.TORCH_THREADS)
    seed=c.TRAIN_SEED;random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    regular=FlowDataset('train');throat=FlowDataset('throat_train')
    ds=ReflectYDataset(ConcatDataset([regular,throat]));val=FlowDataset(('val','throat_val'))
    sampler=WeightedRandomSampler([1.]*len(regular)+[c.THROAT_SAMPLE_WEIGHT]*len(throat),
        num_samples=len(ds),replacement=True,generator=torch.Generator().manual_seed(seed))
    loader=DataLoader(ds,batch_size=c.BATCH_SIZE,sampler=sampler,num_workers=0)
    vloader=DataLoader(val,batch_size=c.BATCH_SIZE,shuffle=False,num_workers=0)
    model=build_model(c.model_architecture(name),base=c.model_width(name),graph_depth=c.GRAPH_DEPTH)
    optimizer=torch.optim.AdamW(model.parameters(),lr=c.LEARNING_RATE,weight_decay=c.WEIGHT_DECAY)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=.5,patience=10)
    dest=c.MODEL_ROOT/name;dest.mkdir(parents=True,exist_ok=True)
    (dest/'best.pt').unlink(missing_ok=True)
    best=float('inf');best_epoch=0;bad=0;history=[]
    for epoch in range(1,c.EPOCHS+1):
        # Same training samples and mirror augmentation per epoch across architectures.
        torch.manual_seed(seed+epoch*100003)
        start=time.perf_counter()
        train_stats=_run_epoch(model,loader,optimizer)
        val_stats=_run_epoch(model,vloader)
        scheduler.step(val_stats['total'])
        history.append({'epoch':epoch,'train':train_stats,'validation':val_stats,
            'lr':optimizer.param_groups[0]['lr'],'seconds':time.perf_counter()-start})
        if val_stats['total']+1e-6<best:
            best=val_stats['total'];best_epoch=epoch;bad=0
            temp=dest/'best.tmp.pt';torch.save(model.state_dict(),temp);temp.replace(dest/'best.pt')
        else:bad+=1
        atomic_json(dest/'history.json',history)
        print(f'[04] {name} epoch={epoch} train={train_stats["total"]:.5g} '
              f'val={val_stats["total"]:.5g} best={best:.5g}',flush=True)
        if bad>=c.PATIENCE:break
    outcome={'model':name,'architecture':c.model_architecture(name),'base_channels':c.model_width(name),
             'parameters':sum(v.numel() for v in model.parameters()),
             'best_epoch':best_epoch,'epochs':len(history),'validation_loss':best,
             'training_cases':len(ds),'validation_cases':len(val)}
    atomic_json(dest/'summary.json',outcome)
    return outcome


def train():
    results={name:train_one(name) for name in c.EXPERIMENT_MODELS}
    atomic_json(c.MODEL_ROOT/'comparison.json',results)
    return results


def load_model(name='topoflux'):
    if name not in c.EXPERIMENT_MODELS:raise ValueError('Unknown model')
    path=c.MODEL_ROOT/name/'best.pt'
    if not path.exists():raise FileNotFoundError(path)
    model=build_model(c.model_architecture(name),base=c.model_width(name),graph_depth=c.GRAPH_DEPTH)
    model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True),strict=True)
    model.eval()
    return model,{'model':name,'parameters':sum(p.numel() for p in model.parameters())}
