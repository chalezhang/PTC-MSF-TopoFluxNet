"""Independent converged OpenFOAM pressure and velocity supervision."""
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
import config as c
from poreai.features import case_features,target


def label_path(split,seed):return c.DATA_ROOT/split/f'geo_{seed}.npz'
def source_path(split,seed):return c.CASE_ROOT/f'2d/{split}/geo_{seed}'


def save_label(split,seed,source,run,p,U):
    if not run.get('converged') or not run.get('final_fields_valid'):
        raise ValueError('OpenFOAM target must be converged and finite')
    x,mask,_=case_features(source)
    y=target(p,U,mask)
    if not np.isfinite(x).all() or not np.isfinite(y).all():raise FloatingPointError('Nonfinite training sample')
    path=label_path(split,seed);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp.npz')
    np.savez_compressed(temp,x=x,y=y);temp.replace(path)
    return {'split':split,'seed':seed,'fluid_cells':int(mask.sum()),'outer_iterations':int(run['simple_outer_iterations'])}


class FlowDataset(Dataset):
    def __init__(self,splits):
        if isinstance(splits,str):splits=(splits,)
        if not splits or any(s not in c.SPLITS for s in splits):raise ValueError('Unknown split')
        self.entries=[]
        for split in splits:
            for seed in c.SPLITS[split]:
                p=label_path(split,seed)
                if not p.is_file():raise FileNotFoundError(p)
                self.entries.append({'split':split,'seed':seed,'path':str(p)})
    def __len__(self):return len(self.entries)
    def __getitem__(self,i):
        with np.load(self.entries[i]['path'],allow_pickle=False) as pack:
            return torch.from_numpy(pack['x'].copy()),torch.from_numpy(pack['y'].copy())
