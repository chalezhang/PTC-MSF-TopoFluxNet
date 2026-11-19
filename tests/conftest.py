from pathlib import Path
import numpy as np
import pytest
import config as c

@pytest.fixture
def sandbox(tmp_path,monkeypatch):
    settings={'NX':32,'NY':32,'LX':32*c.CELL_SIZE,'LY':32*c.CELL_SIZE,
       'DOMAIN':(32*c.CELL_SIZE,32*c.CELL_SIZE),
       'ROOT':tmp_path,'CASE_ROOT':tmp_path/'cases/study','DATA_ROOT':tmp_path/'data/steady',
       'MODEL_ROOT':tmp_path/'models','RESULT_ROOT':tmp_path/'results',
       'TORCH_THREADS':2,'CPU_BUDGET':4,'COLLECT_WORKERS':2,'COLLECT_THREADS_PER_CASE':2,
       'GENERATE_WORKERS':1,'FIGURE_WORKERS':2,'FIGURE_CHUNK_SIZE':1,'RENDER_DPI':44,
       'MODEL_CHANNELS':8,'UNET_CHANNELS':8,'GRAPH_DEPTH':2,
       'EPOCHS':1,'PATIENCE':2,'BATCH_SIZE':2}
    for name,value in settings.items():monkeypatch.setattr(c,name,value)
    monkeypatch.setattr(c,'DEFAULT_SPLITS',('train','throat_train','val','throat_val','test'))
    monkeypatch.setattr(c,'SPLITS',{'train':(101,), 'throat_train':(3101,),
        'val':(2001,), 'throat_val':(5101,), 'test':(10001,)})
    return tmp_path


def populate_labels():
    from poreai.data import label_path
    from poreai.features import features
    for split,ids in c.SPLITS.items():
        for seed in ids:
            mask=np.zeros((c.NY,c.NX),bool)
            mask[5:27,:]=True
            mask[5:13,10:19]=False;mask[19:27,10:19]=False
            x=features(mask)
            yy,xx=np.mgrid[:c.NY,:c.NX]
            target=np.zeros((3,c.NY,c.NX),np.float32)
            target[0]=x[5]
            target[1]=x[0]*np.maximum(0,1-((yy-16)/12)**2)
            target[2]=x[0]*.03*np.sin(xx/5)
            dest=label_path(split,seed);dest.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(dest,x=x,y=target)
