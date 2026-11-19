import csv,json
from pathlib import Path
import numpy as np
import pytest
import torch
import config as c
from poreai.features import features
from poreai.model import build_model
from poreai.topology import hydraulic_edges,neighbour_average


def test_default_models_near_same_capacity():
    a=build_model(c.model_architecture('unet'),c.model_width('unet'))
    b=build_model(c.model_architecture('topoflux'),c.model_width('topoflux'),graph_depth=c.GRAPH_DEPTH)
    na=sum(p.numel() for p in a.parameters());nb=sum(p.numel() for p in b.parameters())
    assert abs(na-nb)/na<.04


def test_pore_throat_conditional_cross_scale_and_gradients():
    torch.set_num_threads(2)
    fluid=np.zeros((32,32),bool);fluid[5:27,:]=True
    x=torch.from_numpy(features(fluid))[None]
    model=build_model('conditioned_multiscale_topoflux',8,graph_depth=2)
    pred,attn=model.explain(x)
    assert pred.shape==(1,3,32,32)
    assert torch.all(pred[:,:,~torch.from_numpy(fluid)]==0)
    assert set(attn)=={'graph_attention','graph_gate','condition_attention','scale_weights',
                       'throat_weight','edge_right','edge_down'}
    assert torch.allclose(attn['scale_weights'].sum(1),torch.ones_like(attn['scale_weights'][:,0]),atol=1e-6)
    (pred[:,1:3]**2).mean().backward()
    assert model.graph.stem.weight.grad is not None
    assert model.condition[0].geometry.weight.grad is not None
    assert model.fuse1.gate.weight.grad is not None
    assert model.dense.mix.weight.grad is not None


def test_graph_edges_cannot_leak_across_solid_pixels():
    fluid=np.zeros((32,32),bool)
    fluid[4:12,1:12]=True;fluid[17:27,19:31]=True
    x=torch.from_numpy(features(fluid))[None]
    r,d=hydraulic_edges(x)
    assert torch.count_nonzero(r[0,0,:,:-1][~torch.from_numpy(fluid[:,:-1]&fluid[:,1:])])==0
    assert torch.count_nonzero(d[0,0,:-1,:][~torch.from_numpy(fluid[:-1,:]&fluid[1:,:])])==0
    z=torch.zeros((1,2,32,32));z[:,:,4:12,1:12]=1;z[:,:,17:27,19:31]=100
    smoothed=neighbour_average(z,(r,d))
    assert torch.allclose(smoothed[:,:,4:12,1:12],torch.ones_like(smoothed[:,:,4:12,1:12]),atol=1e-4)
    assert torch.allclose(smoothed[:,:,17:27,19:31],torch.full_like(smoothed[:,:,17:27,19:31],100),atol=1e-3)


def test_generation_default_splits_have_no_historical_manifest(sandbox,monkeypatch):
    from scripts import generate_cases
    monkeypatch.setattr(generate_cases,'_generate_one',lambda split,seed,root:{'split':split,'seed':seed,'actual_seed':seed})
    result=generate_cases.generate()
    assert len(result)==5
    assert not list(c.CASE_ROOT.glob('*manifest*'))


def test_training_evaluation_parallel_figures_and_numeric_report(sandbox):
    from conftest import populate_labels
    from poreai.train import train
    from scripts.evaluate import evaluate,report
    from scripts.visualize import visualize_all
    populate_labels()
    trained=train()
    assert set(trained)=={'unet','topoflux'}
    evaluated=evaluate(('test',))
    assert evaluated['unet']['cases']==1 and evaluated['topoflux']['cases']==1
    figures=visualize_all()
    assert figures['cases']==5 and figures['figures']>=5*3+5*2*2
    paths=report()
    assert all(Path(p).exists() for p in paths)
    index=c.RESULT_ROOT/'figures/index.csv'
    with index.open(newline='') as f:rows=list(csv.DictReader(f))
    assert len(rows)==figures['figures']
    assert all((c.RESULT_ROOT/'figures'/row['relative_path']).exists() for row in rows)
    assert (c.RESULT_ROOT/'figures/predictions/topoflux/test/geo_10001/conditioned_fusion_maps.png').exists()
    assert (c.RESULT_ROOT/'figures/datasets/augmented_mirror/train/geo_101/mirror_y.png').exists()
    assert not list(c.RESULT_ROOT.glob('**/*.md'))
    assert not list(c.MODEL_ROOT.glob('**/*contract*'))


def test_collect_is_threaded_and_skips_cache(sandbox,monkeypatch):
    from scripts import pipeline
    visited=[]
    def mock_collect(split,seed):
        visited.append((split,seed));return {'split':split,'seed':seed}
    monkeypatch.setattr(pipeline,'_collect_one',mock_collect)
    rows=pipeline.collect(['train','test'])
    assert len(rows)==2 and set(visited)=={('train',101),('test',10001)}
