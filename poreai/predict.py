"""Standalone, one-shot steady-field surrogate; does not launch OpenFOAM."""
from pathlib import Path
import numpy as np
import torch
import config as c
from poreai.features import case_features,physical_prediction
from poreai.train import load_model


def predict_case(case,model=None):
    x,mask,meta=case_features(case)
    if model is None:model,_=load_model()
    torch.set_num_threads(c.TORCH_THREADS)
    with torch.inference_mode():
        estimate=model(torch.from_numpy(x[None]))[0].cpu().numpy()
    p,u=physical_prediction(estimate,mask)
    return {'x':x,'normalized_y':estimate,'p':p,'U':u,'mask':mask,'meta':meta}


def save_prediction(case,out):
    result=predict_case(case)
    out=Path(out);out.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out,p=result['p'],U=result['U'],
        normalized_y=result['normalized_y'],mask=result['mask'].astype(np.uint8))
    return out
