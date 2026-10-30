"""Verify model exports against PyTorch and record deterministic ONNX fixtures."""
import argparse
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import nn
from torchvision.models import mobilenet_v3_small

from train import SmallCNN, BrowserModel


def main():
    p=argparse.ArgumentParser();p.add_argument('--dist',default='dist');p.add_argument('--artifacts',default='artifacts');a=p.parse_args()
    torch.set_num_threads(1)
    dist=Path(a.dist);registry=json.loads((dist/'data/registry.json').read_text());checks=[]
    options=ort.SessionOptions();options.intra_op_num_threads=1
    for m in registry['models']:
        model_path=dist/m['path'];onnx.checker.check_model(str(model_path))
        session=ort.InferenceSession(str(model_path),sess_options=options,providers=['CPUExecutionProvider'])
        size=m['input_size'];x=((np.arange(3*size*size,dtype=np.float32)%251)/250).reshape(1,3,size,size)
        y=session.run(None,{session.get_inputs()[0].name:x})[0]
        entry={'id':m['id'],'shape':list(y.shape),'finite':bool(np.isfinite(y).all()),'fixture':y.reshape(-1)[:16].tolist()}
        assert entry['finite']
        if m['task']=='classify':
            if m['id']=='cnn':model=SmallCNN()
            else:
                model=mobilenet_v3_small(weights=None);model.classifier[-1]=nn.Linear(model.classifier[-1].in_features,2)
            model.load_state_dict(torch.load(Path(a.artifacts)/f"{m['id']}.pt",weights_only=True,map_location='cpu'))
            with torch.inference_mode():expected=BrowserModel(model).eval()(torch.from_numpy(x)).numpy()
            error=float(np.abs(expected-y).max());entry['max_absolute_error_vs_pytorch']=error
            assert error<1e-5,(m['id'],error)
        checks.append(entry)
    manifest=json.loads((dist/'data/manifest.json').read_text());hashes=[r['sha256_pixels'] for r in manifest['records']]
    assert len(set(hashes))==len(hashes),'Duplicate pixels across prepared data'
    report={'export_checks':checks,'unique_dataset_images':len(hashes),'split_counts':manifest['counts']}
    (dist/'data/verification.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':main()
