"""Copy measured experiment outputs into the static app and update the registry."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--artifacts',default='artifacts');p.add_argument('--data',default='data/classification');p.add_argument('--dist',default='dist');a=p.parse_args()
    artifacts,dist,data=Path(a.artifacts),Path(a.dist),Path(a.data)
    (dist/'models').mkdir(parents=True,exist_ok=True);(dist/'data').mkdir(parents=True,exist_ok=True)
    experiments=json.loads((artifacts/'experiments.json').read_text())
    registry=json.loads((dist/'data/registry.json').read_text())
    registry['models']=[m for m in registry['models'] if m['task']=='detect']
    for report in experiments['models']:
        path=artifacts/f"{report['id']}.onnx";shutil.copy2(path,dist/'models'/path.name)
        registry['models'].append({'id':report['id'],'name':report['name'],'mode':'classification','task':'classify','input_size':experiments['input_size'],'classes':experiments['dataset']['classes'],'path':report['path'],'description':report['strategy'],'origin':'Trained in this rebuild on Kaggle version 2','sha256':hashlib.file_digest(open(path,'rb'),'sha256').hexdigest()})
    (dist/'data/registry.json').write_text(json.dumps(registry,indent=2))
    shutil.copy2(artifacts/'experiments.json',dist/'data/experiments.json')
    shutil.copy2(data/'manifest.json',dist/'data/manifest.json')


if __name__=='__main__':main()
