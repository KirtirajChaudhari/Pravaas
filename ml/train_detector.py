"""Train a detector on an explicitly selected and licensed YOLO dataset export."""
import argparse
import json
from pathlib import Path

from ultralytics import YOLO
import yaml


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',required=True)
    p.add_argument('--model',default='yolo11n.pt')
    p.add_argument('--epochs',type=int,default=50)
    p.add_argument('--device',default='cpu')
    p.add_argument('--scratch',action='store_true',help='Use yolo11n.yaml with random initialization')
    a=p.parse_args()
    config=yaml.safe_load(Path(a.data).read_text())
    if not all(config.get(k) for k in ('train','val','test','names')):
        raise SystemExit('Require train, val, test, and names. Keep a genuinely held-out test split.')
    model=YOLO('yolo11n.yaml' if a.scratch else a.model)
    model.train(data=a.data,epochs=a.epochs,imgsz=640,device=a.device,seed=42,deterministic=True,project='artifacts/detectors',name='train',exist_ok=False)
    selected=YOLO(model.trainer.best)
    metrics=selected.val(data=a.data,split='test',device=a.device)
    out=Path(model.trainer.save_dir)/'test-metrics.json'
    out.write_text(json.dumps({k:float(v) for k,v in metrics.results_dict.items()},indent=2))
    print(out)


if __name__=='__main__':main()
