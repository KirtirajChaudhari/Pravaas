"""Export inherited reference models, retaining provenance and chunked weights."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import onnx
from onnx.external_data_helper import set_external_data
from ultralytics import YOLO


def chunk_weights(path, slug, limit=12*1024*1024):
    graph=onnx.load(path)
    chunk=bytearray();index=0;files=[]
    for tensor in graph.graph.initializer:
        if not tensor.HasField('raw_data') or len(tensor.raw_data)<1024:continue
        raw=tensor.raw_data
        if chunk and len(chunk)+len(raw)>limit:
            name=f'{slug}-weights-{index}.bin';(path.parent/name).write_bytes(chunk);files.append(name)
            index+=1;chunk=bytearray()
        name=f'{slug}-weights-{index}.bin'
        set_external_data(tensor,location=name,offset=len(chunk),length=len(raw))
        chunk.extend(raw);tensor.ClearField('raw_data')
    if chunk:(path.parent/name).write_bytes(chunk);files.append(name)
    onnx.save_model(graph,path)
    return [{'path':n,'data':f'models/{n}'} for n in files]


def export(source,out,slug,name,mode,url):
    source=Path(source);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    model=YOLO(source)
    path=Path(model.export(format='onnx',imgsz=416,opset=17,simplify=False,dynamic=False,half=False,device='cpu'))
    target=out/f'{slug}.onnx';shutil.copy2(path,target)
    external=chunk_weights(target,slug) if target.stat().st_size>24*1024*1024 else []
    return {'id':slug,'name':name,'mode':mode,'task':'detect','input_size':416,'classes':list(model.names.values()),'path':f'models/{slug}.onnx','externalData':external,
            'origin':'Inherited reference weights; not trained in this rebuild','source':url,'sha256_source':hashlib.file_digest(open(source,'rb'),'sha256').hexdigest(),
            'description':'Reference YOLO model · 416 px input · human review required','metrics':None}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--rail',required=True);p.add_argument('--obstacles',required=True);p.add_argument('--output',default='dist/models');a=p.parse_args()
    rows=[export(a.rail,a.output,'rail-yolo','YOLOv8n · rail regions','defects','https://github.com/pradeepxkumar/Rail-defect-detection'),export(a.obstacles,a.output,'obstacle-yolo','YOLO11s · track obstacles','obstacles','https://github.com/KirtirajChaudhari/Pravaas')]
    (Path(a.output)/'detectors.json').write_text(json.dumps(rows,indent=2));print(json.dumps(rows,indent=2))
