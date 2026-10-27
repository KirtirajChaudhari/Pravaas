"""Reproducible CPU/GPU experiment: scratch CNN, feature extraction, fine-tuning.

Model selection uses validation loss; the test partition is evaluated once per
selected model. Browser exports include ImageNet normalization inside the graph.
"""
import argparse
import copy
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        layers = []
        for a,b in [(3,16),(16,32),(32,64)]:
            layers += [nn.Conv2d(a,b,3,padding=1), nn.ReLU(), nn.MaxPool2d(2)]
        self.network = nn.Sequential(*layers, nn.AdaptiveAvgPool2d((4,4)), nn.Flatten(), nn.Dropout(.3), nn.Linear(1024,64), nn.ReLU(), nn.Linear(64,2))
    def forward(self,x): return self.network(x)


class BrowserModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.register_buffer('mean',torch.tensor([.485,.456,.406]).view(1,3,1,1))
        self.register_buffer('std',torch.tensor([.229,.224,.225]).view(1,3,1,1))
    def forward(self,x): return self.model((x-self.mean)/self.std).softmax(1)


class Images(Dataset):
    def __init__(self, root, records, augment=False):
        self.rows = records
        self.augment = augment
        self.images = [torch.from_numpy(np.asarray(Image.open(root/r['file']).convert('RGB').resize((128,128)),dtype=np.float32).transpose(2,0,1).copy()/255) for r in records]
    def __len__(self): return len(self.rows)
    def __getitem__(self,i):
        x = self.images[i]
        if self.augment:
            if torch.rand(()) < .5: x = x.flip(2)
            x = (x * (.85 + torch.rand(())*.3)).clamp(0,1)
        return (x-torch.tensor([.485,.456,.406])[:,None,None])/torch.tensor([.229,.224,.225])[:,None,None], self.rows[i]['label']


def run(args):
    torch.set_num_threads(args.threads)
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    root, out = Path(args.data), Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((root/'manifest.json').read_text())
    loaders = {s:DataLoader(Images(root,[r for r in meta['records'] if r['split']==s],s=='train'),batch_size=32,shuffle=s=='train',num_workers=0) for s in ('train','val','test')}
    device = torch.device(args.device)
    reports=[]
    models=[('cnn','Scratch CNN',SmallCNN(),args.epochs,.001,'All parameters initialized randomly')]
    mobile=mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
    mobile.classifier[-1]=nn.Linear(mobile.classifier[-1].in_features,2)
    for p in mobile.features.parameters(): p.requires_grad=False
    models.append(('mobilenet','MobileNetV3 · frozen',mobile,args.epochs,.001,'ImageNet features frozen; classification head trained'))
    for slug,title,model,epochs,lr,strategy in models:
        reports.append(fit(slug,title,model,epochs,lr,strategy,loaders,device,out))
    fine=copy.deepcopy(mobile)
    for p in fine.features[-3:].parameters(): p.requires_grad=True
    reports.append(fit('finetuned','MobileNetV3 · fine-tuned',fine,args.finetune_epochs,.0001,'Starts from selected frozen model; last three feature blocks and head trained',loaders,device,out))
    result={'seed':42,'input_size':128,'dataset':{k:v for k,v in meta.items() if k!='records'},'models':reports,'test_policy':'Checkpoint selected by validation loss. Test used only for final reporting, not checkpoint selection.','hardware':{'device':str(device),'threads':args.threads,'torch':torch.__version__}}
    (out/'experiments.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)


def fit(slug,title,model,epochs,lr,strategy,loaders,device,out):
    model.to(device)
    trainable=sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimizer=torch.optim.AdamW(filter(lambda p:p.requires_grad,model.parameters()),lr=lr,weight_decay=.0001)
    loss_fn=nn.CrossEntropyLoss()
    best_loss=float('inf'); best=None; history=[]; start=time.perf_counter()
    for epoch in range(epochs):
        row={'epoch':epoch+1}
        for split in ('train','val'):
            model.train(split=='train')
            # Frozen BatchNorm statistics must also remain frozen.
            if hasattr(model,'features'):
                for block in model.features:
                    if not any(p.requires_grad for p in block.parameters()):block.eval()
            total=correct=n=0
            with torch.set_grad_enabled(split=='train'):
                for x,y in loaders[split]:
                    x,y=x.to(device),y.to(device)
                    logits=model(x);loss=loss_fn(logits,y)
                    if split=='train':optimizer.zero_grad();loss.backward();optimizer.step()
                    total+=loss.item()*len(y);correct+=(logits.argmax(1)==y).sum().item();n+=len(y)
            row[split+'_loss']=total/n;row[split+'_accuracy']=correct/n
        history.append(row)
        if row['val_loss']<best_loss:best_loss=row['val_loss'];best=copy.deepcopy(model.state_dict());best_epoch=epoch+1
        print(slug,row,flush=True)
    elapsed=time.perf_counter()-start
    model.load_state_dict(best);model.eval()
    truth=[];pred=[];t=time.perf_counter()
    with torch.inference_mode():
        for x,y in loaders['test']:
            pred.extend(model(x.to(device)).argmax(1).cpu().tolist());truth.extend(y.tolist())
    latency=(time.perf_counter()-t)*1000/len(truth)
    p,r,f,_=precision_recall_fscore_support(truth,pred,average='binary',pos_label=0,zero_division=0)
    torch.save(model.cpu().state_dict(),out/f'{slug}.pt')
    browser=BrowserModel(model).eval()
    torch.onnx.export(browser,torch.zeros(1,3,128,128),str(out/f'{slug}.onnx'),input_names=['images'],output_names=['probabilities'],opset_version=17,dynamo=False)
    return {'id':slug,'name':title,'strategy':strategy,'epochs':epochs,'best_epoch':best_epoch,'parameters':sum(p.numel() for p in model.parameters()),'trainable_parameters':trainable,'training_seconds':elapsed,'test_ms_per_image':latency,'accuracy':accuracy_score(truth,pred),'precision':float(p),'recall':float(r),'f1':float(f),'confusion_matrix':confusion_matrix(truth,pred,labels=[0,1]).tolist(),'test_count':len(truth),'history':history,'path':f'models/{slug}.onnx'}


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--data',default='data/classification');p.add_argument('--output',default='artifacts')
    p.add_argument('--epochs',type=int,default=12);p.add_argument('--finetune-epochs',type=int,default=5)
    p.add_argument('--threads',type=int,default=2);p.add_argument('--device',default='cpu')
    run(p.parse_args())
