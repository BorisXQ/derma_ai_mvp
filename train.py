"""Train a configurable EfficientNet-B0 classifier from data/dataset/<class_id>/*.jpg.
Use medically reviewed labels and a patient-level split when patient IDs are available.
This is a research training script, not a clinical validation pipeline.
"""
import argparse, json, random
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset
from torchvision import datasets, transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, balanced_accuracy_score, f1_score

ROOT=Path(__file__).resolve().parent

def main():
 p=argparse.ArgumentParser();p.add_argument('--data',default='data/dataset');p.add_argument('--epochs',type=int,default=8);p.add_argument('--batch-size',type=int,default=16);p.add_argument('--image-size',type=int,default=320);p.add_argument('--lr',type=float,default=1e-4);p.add_argument('--seed',type=int,default=42);p.add_argument('--output',default='model/custom_model.pth');args=p.parse_args()
 random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); data=Path(args.data)
 train_tf=transforms.Compose([transforms.Resize((args.image_size,args.image_size)),transforms.RandomHorizontalFlip(),transforms.RandomRotation(15),transforms.ColorJitter(brightness=.12,contrast=.12,saturation=.08),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
 eval_tf=transforms.Compose([transforms.Resize((args.image_size,args.image_size)),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
 base=datasets.ImageFolder(data,transform=None); classes=base.classes
 if len(classes)<2: raise SystemExit('Need at least 2 class folders under '+str(data))
 counts=np.bincount(base.targets,minlength=len(classes)); print('Classes:',dict(zip(classes,counts.tolist())))
 if min(counts)<10: print('WARNING: at least one class has fewer than 10 images; performance will likely be unreliable. Add substantially more diverse, verified images.')
 idx=np.arange(len(base)); train_idx,val_idx=train_test_split(idx,test_size=.2,random_state=args.seed,stratify=base.targets)
 train_ds=datasets.ImageFolder(data,transform=train_tf); val_ds=datasets.ImageFolder(data,transform=eval_tf)
 train_subset=Subset(train_ds,train_idx); val_subset=Subset(val_ds,val_idx)
 train_counts=np.bincount(np.array(base.targets)[train_idx],minlength=len(classes)); sample_weights=[1.0/max(1,train_counts[base.targets[i]]) for i in train_idx]
 sampler=WeightedRandomSampler(sample_weights,len(sample_weights),replacement=True)
 train_dl=DataLoader(train_subset,batch_size=args.batch_size,sampler=sampler,num_workers=0);val_dl=DataLoader(val_subset,batch_size=args.batch_size,shuffle=False,num_workers=0)
 try: model=efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
 except Exception as e: raise SystemExit('Could not download ImageNet pretrained weights. Connect to internet once or cache torchvision weights first. '+str(e))
 model.classifier[1]=nn.Linear(model.classifier[1].in_features,len(classes));model.to(device)
 weights=torch.tensor([len(train_idx)/max(1*len(classes)*c,1) for c in train_counts],dtype=torch.float32).to(device)
 criterion=nn.CrossEntropyLoss(weight=weights);opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4);best=-1
 for epoch in range(args.epochs):
  model.train();losses=[]
  for x,y in train_dl:
   x,y=x.to(device),y.to(device);opt.zero_grad();loss=criterion(model(x),y);loss.backward();opt.step();losses.append(loss.item())
  model.eval();ys=[];preds=[]
  with torch.inference_mode():
   for x,y in val_dl:
    logits=model(x.to(device));preds.extend(logits.argmax(1).cpu().tolist());ys.extend(y.tolist())
  bal=balanced_accuracy_score(ys,preds); macro=f1_score(ys,preds,average='macro',zero_division=0)
  print(f'Epoch {epoch+1}/{args.epochs}: train_loss={np.mean(losses):.4f} val_balanced_accuracy={bal:.4f} macro_f1={macro:.4f}')
  if macro>best:
   best=macro
   registry_path=ROOT/'configs/classes.json'
   try: registry={c['id']:c for c in json.loads(registry_path.read_text(encoding='utf-8')).get('classes',[])}
   except Exception: registry={}
   labels=[{'id':c,'name_ru':registry.get(c,{}).get('name_ru',c.replace('_',' ').title()),'name_en':registry.get(c,{}).get('name_en',c.replace('_',' ').title())} for c in classes]
   out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True);torch.save({'state_dict':model.cpu().state_dict(),'classes':labels,'image_size':args.image_size,'validation_macro_f1':float(macro),'validation_balanced_accuracy':float(bal),'training_note':'Research checkpoint; validation split is image-level unless patient grouping is supplied.'},out);model.to(device)
 print('\nBest validation macro-F1:',best);print(classification_report(ys,preds,target_names=classes,zero_division=0));print('Saved:',args.output);print('IMPORTANT: image-level random split can overestimate performance when images share patients. Use patient-level split for a serious evaluation.')
if __name__=='__main__': main()
