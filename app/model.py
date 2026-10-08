"""Model loading for Derma AI.

Default mode keeps compatibility with the original published HAM10000 SkinCNN
checkpoint. Custom models are EfficientNet-B0 checkpoints created by train.py.
Research use only; this is not a validated diagnostic system.
"""
import io, json, os, urllib.request
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / 'model'
BASE_PATH = MODEL_DIR / 'model.pth'
CUSTOM_PATH = MODEL_DIR / 'custom_model.pth'
BASE_URL = os.environ.get('MODEL_WEIGHTS_URL', 'https://huggingface.co/iamhmh/derm-cnn-ham10000/resolve/main/model.pth?download=true')
DEFAULT_CLASSES = [
 {'id':'akiec','name_ru':'Актинический кератоз','name_en':'Actinic keratoses'},
 {'id':'bcc','name_ru':'Базальноклеточная карцинома','name_en':'Basal cell carcinoma'},
 {'id':'bkl','name_ru':'Доброкачественный кератоз','name_en':'Benign keratosis'},
 {'id':'df','name_ru':'Дерматофиброма','name_en':'Dermatofibroma'},
 {'id':'nv','name_ru':'Меланоцитарный невус','name_en':'Melanocytic nevus'},
 {'id':'vasc','name_ru':'Сосудистое поражение','name_en':'Vascular lesion'},
 {'id':'mel','name_ru':'Меланома','name_en':'Melanoma'}]
INPUT_SIZE = 224
MEAN = np.array([0.485,0.456,0.406], dtype=np.float32)
STD = np.array([0.229,0.224,0.225], dtype=np.float32)

class SkinCNN(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.features=nn.Sequential(
            nn.Conv2d(3,32,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),nn.BatchNorm2d(32),
            nn.Conv2d(32,64,3,padding=1),nn.ReLU(),nn.Conv2d(64,64,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),nn.BatchNorm2d(64),
            nn.Conv2d(64,128,3,padding=1),nn.ReLU(),nn.Conv2d(128,128,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),nn.BatchNorm2d(128),
            nn.Conv2d(128,256,3,padding=1),nn.ReLU(),nn.Conv2d(256,256,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),nn.AdaptiveAvgPool2d((1,1)))
        self.classifier=nn.Sequential(nn.Flatten(),nn.Dropout(.2),nn.Linear(256,256),nn.ReLU(),nn.BatchNorm1d(256),nn.Linear(256,128),nn.ReLU(),nn.BatchNorm1d(128),nn.Linear(128,64),nn.ReLU(),nn.BatchNorm1d(64),nn.Linear(64,32),nn.ReLU(),nn.BatchNorm1d(32),nn.Linear(32,num_classes))
    def forward(self,x): return self.classifier(self.features(x))

_model=None; _device=None; _classes=None; _model_kind=None; _custom=False

def read_registry():
    path=ROOT/'configs/classes.json'
    try: return [c for c in json.loads(path.read_text(encoding='utf-8')).get('classes',[]) if c.get('enabled',True)]
    except Exception: return DEFAULT_CLASSES

def _download_default_weights():
    if BASE_PATH.exists() and BASE_PATH.stat().st_size > 100_000: return
    MODEL_DIR.mkdir(parents=True,exist_ok=True); tmp=BASE_PATH.with_suffix('.download')
    try:
        req=urllib.request.Request(BASE_URL,headers={'User-Agent':'DermaAI-research/1.0'})
        with urllib.request.urlopen(req,timeout=90) as response, open(tmp,'wb') as out:
            while True:
                chunk=response.read(1024*1024)
                if not chunk: break
                out.write(chunk)
        if tmp.stat().st_size < 100_000: raise RuntimeError('Downloaded model file is too small; check MODEL_WEIGHTS_URL.')
        tmp.replace(BASE_PATH)
    except Exception as exc:
        try: tmp.unlink(missing_ok=True)
        except Exception: pass
        raise RuntimeError('Не удалось загрузить базовые веса модели. Проверь URL MODEL_WEIGHTS_URL и логи Render. Подробности: '+str(exc)) from exc

def _clean_state(obj):
    if isinstance(obj,dict):
        for key in ('state_dict','model_state_dict','model'):
            if key in obj and isinstance(obj[key],dict): obj=obj[key]; break
    if not isinstance(obj,dict): raise RuntimeError('Файл весов имеет неподдерживаемый формат.')
    out={}
    for k,v in obj.items():
        while k.startswith('module.'): k=k[7:]
        out[k]=v
    return out

def get_model():
    global _model,_device,_classes,_model_kind,_custom
    if _model is not None: return _model,_device
    _device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if CUSTOM_PATH.exists():
        from torchvision.models import efficientnet_b0
        ckpt=torch.load(CUSTOM_PATH,map_location=_device,weights_only=False)
        _classes=ckpt.get('classes')
        if not _classes: raise RuntimeError('custom_model.pth не содержит список классов. Запусти обучение из текущего архива.')
        model=efficientnet_b0(weights=None); model.classifier[1]=nn.Linear(model.classifier[1].in_features,len(_classes))
        state=ckpt.get('state_dict',ckpt)
        model.load_state_dict(_clean_state(state),strict=True); _model_kind='custom-trained'; _custom=True
    else:
        _download_default_weights()
        model=SkinCNN(num_classes=7)
        state=torch.load(BASE_PATH,map_location=_device,weights_only=True)
        model.load_state_dict(_clean_state(state),strict=True)
        _classes=DEFAULT_CLASSES; _model_kind='original-seven-class-checkpoint'; _custom=False
    model.to(_device).eval(); _model=model
    return _model,_device

def _letterbox(image,size=INPUT_SIZE):
    image=image.convert('RGB'); image.thumbnail((size,size),Image.Resampling.LANCZOS)
    canvas=Image.new('RGB',(size,size),(128,128,128)); canvas.paste(image,((size-image.width)//2,(size-image.height)//2)); return canvas

def _tensor(image):
    arr=np.asarray(_letterbox(image)).astype('float32')/255.0
    if _custom: arr=(arr-MEAN)/STD
    return torch.from_numpy(np.transpose(arr,(2,0,1)).copy())

def _crop(image,fraction=.88,center=(.5,.5)):
    side=max(1,int(min(image.width,image.height)*fraction)); cx,cy=center
    left=max(0,min(image.width-side,int(image.width*cx-side/2))); top=max(0,min(image.height-side,int(image.height*cy-side/2)))
    return image.crop((left,top,left+side,top+side))

def predict(image,top_k=3):
    model,device=get_model(); image=image.convert('RGB')
    views=[image,_crop(image,.92),_crop(image,.82),_crop(image,.82,(.45,.45)),_crop(image,.82,(.55,.55))]
    batch=torch.stack([_tensor(v) for v in views]).to(device)
    with torch.inference_mode(): probs=torch.softmax(model(batch),dim=1).mean(dim=0)
    values,indices=torch.topk(probs,k=min(top_k,len(_classes))); out=[]
    for value,index in zip(values.cpu().tolist(),indices.cpu().tolist()):
        c=_classes[index]
        out.append({'code':c['id'],'disease':c.get('name_ru',c['id']),'disease_en':c.get('name_en',c['id']),'confidence':round(float(value),4),'confidence_percent':round(float(value)*100,2)})
    return out

def model_info():
    get_model()
    return {'model':'EfficientNet-B0' if _custom else 'SkinCNN','model_kind':_model_kind,'input_size':INPUT_SIZE,'classes':_classes,'device':str(_device),'custom_model_loaded':_custom}
