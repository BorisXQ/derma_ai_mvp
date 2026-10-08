import os, json, urllib.request
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / 'model'
CUSTOM_PATH = MODEL_DIR / 'custom_model.pth'
BASE_PATH = MODEL_DIR / 'efficientnet_b0_320.pth'
BASE_URL = os.environ.get('MODEL_WEIGHTS_URL', 'https://huggingface.co/adeIaide/skin-lesion-7class-efficientnet/resolve/main/isic2018_7class_efficientnet_b0_320.pt?download=true')
DEFAULT_CLASSES = [
 {'id':'akiec','name_ru':'Актинический кератоз','name_en':'Actinic keratoses'},
 {'id':'bcc','name_ru':'Базальноклеточная карцинома','name_en':'Basal cell carcinoma'},
 {'id':'bkl','name_ru':'Доброкачественный кератоз','name_en':'Benign keratosis'},
 {'id':'df','name_ru':'Дерматофиброма','name_en':'Dermatofibroma'},
 {'id':'nv','name_ru':'Меланоцитарный невус','name_en':'Melanocytic nevus'},
 {'id':'vasc','name_ru':'Сосудистое поражение','name_en':'Vascular lesion'},
 {'id':'mel','name_ru':'Меланома','name_en':'Melanoma'}]
INPUT_SIZE = 320
MEAN = np.array([0.485,0.456,0.406], dtype=np.float32)
STD = np.array([0.229,0.224,0.225], dtype=np.float32)
_model = None
_device = None
_classes = None
_model_kind = None

def read_registry():
    path=ROOT/'configs/classes.json'
    try:
        data=json.loads(path.read_text(encoding='utf-8'))
        return [c for c in data.get('classes',[]) if c.get('enabled',True)]
    except Exception:
        return DEFAULT_CLASSES

def _ensure_base_weights():
    if BASE_PATH.exists() and BASE_PATH.stat().st_size > 1_000_000: return
    MODEL_DIR.mkdir(parents=True,exist_ok=True)
    tmp=BASE_PATH.with_suffix('.download')
    urllib.request.urlretrieve(BASE_URL,tmp)
    tmp.replace(BASE_PATH)

def build_model(n_classes):
    m=efficientnet_b0(weights=None)
    m.classifier[1]=nn.Linear(m.classifier[1].in_features,n_classes)
    return m

def _state_dict(obj):
    if isinstance(obj,dict):
        for k in ('state_dict','model_state_dict','model'):
            if k in obj and isinstance(obj[k],dict): obj=obj[k]; break
    if not isinstance(obj,dict): raise RuntimeError('Unsupported checkpoint format')
    out={}
    for k,v in obj.items():
        while k.startswith('module.'): k=k[7:]
        if k.startswith('model.'): k=k[6:]
        out[k]=v
    return out

def get_model():
    global _model,_device,_classes,_model_kind
    if _model is not None: return _model,_device
    _device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if CUSTOM_PATH.exists():
        checkpoint=torch.load(CUSTOM_PATH,map_location=_device,weights_only=False)
        _classes=checkpoint.get('classes')
        if not _classes: raise RuntimeError('custom_model.pth has no class metadata')
        model=build_model(len(_classes))
        model.load_state_dict(_state_dict(checkpoint),strict=True)
        _model_kind='custom-trained'
    else:
        _ensure_base_weights()
        model=build_model(7)
        state=_state_dict(torch.load(BASE_PATH,map_location=_device,weights_only=False))
        model.load_state_dict(state,strict=True)
        _classes=DEFAULT_CLASSES
        _model_kind='published-seven-class-checkpoint'
    model.to(_device).eval(); _model=model
    return _model,_device

def _square_crop(image, fraction=0.88, center=(0.5,0.5)):
    side=max(1,int(min(image.width,image.height)*fraction)); cx,cy=center
    left=max(0,min(image.width-side,int(image.width*cx-side/2)))
    top=max(0,min(image.height-side,int(image.height*cy-side/2)))
    return image.crop((left,top,left+side,top+side))

def _letterbox(image,size=INPUT_SIZE):
    image=image.convert('RGB'); image.thumbnail((size,size),Image.Resampling.LANCZOS)
    canvas=Image.new('RGB',(size,size),(128,128,128)); canvas.paste(image,((size-image.width)//2,(size-image.height)//2))
    return canvas

def _tensor(image):
    arr=np.asarray(_letterbox(image)).astype('float32')/255.0
    arr=(arr-MEAN)/STD
    return torch.from_numpy(np.transpose(arr,(2,0,1)).copy())

def _views(image):
    image=image.convert('RGB')
    crops=[image,_square_crop(image,.92),_square_crop(image,.82),_square_crop(image,.82,(.45,.45)),_square_crop(image,.82,(.55,.55))]
    return [_tensor(x) for x in crops]

def predict(image,top_k=3):
    model,device=get_model(); batch=torch.stack(_views(image)).to(device)
    with torch.inference_mode(): probs=torch.softmax(model(batch),dim=1).mean(dim=0)
    values,indices=torch.topk(probs,k=min(top_k,len(_classes)))
    out=[]
    for value,index in zip(values.cpu().tolist(),indices.cpu().tolist()):
        c=_classes[index]
        out.append({'code':c['id'],'disease':c.get('name_ru',c['id']),'disease_en':c.get('name_en',c['id']),
                    'confidence':round(float(value),4),'confidence_percent':round(float(value)*100,2)})
    return out

def model_info():
    get_model()
    return {'model':'EfficientNet-B0','model_kind':_model_kind,'input_size':INPUT_SIZE,'classes':_classes,
            'device':str(_device),'custom_model_loaded':_model_kind=='custom-trained'}
