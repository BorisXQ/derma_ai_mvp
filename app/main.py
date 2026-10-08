from pathlib import Path
import io, json, os, re, time, secrets, zipfile
from collections import defaultdict, deque
from fastapi import FastAPI, File, UploadFile, HTTPException, Request, Header, Form
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps, UnidentifiedImageError
from .model import predict, model_info, read_registry

ROOT=Path(__file__).resolve().parent.parent
STATIC=ROOT/'static'; CONFIG=ROOT/'configs/classes.json'; DATA=ROOT/'data/dataset'
MAX_FILE_SIZE=12*1024*1024; ALLOWED={'image/jpeg','image/png','image/webp'}
RATE_LIMIT=20; RATE_WINDOW=60; hits=defaultdict(deque)
app=FastAPI(title='Derma AI Universal Research Platform',version='4.0.0',description='Extensible research prototype; not a diagnostic device.')
app.mount('/static',StaticFiles(directory=STATIC),name='static')

def rate_limit(request):
    ip=request.client.host if request.client else 'unknown'; now=time.time(); q=hits[ip]
    while q and now-q[0]>RATE_WINDOW:q.popleft()
    if len(q)>=RATE_LIMIT: raise HTTPException(429,'Too many requests. Try again in a minute.')
    q.append(now)

def load_config():
    try: return json.loads(CONFIG.read_text(encoding='utf-8'))
    except Exception: return {'schema_version':1,'classes':read_registry()}

def save_config(obj):
    CONFIG.parent.mkdir(parents=True,exist_ok=True); tmp=CONFIG.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8'); tmp.replace(CONFIG)

def admin_auth(authorization):
    expected=os.environ.get('DERMA_ADMIN_TOKEN','').strip()
    if not expected: raise HTTPException(503,'Admin is disabled. Set DERMA_ADMIN_TOKEN in environment first.')
    supplied=(authorization or '').removeprefix('Bearer ').strip()
    if not secrets.compare_digest(supplied,expected): raise HTTPException(401,'Invalid admin token.')

def valid_id(value): return bool(re.fullmatch(r'[a-z0-9][a-z0-9_-]{1,39}',value or ''))

@app.get('/',include_in_schema=False)
async def index(): return FileResponse(STATIC/'index.html')
@app.get('/admin',include_in_schema=False)
async def admin(): return FileResponse(STATIC/'admin.html')
@app.get('/api/v1/health')
async def health(): return {'status':'ok','service':'derma-ai-universal','version':'4.0.0'}
@app.get('/api/v1/model')
async def model_endpoint():
    info=model_info(); info['warning']='Research only. Not a medical diagnosis. Existing checkpoint is trained on dermoscopic images and supports only its original seven classes until a custom model is trained.'
    return info
@app.get('/api/v1/classes')
async def list_classes():
    cfg=load_config()
    custom=ROOT/'model/custom_model.pth'
    if custom.exists():
        try:
            import torch
            ckpt=torch.load(custom,map_location='cpu',weights_only=False)
            trained={c['id'] for c in ckpt.get('classes',[])}
            model_kind='custom-trained'
        except Exception:
            trained=set(); model_kind='custom-checkpoint-invalid'
    else:
        trained={'akiec','bcc','bkl','df','nv','vasc','mel'}
        model_kind='published-seven-class-checkpoint'
    result=[]
    for c in cfg.get('classes',[]):
        x=dict(c); x['trained_in_current_model']=x.get('id') in trained; x['sample_count']=len(list((DATA/x.get('id','')).glob('*'))) if (DATA/x.get('id','')).exists() else 0
        result.append(x)
    return {'classes':result,'model_kind':model_kind}
@app.post('/api/v1/admin/classes')
async def add_class(request:Request,authorization:str|None=Header(default=None),id:str=Form(...),name_ru:str=Form(...),name_en:str=Form(''),description:str=Form('')):
    admin_auth(authorization)
    id=id.strip().lower(); name_ru=name_ru.strip(); name_en=name_en.strip(); description=description.strip()
    if not valid_id(id): raise HTTPException(400,'ID: 2-40 символов, латиница в нижнем регистре, цифры, _ или -; первый символ — буква/цифра.')
    if not name_ru or len(name_ru)>100: raise HTTPException(400,'Введите название до 100 символов.')
    cfg=load_config(); classes=cfg.setdefault('classes',[])
    if any(c.get('id')==id for c in classes): raise HTTPException(409,'Класс с таким ID уже существует.')
    classes.append({'id':id,'name_ru':name_ru,'name_en':name_en or id,'description':description,'enabled':True,'source':'custom; requires labelled examples and retraining'})
    save_config(cfg); (DATA/id).mkdir(parents=True,exist_ok=True)
    return {'ok':True,'class':classes[-1],'message':'Класс создан. Модель пока НЕ обучена распознавать его: добавьте размеченные изображения и переобучите модель.'}
@app.post('/api/v1/admin/classes/{class_id}/images')
async def upload_samples(class_id:str,files:list[UploadFile]=File(...),authorization:str|None=Header(default=None)):
    admin_auth(authorization)
    cfg=load_config()
    if not any(c.get('id')==class_id for c in cfg.get('classes',[])): raise HTTPException(404,'Unknown class')
    if len(files)>100: raise HTTPException(400,'Максимум 100 изображений за одну загрузку.')
    folder=DATA/class_id; folder.mkdir(parents=True,exist_ok=True); saved=[]
    for f in files:
        if f.content_type not in ALLOWED: continue
        data=await f.read()
        if not data or len(data)>MAX_FILE_SIZE: continue
        try:
            im=Image.open(io.BytesIO(data)); im.verify(); im=ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert('RGB')
            if min(im.size)<64: continue
            ext={'image/jpeg':'.jpg','image/png':'.png','image/webp':'.webp'}[f.content_type]
            name=f'{int(time.time()*1000)}_{secrets.token_hex(4)}{ext}'; im.save(folder/name,quality=94); saved.append(name)
        except (UnidentifiedImageError,OSError): continue
    return {'ok':True,'saved':len(saved),'files':saved,'message':'Images saved as training examples. Verify labels/consent and retrain locally; upload storage on Render may be ephemeral.'}
@app.get('/api/v1/admin/dataset.zip')
async def export_dataset(authorization:str|None=Header(default=None)):
    admin_auth(authorization)
    mem=io.BytesIO()
    with zipfile.ZipFile(mem,'w',zipfile.ZIP_DEFLATED) as z:
        for path in DATA.rglob('*'):
            if path.is_file() and path.suffix.lower() in {'.jpg','.jpeg','.png','.webp'}:
                z.write(path,Path('dataset')/path.relative_to(DATA))
        z.writestr('README.txt','Exported Derma AI labeled examples. Verify all labels before training. These images may contain sensitive personal data; store securely.\n')
    return Response(mem.getvalue(),media_type='application/zip',headers={'Content-Disposition':'attachment; filename=derma_ai_dataset.zip'})

@app.post('/api/v1/analyze')
async def analyze(request:Request,file:UploadFile=File(...)):
    rate_limit(request)
    if file.content_type not in ALLOWED: raise HTTPException(415,'Only JPEG, PNG and WEBP images are accepted.')
    data=await file.read()
    if len(data)>MAX_FILE_SIZE: raise HTTPException(413,'Maximum image size is 12 MB.')
    if not data: raise HTTPException(400,'Empty file.')
    try:
        Image.open(io.BytesIO(data)).verify(); image=ImageOps.exif_transpose(Image.open(io.BytesIO(data)).convert('RGB'))
    except (UnidentifiedImageError,OSError): raise HTTPException(400,'Invalid image.')
    if min(image.size)<64: raise HTTPException(400,'Minimum image dimensions are 64x64.')
    try:
        results=predict(image,top_k=3)
    except Exception as exc:
        # Return a structured JSON error rather than letting the browser receive an HTML 500 page.
        import logging
        logging.exception('Image inference failed')
        raise HTTPException(status_code=503, detail='Модель сейчас не смогла обработать изображение. Проверь логи Render: '+str(exc)[:500]) from exc
    if not results: raise HTTPException(status_code=503, detail='Модель не вернула прогноз.')
    best=results[0]
    if best['confidence']<.45 or (len(results)>1 and best['confidence']-results[1]['confidence']<.12): interp='Нужна дополнительная проверка'
    elif best['confidence']>=.80: interp='Высокая уверенность модели'
    else: interp='Умеренная уверенность модели'
    return {'success':True,'filename':file.filename,'image':{'width':image.width,'height':image.height},'prediction':best,'top_predictions':results,'interpretation':interp,'warning':'Исследовательская классификация, не медицинский диагноз. Проценты не являются клинической вероятностью. Модель может ошибаться, особенно на обычных фото смартфона.'}
