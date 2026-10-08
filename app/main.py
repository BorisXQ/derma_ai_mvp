
from pathlib import Path
import io
import time
from collections import defaultdict, deque

from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

from .model import get_model, predict, LABELS_RU

APP_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = APP_DIR / "static"

MAX_FILE_SIZE = 8 * 1024 * 1024
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}
RATE_LIMIT = 20
RATE_WINDOW = 60

hits = defaultdict(deque)

app = FastAPI(
    title="Derma AI MVP",
    version="0.1.0",
    description="Research-only dermatology image classification MVP."
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

def check_rate_limit(request: Request):
    ip = request.client.host if request.client else "unknown"
    now = time.time()
    q = hits[ip]
    while q and now - q[0] > RATE_WINDOW:
        q.popleft()
    if len(q) >= RATE_LIMIT:
        raise HTTPException(status_code=429, detail="Too many requests. Try again in a minute.")
    q.append(now)

@app.get("/", include_in_schema=False)
async def index():
    return FileResponse(STATIC_DIR / "index.html")

@app.get("/api/v1/health")
async def health():
    return {"status": "ok", "service": "derma-ai-mvp"}

@app.get("/api/v1/model")
async def model_info():
    return {
        "model": "derm-cnn-ham10000",
        "architecture": "SkinCNN",
        "dataset": "HAM10000",
        "input": "28x28 RGB",
        "classes": list(LABELS_RU.values()),
        "device": "cuda" if __import__("torch").cuda.is_available() else "cpu",
        "warning": "Research/education only. Not a medical diagnosis."
    }

@app.post("/api/v1/analyze")
async def analyze(request: Request, file: UploadFile = File(...)):
    check_rate_limit(request)

    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=415, detail="Only JPEG, PNG and WEBP images are accepted.")

    data = await file.read()
    if len(data) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="Image is too large. Maximum size is 8 MB.")
    if not data:
        raise HTTPException(status_code=400, detail="Empty file.")

    try:
        image = Image.open(io.BytesIO(data))
        image.verify()
        image = Image.open(io.BytesIO(data)).convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="The uploaded file is not a valid image.")

    if image.width < 32 or image.height < 32:
        raise HTTPException(status_code=400, detail="Image is too small. Minimum is 32x32 pixels.")

    results = predict(image, top_k=3)
    best = results[0]

    return {
        "success": True,
        "filename": file.filename,
        "image": {"width": image.width, "height": image.height},
        "prediction": best,
        "top_predictions": results,
        "interpretation": (
            "Высокая уверенность" if best["confidence"] >= 0.80
            else "Средняя уверенность" if best["confidence"] >= 0.50
            else "Низкая уверенность"
        ),
        "warning": (
            "Это исследовательская предварительная классификация изображения, "
            "а не медицинский диагноз. Модель обучена на дерматоскопических изображениях "
            "HAM10000; качество на обычных фотографиях кожи может отличаться."
        )
    }
