# Derma AI MVP

Public web MVP for preliminary dermatology image classification.

## What is included

- Web interface
- Image upload
- JPEG/PNG/WEBP validation
- 8 MB upload limit
- Pillow image verification
- FastAPI backend
- `POST /api/v1/analyze`
- `GET /api/v1/health`
- `GET /api/v1/model`
- PyTorch inference
- 7-class HAM10000 skin-lesion model
- Top-3 predictions
- Confidence score
- No intentional permanent storage of uploaded images
- Render deployment configuration

## Important limitation

The bundled model is a small CNN trained on HAM10000 dermatoscopic images. It is a research/education model and is NOT a clinical diagnostic system. It may perform poorly on ordinary smartphone/clinical photos.

The model weights come from:
https://huggingface.co/iamhmh/derm-cnn-ham10000

The model and HAM10000 data are under CC BY-NC 4.0. This MVP is therefore NOT cleared for commercial use. Replace the model and perform a license/compliance review before monetization.

## Deploy for free on Render

1. Create a GitHub repository.
2. Upload all files from this folder.
3. Open https://render.com/
4. Create a Web Service.
5. Connect the GitHub repository.
6. Runtime: Python.
7. Build command:
   pip install -r requirements.txt
8. Start command:
   uvicorn app.main:app --host 0.0.0.0 --port $PORT
9. Plan: Free.
10. Deploy.

The first request after the service sleeps can take longer. The model weights are downloaded automatically on first inference.

## API

GET /api/v1/health

GET /api/v1/model

POST /api/v1/analyze
multipart/form-data:
file=<image>

## Local development

Not required for deployment. If you later want to debug locally:

python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload

Then open http://127.0.0.1:8000

## Future replacement

For the real ML roadmap, replace the MVP classifier with a properly evaluated clinical-image pipeline:
SCIN / PAD-UFES-20 / other appropriately licensed datasets -> patient/case-level split -> training -> external validation -> calibration -> Grad-CAM -> API.
