# Derma AI Universal — extensible research prototype

This repository makes **adding a class easy**, but it cannot make a model know a new disease merely by adding a label. New classes need many correctly labeled, diverse images and a new training run. The project is not a medical device and must not be used to diagnose or rule out disease.

## Features
- FastAPI image analysis API and web UI.
- `/admin` class management: add a class and upload labeled image examples.
- `configs/classes.json` is the editable class registry.
- `data/dataset/<class_id>/` is the folder structure expected by training.
- `train.py` discovers class folders automatically and trains an EfficientNet-B0 head with ImageNet pretrained weights, weighted sampling and augmentation.
- When `model/custom_model.pth` exists, inference uses its dynamically sized head and stored class metadata. Without it, inference falls back to the published 7-class ISIC/HAM10000 checkpoint.

## Add a disease
1. Set a strong `DERMA_ADMIN_TOKEN` environment variable in Render or your local shell.
2. Open `/admin`, enter the token, and create a class, e.g. `acne` / `Акне`.
3. Upload verified, consented, correctly labeled examples for that class. The admin UI stores them under `data/dataset/<class_id>/` on the running server. **Render's filesystem may be ephemeral**; for real projects use a persistent disk or, preferably, store datasets in a controlled external location and train locally.
4. Build a local dataset in folders, for example:

```text
data/dataset/
  acne/        # many reviewed examples
  rosacea/
  eczema/
  psoriasis/
  akiec/
  bcc/
  bkl/
  df/
  mel/
  nv/
  vasc/
```

5. Train locally: `python train.py --data data/dataset --epochs 8 --image-size 320`.
6. Inspect per-class report, balanced accuracy and macro-F1. Do not deploy a checkpoint just because training completed. Test on an independent patient-level dataset from the intended phone-photo setting.
7. Deploy the resulting `model/custom_model.pth` and matching class metadata together. Do not commit sensitive patient photos to a public GitHub repository.

## Run locally
Python 3.11 recommended. Install dependencies with `pip install -r requirements.txt`, then `uvicorn app.main:app --host 0.0.0.0 --port 8000`. Set `DERMA_ADMIN_TOKEN` before using admin write actions. On first inference, if no custom checkpoint exists, the app downloads the published 7-class checkpoint; deployment therefore needs outbound internet unless the weights are packaged.

## Render
- Build command: `pip install -r requirements.txt`
- Start command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
- Set `DERMA_ADMIN_TOKEN` in Render Environment.
- Use persistent storage for uploaded examples if you intentionally collect them on Render. Otherwise use local/external storage and keep only model artifacts in deployment.

## Accuracy limitations
The fallback checkpoint is a 7-class model trained primarily on dermoscopic lesion images (HAM10000 / ISIC 2018 Task 3). It does not natively recognize acne, rosacea, eczema, psoriasis, or all skin diseases. A new registry entry is marked as not trained until a custom checkpoint contains that class. Softmax scores are not calibrated clinical probabilities. The training script is a starting point, not clinical validation; its default split is image-level and can leak patient-specific information if multiple images per patient are present. For credible performance, use patient-level splits, external validation, sufficient examples per class, diverse skin tones/devices, calibration, and dermatologist oversight.
