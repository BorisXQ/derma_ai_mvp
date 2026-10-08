# Derma AI MVP — fixed input pipeline

Public web MVP for preliminary dermatology image classification.

## What was fixed

- The CNN no longer has a hard dependency on a `28x28` spatial output.
- Added `AdaptiveAvgPool2d(1, 1)`, so the existing checkpoint remains compatible while the network can process larger tensors.
- Uploaded images are EXIF-rotated correctly (important for phone photos).
- Inference preprocessing now accepts arbitrary image dimensions, preserves aspect ratio, and letterboxes to `224x224` instead of directly squashing every image to `28x28`.
- The API metadata now reports the real inference preprocessing.
- The medical limitation is stated explicitly: the checkpoint is still trained on HAM10000 dermatoscopic images.

## Important medical limitation

This is a research/education prototype, NOT a medical diagnostic system.

Changing inference from 28x28 to 224x224 does **not** retrain the model on smartphone photographs. It removes the technical 28x28 bottleneck, but it cannot guarantee better clinical accuracy. In particular, if a lesion occupies only a small part of a full phone photo, the model can still fail.

For a genuinely useful smartphone-photo system, the next ML step should be a model trained/validated on clinical photographs, ideally with lesion localization/cropping and patient/case-level validation.

The model weights come from:
https://huggingface.co/iamhmh/derm-cnn-ham10000

The model and HAM10000 data are under CC BY-NC 4.0. This MVP is therefore NOT cleared for commercial use without reviewing the applicable licenses.

## Deploy on Render

1. Replace the files in your GitHub repository with the contents of this project.
2. Keep the existing Render Web Service connected to that GitHub repository.
3. Push/commit the replacement files to the same branch Render deploys.
4. Render will detect the new commit and start a new deploy automatically.
5. Build command stays:
   `pip install -r requirements.txt`
6. Start command stays:
   `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
7. After deployment finishes, open your existing `.onrender.com` URL.

If Render does not auto-deploy, open the service in Render and use **Manual Deploy → Deploy latest commit**.

The model weights are downloaded automatically on first inference if `model/model.pth` is not already present. Render Free services may sleep when idle, so the first request after sleeping can take longer.

## API

- `GET /api/v1/health`
- `GET /api/v1/model`
- `POST /api/v1/analyze` with `multipart/form-data` field `file`

## Local development

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate

pip install -r requirements.txt
uvicorn app.main:app --reload
```

Then open `http://127.0.0.1:8000`.

## Future ML roadmap

For a real smartphone-photo model:

1. Use appropriately licensed clinical-photo datasets.
2. Split by patient/case, not by individual image.
3. Train at an appropriate resolution with augmentation.
4. Add lesion localization/cropping or a segmentation/detection stage.
5. Validate on an external clinical dataset.
6. Calibrate probabilities and report sensitivity/specificity, not only accuracy.
7. Add a clear abstain/low-quality path.
8. Keep the product positioned as decision support unless/ until clinically validated.

## V2 model improvements

- 224x224 inference instead of fixed 28x28 spatial input.
- Aspect-ratio-preserving preprocessing.
- EXIF orientation handling.
- Test-time augmentation: full image + focused center crop + mirrored crop.
- `train.py` for real fine-tuning with augmentation, class-balanced sampling and validation.

The included checkpoint is the existing compatible HAM10000 checkpoint; it is **not claimed to be newly trained** on smartphone photos. See `docs/RETRAIN.md` for real fine-tuning.
