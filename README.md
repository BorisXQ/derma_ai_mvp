# Derma AI MVP V3

V3 replaces the original 28x28 custom CNN with an EfficientNet-B0 7-class classifier using 320x320 ImageNet-style preprocessing and multi-view test-time augmentation.

## Important
This is a research/education classifier, not a medical diagnostic device. The default checkpoint is trained primarily on dermoscopic HAM10000/ISIC 2018 images. It can be substantially less reliable on ordinary smartphone photos.

## Deployment
Keep the existing Render service connected to GitHub. Replace the repository files with this project and commit. Render will rebuild automatically.

Start command:
`uvicorn app.main:app --host 0.0.0.0 --port $PORT`

The model checkpoint is downloaded on first startup from the configured `MODEL_WEIGHTS_URL`. You can override it with a Render environment variable pointing to a compatible EfficientNet-B0 7-class checkpoint.

## Real fine-tuning
For the strongest version, train with `python train_v3.py --data data --epochs 18`, using HAM10000 metadata and images. The script uses lesion_id-aware splitting, weighted sampling and macro-F1 model selection. For smartphone performance, fine-tune again on clinical photographs with reliable labels; HAM10000 itself is primarily dermoscopic.

## Why V3
The old checkpoint explicitly expected 28x28 input. V3 uses a much stronger ImageNet-pretrained EfficientNet-B0 and 320px inputs. The inference endpoint also avoids calling raw softmax output a medical probability and can report that the model needs additional verification when confidence/margin is weak.
