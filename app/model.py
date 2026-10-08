
import json
import os
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn as nn

MODEL_DIR = Path(__file__).resolve().parent.parent / "model"
WEIGHTS_PATH = MODEL_DIR / "model.pth"
LABELS_PATH = MODEL_DIR / "labels.json"

class SkinCNN(nn.Module):
    def __init__(self, num_classes: int = 7):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2), nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2), nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2), nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            # The original model was trained with 28x28 inputs.
            # Adaptive pooling preserves the existing weights while allowing
            # inference on larger images without a fixed spatial dimension.
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(0.2),
            nn.Linear(256, 256), nn.ReLU(), nn.BatchNorm1d(256),
            nn.Linear(256, 128), nn.ReLU(), nn.BatchNorm1d(128),
            nn.Linear(128, 64), nn.ReLU(), nn.BatchNorm1d(64),
            nn.Linear(64, 32), nn.ReLU(), nn.BatchNorm1d(32),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

LABELS = {
    "akiec": "Actinic keratoses",
    "bcc": "Basal cell carcinoma",
    "bkl": "Benign keratosis",
    "df": "Dermatofibroma",
    "nv": "Melanocytic nevus",
    "vasc": "Vascular lesion",
    "mel": "Melanoma",
}

LABELS_RU = {
    "akiec": "Актинический кератоз",
    "bcc": "Базальноклеточная карцинома",
    "bkl": "Доброкачественный кератоз",
    "df": "Дерматофиброма",
    "nv": "Меланоцитарный невус",
    "vasc": "Сосудистое поражение",
    "mel": "Меланома",
}

def ensure_weights():
    if WEIGHTS_PATH.exists():
        return
    import urllib.request
    url = os.environ.get(
        "MODEL_WEIGHTS_URL",
        "https://huggingface.co/iamhmh/derm-cnn-ham10000/resolve/main/model.pth?download=true"
    )
    WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, WEIGHTS_PATH)

_model = None
_device = None

def get_model():
    global _model, _device
    if _model is not None:
        return _model, _device

    ensure_weights()
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SkinCNN(num_classes=7)
    state = torch.load(WEIGHTS_PATH, map_location=_device, weights_only=True)
    model.load_state_dict(state)
    model.to(_device)
    model.eval()
    _model = model
    return _model, _device

def _prepare_image(image: Image.Image, size: int = 224) -> Image.Image:
    """Prepare arbitrary phone/camera images for inference.

    Keep aspect ratio and pad instead of stretching a portrait/landscape photo.
    224x224 gives the convolutional network more spatial information before
    adaptive pooling. The model weights remain backward-compatible with the
    original 28x28 checkpoint.
    """
    image = image.convert("RGB")
    image.thumbnail((size, size), Image.Resampling.LANCZOS)

    canvas = Image.new("RGB", (size, size), (0, 0, 0))
    x = (size - image.width) // 2
    y = (size - image.height) // 2
    canvas.paste(image, (x, y))
    return canvas

def predict(image: Image.Image, top_k: int = 3):
    model, device = get_model()
    image = _prepare_image(image, size=224)
    arr = np.asarray(image).astype("float32") / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    tensor = torch.from_numpy(arr).unsqueeze(0).to(device)

    with torch.inference_mode():
        probs = torch.softmax(model(tensor), dim=1)[0]

    values, indices = torch.topk(probs, k=min(top_k, len(probs)))
    results = []
    ordered_keys = ["akiec", "bcc", "bkl", "df", "nv", "vasc", "mel"]
    for value, index in zip(values.cpu().tolist(), indices.cpu().tolist()):
        key = ordered_keys[index]
        results.append({
            "code": key,
            "disease": LABELS_RU[key],
            "disease_en": LABELS[key],
            "confidence": round(float(value), 4),
            "confidence_percent": round(float(value) * 100, 2),
        })

    return results
