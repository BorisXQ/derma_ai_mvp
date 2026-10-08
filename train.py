"""Fine-tune SkinCNN on a folder dataset.

Expected layout:
  data/train/{akiec,bcc,bkl,df,nv,vasc,mel}/*.jpg
  data/val/{akiec,bcc,bkl,df,nv,vasc,mel}/*.jpg

This script uses ImageNet-style augmentation and a WeightedRandomSampler to
reduce the strong HAM10000 class imbalance. It is intentionally NOT run on
Render: training should be done on a machine with a GPU, then the resulting
model/model.pth can be deployed with the web app.
"""
import argparse
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms
from app.model import SkinCNN, ORDERED_KEYS

IMG = 224
TRAIN_TF = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(IMG, scale=(0.72, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.12),
    transforms.ToTensor(),
])
VAL_TF = transforms.Compose([
    transforms.Resize((IMG, IMG)),
    transforms.ToTensor(),
])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data')
    ap.add_argument('--epochs', type=int, default=12)
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--out', default='model/model.pth')
    args = ap.parse_args()

    root = Path(args.data)
    train_ds = datasets.ImageFolder(root/'train', transform=TRAIN_TF)
    val_ds = datasets.ImageFolder(root/'val', transform=VAL_TF)
    expected = {k: i for i, k in enumerate(ORDERED_KEYS)}
    if train_ds.class_to_idx != expected or val_ds.class_to_idx != expected:
        raise RuntimeError(f'Classes must be exactly {expected}; got train={train_ds.class_to_idx}, val={val_ds.class_to_idx}')

    counts = torch.bincount(torch.tensor(train_ds.targets), minlength=7).float()
    weights = 1.0 / counts.clamp_min(1)
    sample_weights = weights[torch.tensor(train_ds.targets)]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SkinCNN(7).to(device)
    ckpt = Path('model/model.pth')
    if ckpt.exists():
        try:
            model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
            print('Loaded existing checkpoint for fine-tuning:', ckpt)
        except Exception as e:
            print('Could not load existing checkpoint; training from scratch:', e)

    # Class-weighted CE + balanced sampler gives minority classes more signal.
    criterion = nn.CrossEntropyLoss(weight=(weights / weights.mean()).to(device))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    best = 0.0
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train(); total = correct = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward(); opt.step()
            total += y.numel(); correct += (model(x).argmax(1) == y).sum().item()
        model.eval(); vtotal = vcorrect = 0
        with torch.inference_mode():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                vcorrect += (model(x).argmax(1) == y).sum().item(); vtotal += y.numel()
        vacc = vcorrect / max(1, vtotal)
        print(f'epoch={epoch} train_acc={correct/max(1,total):.4f} val_acc={vacc:.4f}')
        if vacc > best:
            best = vacc
            torch.save(model.state_dict(), out)
            print('saved', out)

if __name__ == '__main__':
    main()
