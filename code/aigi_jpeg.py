#!/usr/bin/env python3
"""
CE7454 - 3-class AIGI detection (0 real, 1 Model A, 2 Model B).
Same model as aigi_simple.py, plus training-only JPEG re-compression augmentation
(quality 75-98) for robustness to compression differences between splits.

Run:           python aigi_jpeg.py
Predict only:  PREDICT_ONLY=1 python aigi_jpeg.py
"""
import io, json, os, random, time, zipfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

# ============================== CONFIG ==============================
ROOT = "CE7454_project/final_dataset"
TRAIN_DIR = os.path.join(ROOT, "train")
DEV_DIR = os.path.join(ROOT, "dev")
TEST_DIR = os.path.join(ROOT, "test")
OUT_DIR = "CE7454_project/outputs_jpeg"

SEED = 42
EPOCHS = 30
MAX_MINUTES = 35          # stop training early if this wall-clock limit is reached
CROP = 256
BATCH = 32
LR = 1e-3
VAL_FRACTION = 0.05
NUM_WORKERS = min(8, os.cpu_count() or 4)

JPEG_P = 0.8              # fraction of training images re-compressed
JPEG_Q = (75, 98)         # broad quality range: general robustness to JPEG re-compression
VAL_JPEG_Q = 91           # val images re-compressed once, so val acc estimates test acc
# ====================================================================


# ------------------------------ model ------------------------------
def group_norm(c):
    return nn.GroupNorm(8, c)


class ResidualUnit(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.norm1 = group_norm(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.norm2 = group_norm(cout)
        self.shortcut = None
        if stride != 1 or cin != cout:
            self.shortcut = nn.Sequential(
                nn.Conv2d(cin, cout, 1, stride, bias=False), group_norm(cout)
            )

    def forward(self, x):
        out = F.silu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        skip = x if self.shortcut is None else self.shortcut(x)
        return F.silu(out + skip)


class NoiseResidualFilter(nn.Module):
    """Fixed (non-trainable) SRM-style high-pass filters, per RGB channel."""

    def __init__(self):
        super().__init__()
        k1 = torch.tensor([[0, 0, 0, 0, 0],
                           [0, -1, 2, -1, 0],
                           [0, 2, -4, 2, 0],
                           [0, -1, 2, -1, 0],
                           [0, 0, 0, 0, 0]], dtype=torch.float32) / 4.0
        k2 = torch.tensor([[-1, 2, -2, 2, -1],
                           [2, -6, 8, -6, 2],
                           [-2, 8, -12, 8, -2],
                           [2, -6, 8, -6, 2],
                           [-1, 2, -2, 2, -1]], dtype=torch.float32) / 12.0
        k3 = torch.zeros(5, 5)
        k3[2, 1], k3[2, 2] = 1.0, -1.0
        kernels = torch.stack([k1, k2, k3]).unsqueeze(1).repeat(3, 1, 1, 1)  # (9,1,5,5)
        self.register_buffer("kernels", kernels)  # buffer => not trainable

    def forward(self, x):
        x = x.repeat_interleave(3, dim=1)  # (B,9,H,W)
        r = F.conv2d(x, self.kernels.to(x.dtype), padding=2, groups=9)
        return torch.clamp(r, -3.0, 3.0)


class DualStreamForensicNet(nn.Module):
    def __init__(self, num_classes=3, drop=0.3):
        super().__init__()
        self.noise_filter = NoiseResidualFilter()
        self.color_stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1, bias=False), group_norm(32), nn.SiLU(inplace=True))
        self.noise_stem = nn.Sequential(
            nn.Conv2d(9, 32, 3, 1, 1, bias=False), group_norm(32), nn.SiLU(inplace=True))
        self.merge = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1, bias=False), group_norm(64), nn.SiLU(inplace=True))

        def stage(cin, cout):
            return nn.Sequential(ResidualUnit(cin, cout, 2), ResidualUnit(cout, cout, 1))

        self.stage1 = stage(64, 64)
        self.stage2 = stage(64, 128)
        self.stage3 = stage(128, 256)
        self.stage4 = stage(256, 384)
        self.dropout = nn.Dropout(drop)
        self.classifier = nn.Linear(384, num_classes)

    def forward(self, x):
        a = self.color_stem(x)
        b = self.noise_stem(self.noise_filter(x))
        x = self.merge(torch.cat([a, b], dim=1))
        x = self.stage4(self.stage3(self.stage2(self.stage1(x))))
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.classifier(self.dropout(x))


def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ------------------------------ data ------------------------------
IMG_EXTS = {".jpg", ".jpeg", ".png"}


def list_images(folder):
    files = [p for p in Path(folder).iterdir() if p.suffix.lower() in IMG_EXTS]

    def key(p):
        head = p.stem.split("_")[0]
        return (int(head) if head.isdigit() else float("inf"), p.name)

    return sorted(files, key=key)


def parse_label(path):
    return int(path.stem.rsplit("_", 1)[1])  # "10002_0.jpg" -> 0


def load_rgb(path):
    with Image.open(path) as im:
        return im.convert("RGB")


def jpeg(img, quality):
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    with Image.open(buf) as im:
        return im.convert("RGB")


def to_tensor(img):
    arr = np.asarray(img, dtype=np.uint8).copy()
    return torch.from_numpy(arr).permute(2, 0, 1).float().div_(255.0)


class CropDataset(Dataset):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, label = self.items[i]
        img = load_rgb(path)
        if random.random() < JPEG_P:                    # re-compress the FULL image, like the test set
            img = jpeg(img, random.randint(*JPEG_Q))
        w, h = img.size
        c = min(CROP, w, h) // 8 * 8
        left = random.randrange(0, (w - c) // 8 + 1) * 8   # aligned to the 8x8 JPEG grid
        top = random.randrange(0, (h - c) // 8 + 1) * 8
        img = img.crop((left, top, left + c, top + c))
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        return to_tensor(img), label


class FullDataset(Dataset):
    def __init__(self, paths, labels=None, jpeg_q=None):
        self.paths, self.labels, self.jpeg_q = paths, labels, jpeg_q

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = load_rgb(self.paths[i])
        if self.jpeg_q is not None:
            img = jpeg(img, self.jpeg_q)
        y = self.labels[i] if self.labels is not None else -1
        return to_tensor(img), y


# ------------------------------ train / predict ------------------------------
@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    preds, labels = [], []
    for x, y in loader:
        preds.append(model(x.to(device)).argmax(1).cpu())
        labels.append(y)
    return torch.cat(preds).numpy(), torch.cat(labels).numpy()


def train(device, best_path):
    items = [(p, parse_label(p)) for p in list_images(TRAIN_DIR)]
    random.shuffle(items)
    n_val = int(len(items) * VAL_FRACTION)
    val_items, train_items = items[:n_val], items[n_val:]
    print(f"train {len(train_items)}  val {len(val_items)}  "
          f"per class {np.bincount([l for _, l in items], minlength=3).tolist()}")

    train_loader = DataLoader(CropDataset(train_items), batch_size=BATCH, shuffle=True,
                              drop_last=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(FullDataset([p for p, _ in val_items], [l for _, l in val_items],
                                        jpeg_q=VAL_JPEG_Q),
                            batch_size=16, shuffle=False, num_workers=NUM_WORKERS)

    model = DualStreamForensicNet().to(device)
    n = count_trainable_parameters(model)
    print(f"Trainable parameters: {n:,}")
    assert n <= 12_300_000

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS * len(train_loader))
    loss_fn = nn.CrossEntropyLoss()
    best, t0 = -1.0, time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total, correct, seen = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item() * y.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            seen += y.size(0)

        preds, labels = predict(model, val_loader, device)
        acc = float((preds == labels).mean())
        print(f"epoch {epoch}/{EPOCHS}  loss {total / seen:.4f}  train acc {correct / seen:.4f}  "
              f"val acc (q{VAL_JPEG_Q}) {acc:.4f}  val preds {np.bincount(preds, minlength=3).tolist()}  "
              f"[{(time.time() - t0) / 60:.1f} min]", flush=True)
        if acc >= best:
            best = acc
            torch.save(model.state_dict(), best_path)
        if (time.time() - t0) / 60 > MAX_MINUTES:
            print("time limit reached, stopping training")
            break
    print(f"best val acc {best:.4f}")


def predict_folder(model, folder, name, device):
    if not os.path.isdir(folder):
        print(f"skip {name}: {folder} not found")
        return
    paths = list_images(folder)
    loader = DataLoader(FullDataset(paths), batch_size=1, shuffle=False, num_workers=NUM_WORKERS)
    preds, _ = predict(model, loader, device)
    data = {p.name: int(c) for p, c in zip(paths, preds)}
    text = json.dumps(data, indent=2)
    with open(os.path.join(OUT_DIR, f"{name}_predictions.json"), "w") as f:
        f.write(text)
    with zipfile.ZipFile(os.path.join(OUT_DIR, f"{name}_submission.zip"), "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("predictions.json", text)
    print(f"{name}: {len(data)} predictions, per class {np.bincount(preds, minlength=3).tolist()}")


def main():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    best_path = os.path.join(OUT_DIR, "best.pt")
    if not (os.environ.get("PREDICT_ONLY") == "1" and os.path.isfile(best_path)):
        train(device, best_path)
    model = DualStreamForensicNet().to(device)
    model.load_state_dict(torch.load(best_path, map_location=device))
    predict_folder(model, DEV_DIR, "dev", device)
    predict_folder(model, TEST_DIR, "test", device)


if __name__ == "__main__":
    main()
