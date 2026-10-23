#!/usr/bin/env python3
"""
CE7454 - AI-generated image detection + source attribution (3 classes)
  0 = real photo, 1 = Model A, 2 = Model B

Tri-view forensic network trained FROM SCRATCH (~8.5M trainable params, limit 12.3M):
  * RGB branch      : raw colour/tone statistics
  * NPR branch      : x - up(down(x)) residuals -> upsampling traces          (Tan et al., CVPR 2024)
  * Noise branch    : fixed SRM + learnable Bayar high-pass residuals          (Fridrich; Bayar & Stamm)
  * Spectral branch : tiled 64x64 log power spectrum of a Laplacian residual   (Frank et al. 2020; Corvi et al. 2023)
  * Heads           : 3-class head + auxiliary real-vs-fake head

Training: 50 epochs on 8-px-aligned random 256 crops (no resizing, flips, light JPEG re-compression),
then 5 epochs fine-tuning on full 512 images. Inference: one forward pass on the full image
(no test-time augmentation, no ensembling), using the EMA weights of the single trained model.

Run:  python aigi_forensic.py
      (if only the prediction step failed: PREDICT_ONLY=1 python aigi_forensic.py)
Outputs (OUT_DIR):
  dev_predictions.json / test_predictions.json   predictions to submit
  dev_submission.zip / test_submission.zip       same, zipped with predictions.json at the top level
  results.json (val accuracy + confusion matrix), history.json (per-epoch curves), best.pt
"""

import copy
import io
import json
import os
import random
import time
import zipfile
from collections import Counter
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
OUT_DIR = "CE7454_project/outputs"

SEED = 42
VAL_FRACTION = 0.10       # 10% of train held out to report accuracy / confusion matrix

# phase 1: crops
CROP = 256
EPOCHS = 50
BATCH = 32
LR = 2e-3

# phase 2: full-resolution fine-tuning
FT_EPOCHS = 5
FT_CROP = 512
FT_BATCH = 8
FT_LR = 2e-4

EVAL_BATCH = 16
WEIGHT_DECAY = 0.05
LABEL_SMOOTHING = 0.05
AUX_WEIGHT = 0.3
JPEG_AUG_P = 0.2
JPEG_Q = (80, 98)
EMA_DECAY = 0.999
GRAD_CLIP = 5.0
NUM_WORKERS = min(8, os.cpu_count() or 4)
PARAM_LIMIT = 12_300_000
NUM_CLASSES = 3
SPEC_TILE = 64

# Key format inside predictions.json. Check the starting kit's sample file:
#   "filename" -> {"12.jpg": 1, ...}     "stem" -> {"12": 1, ...}
PRED_KEY = "filename"   # required: keys like "0.jpg"
RUN_SHORTCUT_CHECK = True   # prints JPEG-table / size stats per class (analysis only)
# ====================================================================


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


# ------------------------------ data ------------------------------
FLIP_LR = Image.Transpose.FLIP_LEFT_RIGHT if hasattr(Image, "Transpose") else Image.FLIP_LEFT_RIGHT
IMG_EXTS = {".jpg", ".jpeg", ".png"}


def list_images(folder):
    files = [p for p in Path(folder).iterdir() if p.suffix.lower() in IMG_EXTS]

    def sort_key(p):
        head = p.stem.split("_")[0]
        return (int(head) if head.isdigit() else float("inf"), p.name)

    return sorted(files, key=sort_key)


def parse_label(path):
    return int(path.stem.rsplit("_", 1)[1])  # "10002_0.jpg" -> 0


def load_rgb(path):
    with Image.open(path) as im:
        return im.convert("RGB")


def to_tensor(img):
    arr = np.asarray(img, dtype=np.uint8).copy()
    return torch.from_numpy(arr).permute(2, 0, 1).float().div_(255.0)


def jpeg_recompress(img, quality):
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    with Image.open(buf) as im:
        return im.convert("RGB")


class TrainDataset(Dataset):
    def __init__(self, items, crop):
        self.items = items
        self.crop = crop

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, label = self.items[i]
        img = load_rgb(path)
        w, h = img.size
        c = min(self.crop, w, h) // 8 * 8
        x0 = random.randrange(0, (w - c) // 8 + 1) * 8   # aligned to the 8x8 JPEG grid
        y0 = random.randrange(0, (h - c) // 8 + 1) * 8
        img = img.crop((x0, y0, x0 + c, y0 + c))
        if random.random() < JPEG_AUG_P:
            img = jpeg_recompress(img, random.randint(*JPEG_Q))
        if random.random() < 0.5:
            img = img.transpose(FLIP_LR)
        return to_tensor(img), label


class FullImageDataset(Dataset):
    def __init__(self, paths, labels=None):
        self.paths = paths
        self.labels = labels

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        x = to_tensor(load_rgb(self.paths[i]))
        y = self.labels[i] if self.labels is not None else -1
        return x, y


def stratified_split(items, frac, seed):
    rng = random.Random(seed)
    by_class = {}
    for it in items:
        by_class.setdefault(it[1], []).append(it)
    train, val = [], []
    for cls in sorted(by_class):
        group = by_class[cls][:]
        rng.shuffle(group)
        n_val = int(round(len(group) * frac))
        val += group[:n_val]
        train += group[n_val:]
    rng.shuffle(train)
    return train, val


def shortcut_check(items, per_class=300):
    """Image sizes and JPEG quantisation tables per class. Disjoint tables between classes mean
    compression settings alone could separate them, which is a shortcut rather than forensics."""
    print("\n=== shortcut check (JPEG tables / sizes per class) ===")
    table_sets = {}
    for cls in range(NUM_CLASSES):
        paths = [p for p, l in items if l == cls][:per_class]
        tables, sizes = Counter(), Counter()
        for p in paths:
            with Image.open(p) as im:
                sizes[im.size] += 1
                q = getattr(im, "quantization", None)
                key = hash(tuple(tuple(v) for _, v in sorted(q.items()))) if q else None
                tables[key] += 1
        table_sets[cls] = set(tables)
        top_share = tables.most_common(1)[0][1] / max(len(paths), 1)
        print(f"class {cls}: {len(tables)} distinct quant tables (top covers {top_share:.0%}), sizes {dict(sizes)}")
    for a in range(NUM_CLASSES):
        for b in range(a + 1, NUM_CLASSES):
            print(f"classes {a} & {b}: {len(table_sets[a] & table_sets[b])} shared quant tables")
    print()


# ------------------------------ model ------------------------------
class BasicBlock(nn.Module):
    def __init__(self, inp, out, stride=1):
        super().__init__()
        self.c1 = nn.Conv2d(inp, out, 3, stride, 1, bias=False)
        self.b1 = nn.BatchNorm2d(out)
        self.c2 = nn.Conv2d(out, out, 3, 1, 1, bias=False)
        self.b2 = nn.BatchNorm2d(out)
        if stride == 1 and inp == out:
            self.sc = nn.Identity()
        else:
            self.sc = nn.Sequential(nn.Conv2d(inp, out, 1, stride, bias=False), nn.BatchNorm2d(out))

    def forward(self, x):
        y = F.relu(self.b1(self.c1(x)), inplace=True)
        y = self.b2(self.c2(y))
        return F.relu(y + self.sc(x), inplace=True)


class NPR(nn.Module):
    """Neighbouring-pixel residual: what remains after nearest down/up-sampling (Tan et al., CVPR 2024)."""

    def forward(self, x):
        down = F.interpolate(x, scale_factor=0.5, mode="nearest")
        return x - F.interpolate(down, scale_factor=2.0, mode="nearest")


class SRM(nn.Module):
    """Three classic fixed SRM high-pass kernels per RGB channel -> 9 truncated residual maps."""

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
        k3 = torch.tensor([[0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0],
                           [0, 1, -2, 1, 0],
                           [0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0]], dtype=torch.float32) / 2.0
        self.register_buffer("weight", torch.stack([k1, k2, k3] * 3).unsqueeze(1))  # (9,1,5,5)

    def forward(self, x):
        y = F.conv2d(x * 255.0, self.weight.to(x.dtype), padding=2, groups=3)
        return torch.clamp(y, -3.0, 3.0) / 3.0


class BayarConv(nn.Module):
    """Constrained conv (Bayar & Stamm): centre tap -1, other taps sum to 1 -> learnable high-pass."""

    def __init__(self, inp, out, k=5):
        super().__init__()
        self.k = k
        self.weight = nn.Parameter(torch.randn(out, inp, k * k - 1) * 0.02)

    def forward(self, x):
        n = self.k * self.k - 1
        w = self.weight - self.weight.mean(-1, keepdim=True) + 1.0 / n
        c = n // 2
        w = torch.cat([w[..., :c], -torch.ones_like(w[..., :1]), w[..., c:]], dim=-1)
        w = w.view(w.size(0), w.size(1), self.k, self.k)
        return F.conv2d(x, w.to(x.dtype), padding=self.k // 2)


class SpectralBranch(nn.Module):
    """Laplacian residual -> FFT power spectra of non-overlapping 64x64 tiles, averaged, log, centred.
    The map size depends only on the tile size, so 256 crops and 512 images look the same here."""

    def __init__(self, tile=SPEC_TILE, out_dim=128):
        super().__init__()
        self.tile = tile
        lap = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32)
        self.register_buffer("lap", lap.view(1, 1, 3, 3).repeat(3, 1, 1, 1))
        self.net = nn.Sequential(
            nn.BatchNorm2d(3),
            nn.Conv2d(3, 32, 3, 1, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            BasicBlock(32, 64, 2),
            BasicBlock(64, 128, 2),
            BasicBlock(128, out_dim, 2),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, x):
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = x.float() * 255.0
            r = F.conv2d(x, self.lap, padding=1, groups=3)
            t = self.tile
            h, w = r.shape[-2] // t * t, r.shape[-1] // t * t
            r = r[..., :h, :w].contiguous()
            tiles = r.unfold(2, t, t).unfold(3, t, t)                    # B,3,nh,nw,t,t
            power = torch.fft.fft2(tiles).abs().pow(2).mean(dim=(2, 3))  # B,3,t,t
            spec = torch.fft.fftshift(torch.log1p(power), dim=(-2, -1))
        return self.net(spec).flatten(1)


def make_branch(inp):
    return nn.Sequential(
        nn.BatchNorm2d(inp),
        nn.Conv2d(inp, 32, 3, 1, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
        BasicBlock(32, 64, 1),   # full resolution, no early pooling
        BasicBlock(64, 64, 2),
    )


class ForensicNet(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.srm = SRM()
        self.bayar = BayarConv(3, 6)
        self.noise_branch = make_branch(9 + 6)
        self.npr = NPR()
        self.npr_branch = make_branch(3)
        self.rgb_branch = make_branch(3)
        self.fuse = nn.Sequential(nn.Conv2d(192, 128, 1, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.trunk = nn.Sequential(
            BasicBlock(128, 128, 2), BasicBlock(128, 128),
            BasicBlock(128, 256, 2), BasicBlock(256, 256),
            BasicBlock(256, 384, 2), BasicBlock(384, 384),
        )
        self.spectral = SpectralBranch()
        feat = 384 * 2 + 128
        self.drop = nn.Dropout(0.3)
        self.fc = nn.Linear(feat, num_classes)
        self.aux = nn.Linear(feat, 1)  # real (0) vs fake (1)

    def forward(self, x):
        a = self.noise_branch(torch.cat([self.srm(x), self.bayar(x)], dim=1))
        b = self.npr_branch(self.npr(x))
        c = self.rgb_branch(x)
        f = self.trunk(self.fuse(torch.cat([a, b, c], dim=1)))
        f = torch.cat([F.adaptive_avg_pool2d(f, 1), F.adaptive_max_pool2d(f, 1)], dim=1).flatten(1)
        f = torch.cat([f, self.spectral(x).to(f.dtype)], dim=1)
        f = self.drop(f)
        return self.fc(f), self.aux(f).squeeze(1)


def count_trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class EMA:
    def __init__(self, model, decay):
        self.decay = decay
        self.model = copy.deepcopy(model).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        src = model.state_dict()
        for k, v in self.model.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(self.decay).add_(src[k].detach(), alpha=1.0 - self.decay)
            else:
                v.copy_(src[k])


# ------------------------------ train / eval ------------------------------
@torch.no_grad()
def predict_probs(model, loader, device):
    model.eval()
    probs, labels = [], []
    amp = device.type == "cuda"
    for x, y in loader:
        x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
            p = F.softmax(model(x)[0].float(), dim=1)  # single forward pass, no test-time augmentation
        probs.append(p.cpu())
        labels.append(y)
    return torch.cat(probs).numpy(), torch.cat(labels).numpy()


def report(probs, labels):
    pred = probs.argmax(1)
    acc = float((pred == labels).mean())
    veracity = float(((pred > 0) == (labels > 0)).mean())
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
    for t, p in zip(labels, pred):
        cm[t, p] += 1
    return acc, veracity, cm


def param_groups(model, wd):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim <= 1 or "bayar" in name else decay).append(p)
    return [{"params": decay, "weight_decay": wd}, {"params": no_decay, "weight_decay": 0.0}]


def make_train_loader(items, crop, batch):
    return DataLoader(
        TrainDataset(items, crop), batch_size=batch, shuffle=True, drop_last=True,
        num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=NUM_WORKERS > 0,
    )


class Trainer:
    def __init__(self, model, ema, device, val_loader, best_path):
        self.model, self.ema, self.device = model, ema, device
        self.val_loader, self.best_path = val_loader, best_path
        self.best = {"val_acc": -1.0}
        self.history = []
        self.t_start = time.time()
        self.epoch_times = {"crop256": [], "full512": []}
        self.amp = device.type == "cuda"
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp)
        self.ce = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

    def eta(self, tag, epoch, epochs):
        crop = self.epoch_times["crop256"]
        full = self.epoch_times["full512"]
        crop_t = float(np.mean(crop)) if crop else 0.0
        full_t = float(np.mean(full)) if full else 4.0 * crop_t  # rough guess until phase 2 starts
        if tag == "crop256":
            left = (epochs - epoch) * crop_t + FT_EPOCHS * full_t
        else:
            left = (epochs - epoch) * full_t
        elapsed = time.time() - self.t_start

        def fmt(sec):
            return f"{int(sec // 3600)}h{int(sec % 3600 // 60):02d}m"

        finish = time.strftime("%a %H:%M", time.localtime(time.time() + left))
        return f"elapsed {fmt(elapsed)}  remaining ~{fmt(left)}  (finish ~{finish})"

    def run(self, loader, opt, sched, epochs, tag):
        for epoch in range(1, epochs + 1):
            self.model.train()
            t0, tot_loss, correct, seen = time.time(), 0.0, 0, 0
            for x, y in loader:
                x = x.to(self.device, non_blocking=True).to(memory_format=torch.channels_last)
                y = y.to(self.device, non_blocking=True)
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.amp):
                    logits, aux = self.model(x)
                    loss = self.ce(logits, y)
                    if aux is not None:
                        loss = loss + AUX_WEIGHT * F.binary_cross_entropy_with_logits(aux.float(), (y > 0).float())
                opt.zero_grad(set_to_none=True)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(self.model.parameters(), GRAD_CLIP)
                self.scaler.step(opt)
                self.scaler.update()
                sched.step()
                self.ema.update(self.model)
                tot_loss += loss.item() * y.size(0)
                correct += (logits.argmax(1) == y).sum().item()
                seen += y.size(0)
                if epoch == 1 and seen // y.size(0) % 100 == 0:
                    print(f"  [{tag}] epoch 1: {seen}/{len(loader.dataset)} images, "
                          f"{time.time() - t0:.0f}s so far", flush=True)

            rec = {"phase": tag, "epoch": epoch, "train_loss": tot_loss / seen,
                   "train_acc": correct / seen, "lr": sched.get_last_lr()[0]}
            msg = (f"[{tag}] epoch {epoch:02d}/{epochs}  loss {tot_loss / seen:.4f}  "
                   f"train acc {correct / seen:.4f}  lr {sched.get_last_lr()[0]:.2e}")
            if self.val_loader is not None:
                probs, labels = predict_probs(self.ema.model, self.val_loader, self.device)
                acc, ver, cm = report(probs, labels)
                rec.update(val_acc=acc, val_real_fake_acc=ver)
                msg += f"  | val acc {acc:.4f}  real/fake {ver:.4f}"
                if acc >= self.best["val_acc"]:
                    self.best = {"val_acc": acc, "real_fake_acc": ver, "confusion_matrix": cm.tolist(),
                                 "phase": tag, "epoch": epoch}
                    torch.save(self.ema.model.state_dict(), self.best_path)
                    msg += "  *saved*"
                    print("confusion matrix (rows true 0/1/2, cols predicted):\n", cm)
            else:
                torch.save(self.ema.model.state_dict(), self.best_path)
            self.history.append(rec)
            with open(os.path.join(OUT_DIR, "history.json"), "w") as fh:
                json.dump(self.history, fh, indent=2)
            dt = time.time() - t0
            self.epoch_times[tag].append(dt)
            print(msg + f"  ({dt:.0f}s)  |  {self.eta(tag, epoch, epochs)}", flush=True)


def train(device):
    items = [(p, parse_label(p)) for p in list_images(TRAIN_DIR)]
    counts = np.bincount([l for _, l in items], minlength=NUM_CLASSES)
    print(f"train images: {len(items)}  per class: {counts.tolist()}")
    print("label parse check:", [(p.name, l) for p, l in items[:3]])
    assert set(l for _, l in items) <= set(range(NUM_CLASSES)), "unexpected labels in filenames"
    if RUN_SHORTCUT_CHECK:
        shortcut_check(items)

    if VAL_FRACTION > 0:
        train_items, val_items = stratified_split(items, VAL_FRACTION, SEED)
    else:
        train_items, val_items = items, []
    print(f"train split: {len(train_items)}  val split: {len(val_items)}")
    val_loader = None
    if val_items:
        val_loader = DataLoader(
            FullImageDataset([p for p, _ in val_items], [l for _, l in val_items]),
            batch_size=EVAL_BATCH, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True,
        )

    model = ForensicNet().to(device).to(memory_format=torch.channels_last)
    n_params = count_trainable(model)
    print(f"trainable parameters: {n_params:,} (limit {PARAM_LIMIT:,})")
    assert n_params <= PARAM_LIMIT, "model exceeds the parameter limit"

    ema = EMA(model, EMA_DECAY)
    best_path = os.path.join(OUT_DIR, "best.pt")
    trainer = Trainer(model, ema, device, val_loader, best_path)

    loader = make_train_loader(train_items, CROP, BATCH)
    opt = torch.optim.AdamW(param_groups(model, WEIGHT_DECAY), lr=LR)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=LR, total_steps=EPOCHS * len(loader), pct_start=0.1,
        div_factor=20, final_div_factor=200)
    trainer.run(loader, opt, sched, EPOCHS, "crop256")

    if FT_EPOCHS > 0:
        loader = make_train_loader(train_items, FT_CROP, FT_BATCH)
        opt = torch.optim.AdamW(param_groups(model, WEIGHT_DECAY), lr=FT_LR)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=FT_EPOCHS * len(loader), eta_min=FT_LR / 20)
        trainer.run(loader, opt, sched, FT_EPOCHS, "full512")

    results = {"seed": SEED, "trainable_params": n_params,
               "val_fraction": VAL_FRACTION, "epochs": EPOCHS, "ft_epochs": FT_EPOCHS,
               "jpeg_aug_p": JPEG_AUG_P, **trainer.best}
    with open(os.path.join(OUT_DIR, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    if val_loader is not None:
        print(f"best val acc: {trainer.best['val_acc']:.4f}")
        print("best confusion matrix:", trainer.best["confusion_matrix"])
    return best_path


def write_submission(paths, preds, name):
    key = (lambda p: p.name) if PRED_KEY == "filename" else (lambda p: p.stem)
    data = {key(p): int(c) for p, c in zip(paths, preds)}
    text = json.dumps(data, indent=2)
    json_path = os.path.join(OUT_DIR, f"{name}_predictions.json")
    with open(json_path, "w") as f:
        f.write(text)
    zip_path = os.path.join(OUT_DIR, f"{name}_submission.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("predictions.json", text)
    with open(json_path, encoding="utf-8") as f:
        check = json.load(f)
    assert len(check) == len(paths) == len(set(p.name for p in paths)), "missing or duplicate keys"
    assert all(k.endswith(".jpg") and "/" not in k for k in check), "keys must be bare .jpg filenames"
    assert all(type(v) is int and v in (0, 1, 2) for v in check.values()), "labels must be ints 0/1/2"
    print(f"wrote {json_path}  (format check passed)")
    print(f"wrote {zip_path}  ({len(data)} predictions, per class {np.bincount(preds, minlength=NUM_CLASSES).tolist()})")


def predict_folder(model, folder, name, device):
    if not os.path.isdir(folder):
        print(f"skipping {name}: folder not found -> {folder}")
        return
    paths = list_images(folder)
    sizes = set()
    for p in paths:
        with Image.open(p) as im:
            sizes.add(im.size)
    bs = EVAL_BATCH if len(sizes) == 1 else 1
    print(f"{name}: {len(paths)} images, sizes {sorted(sizes)[:5]}")
    loader = DataLoader(FullImageDataset(paths), batch_size=bs, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=True)
    probs, _ = predict_probs(model, loader, device)
    np.save(os.path.join(OUT_DIR, f"{name}_probs.npy"), probs)
    write_submission(paths, probs.argmax(1), name)


def main():
    seed_everything(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"seed: {SEED} | device: {device}",
          torch.cuda.get_device_name(0) if device.type == "cuda" else "")

    best_path = os.path.join(OUT_DIR, "best.pt")
    if os.environ.get("PREDICT_ONLY") == "1" and os.path.isfile(best_path):
        print("PREDICT_ONLY=1: skipping training, loading", best_path)
    else:
        best_path = train(device)

    model = ForensicNet().to(device).to(memory_format=torch.channels_last)
    model.load_state_dict(torch.load(best_path, map_location=device))
    predict_folder(model, DEV_DIR, "dev", device)
    predict_folder(model, TEST_DIR, "test", device)


if __name__ == "__main__":
    main()