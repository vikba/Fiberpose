#!/usr/bin/env python3
"""
train.py

Train ResUnet with precomputed dt targets.

Channel:
  pred[:,0] -> dt

Losses:
  - weighted Huber loss on dt
  - Eikonal loss on |grad(dt)| inside objects

Requirements:
  pip install torch torchvision tifffile numpy opencv-python

Expected training layout:
  dataset/
    images/
    targets/

Optional validation layout:
  dataset/
    val_images/
    val_targets/

Optional CSV with per-image background weights:
  stem,bg_weight
  0001,1
  0002,0
  0003,1
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

import tifffile as tiff

# -------------------------
# Model import
# -------------------------
try:
    from sources.resunet import ResUnet
except ModuleNotFoundError:
    import sys
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from sources.resunet import ResUnet


# -------------------------
# IO helpers
# -------------------------
def read_image_any(path: Path) -> np.ndarray:
    """Return grayscale float32 image in [0,1]."""
    if path.suffix.lower() in [".tif", ".tiff"]:
        im = tiff.imread(str(path))
    else:
        import cv2
        im = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if im is None:
            raise FileNotFoundError(f"Could not read image: {path}")

    im = np.asarray(im)

    if im.ndim == 3:
        if im.shape[-1] in (3, 4):
            im = im[..., :3].mean(axis=-1)
        else:
            im = im[..., 0]

    im = im.astype(np.float32)
    vmin, vmax = float(im.min()), float(im.max())
    if vmax > vmin:
        im = (im - vmin) / (vmax - vmin)
    else:
        im = np.zeros_like(im, dtype=np.float32)
    return im


def _template_to_regex(template: str) -> re.Pattern:
    escaped = re.escape(template)
    pattern = escaped.replace(r"\{stem\}", r"(?P<stem>.+)")
    return re.compile(f"^{pattern}$")


def _load_u16_png(path: Path) -> np.ndarray:
    import cv2
    arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise FileNotFoundError(f"Could not read target PNG: {path}")
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr.astype(np.uint16, copy=False)


def _denormalize_u16(arr_u16: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    if vmax <= vmin:
        return np.zeros_like(arr_u16, dtype=np.float32)
    scaled = arr_u16.astype(np.float32) / 65535.0
    return scaled * (vmax - vmin) + vmin


def _load_float_tif(path: Path) -> np.ndarray:
    arr = tiff.imread(str(path))
    arr = np.asarray(arr)
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr.astype(np.float32, copy=False)


def _to_01(t: torch.Tensor) -> torch.Tensor:
    t = t.detach().float()
    vmin = float(t.min())
    vmax = float(t.max())
    if vmax <= vmin:
        return torch.zeros_like(t)
    return (t - vmin) / (vmax - vmin)


def _np_to_u8(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    vmin = float(x.min())
    vmax = float(x.max())
    if vmax <= vmin:
        return np.zeros_like(x, dtype=np.uint8)
    y = (x - vmin) / (vmax - vmin)
    return np.clip(y * 255.0, 0, 255).astype(np.uint8)


def load_bg_weights_csv(csv_path: str | None) -> dict[str, float]:
    """
    CSV format:
      stem,bg_weight
      0001,1
      0002,0
    """
    weights: dict[str, float] = {}
    if csv_path is None:
        return weights

    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"stem", "bg_weight"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(
                f"Background weights CSV must contain columns {sorted(required)}; "
                f"got {reader.fieldnames}"
            )

        for row in reader:
            stem = row["stem"].strip()
            bg_weight = float(row["bg_weight"])
            weights[stem] = bg_weight

    return weights


# -------------------------
# Augmentations
# -------------------------
def random_flip_rotate(
    x: np.ndarray,
    dt: np.ndarray,
    rng: np.random.Generator
) -> Tuple[np.ndarray, np.ndarray]:
    """
    x:  (H,W)
    dt: (H,W)
    """
    k = int(rng.integers(0, 4))
    if k:
        x = np.rot90(x, k, axes=(0, 1))
        dt = np.rot90(dt, k, axes=(0, 1))

    if bool(rng.integers(0, 2)):
        x = np.flip(x, axis=1)
        dt = np.flip(dt, axis=1)

    if bool(rng.integers(0, 2)):
        x = np.flip(x, axis=0)
        dt = np.flip(dt, axis=0)

    return x, dt


def random_crop(
    x: np.ndarray,
    dt: np.ndarray,
    crop: int,
    rng: np.random.Generator
) -> Tuple[np.ndarray, np.ndarray]:
    H, W = x.shape
    if crop <= 0 or (H <= crop and W <= crop):
        return x, dt

    ch = min(crop, H)
    cw = min(crop, W)
    y0 = int(rng.integers(0, H - ch + 1)) if H > ch else 0
    x0 = int(rng.integers(0, W - cw + 1)) if W > cw else 0

    x_c = x[y0:y0 + ch, x0:x0 + cw]
    dt_c = dt[y0:y0 + ch, x0:x0 + cw]
    return x_c, dt_c


# -------------------------
# Dataset
# -------------------------
class DtTargetsDataset(Dataset):
    def __init__(
        self,
        images_dir: Path,
        targets_dir: Path,
        target_template: str = "{stem}_{kind}.tif",
        image_template: str | None = None,
        crop: int = 512,
        augment: bool = True,
        seed: int = 0,
        bg_weights_csv: str | None = None,
        default_bg_weight: float = 0.0,
    ):
        self.images_dir = images_dir
        self.targets_dir = targets_dir
        self.target_template = target_template
        self.crop = crop
        self.augment = augment
        self.rng = np.random.default_rng(seed)

        self.image_template = image_template
        self._image_re = _template_to_regex(self.image_template) if self.image_template else None

        self.bg_weights_map = load_bg_weights_csv(bg_weights_csv)
        self.default_bg_weight = float(default_bg_weight)

        self.image_paths = sorted([p for p in images_dir.iterdir() if p.is_file()])
        if not self.image_paths:
            raise FileNotFoundError(f"No images found in: {images_dir}")

        self.target_stems: list[str] = []
        self.dt_paths: list[Path | None] = []
        matched_images: list[Path] = []

        for ip in self.image_paths:
            if self._image_re is not None:
                m = self._image_re.match(ip.name)
                if not m:
                    continue
                stem = m.group("stem")
            else:
                stem = ip.stem

            dt_tif = targets_dir / self.target_template.format(stem=stem, kind="dt")

            if dt_tif.exists():
                self.dt_paths.append(dt_tif)
            else:
                meta_path = targets_dir / f"{stem}_targets.json"
                dt_path = targets_dir / f"{stem}_dt.png"
                if not (meta_path.exists() and dt_path.exists()):
                    raise FileNotFoundError(
                        f"Missing dt target for {ip.name}: expected "
                        f"{dt_tif.name} or {stem}_dt.png and {stem}_targets.json"
                    )
                self.dt_paths.append(None)

            matched_images.append(ip)
            self.target_stems.append(stem)

        self.image_paths = matched_images
        if not self.image_paths:
            msg = (
                f"No images matched template {self.image_template} in {images_dir}"
                if self.image_template else
                f"No images found in: {images_dir}"
            )
            raise FileNotFoundError(msg)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        ip = self.image_paths[idx]
        stem = self.target_stems[idx]
        dt_path = self.dt_paths[idx]

        x = read_image_any(ip)

        if dt_path is not None:
            dt = _load_float_tif(dt_path)
        else:
            meta_path = self.targets_dir / f"{stem}_targets.json"
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)

            dt_u16 = _load_u16_png(self.targets_dir / f"{stem}_dt.png")
            dt = _denormalize_u16(dt_u16, meta["dt_min"], meta["dt_max"]).astype(np.float32)

        if self.augment:
            x, dt = random_crop(x, dt, self.crop, self.rng)
            x, dt = random_flip_rotate(x, dt, self.rng)
        elif self.crop > 0:
            x, dt = random_crop(x, dt, self.crop, self.rng)

        x_t = torch.from_numpy(np.ascontiguousarray(x[None, ...]))    # (1,H,W)
        y_t = torch.from_numpy(np.ascontiguousarray(dt[None, ...]))   # (1,H,W)

        bgw = np.float32(self.bg_weights_map.get(stem, self.default_bg_weight))
        bgw_t = torch.tensor(bgw, dtype=torch.float32)

        return x_t, y_t, bgw_t, stem


# -------------------------
# Loss
# -------------------------
class WeightedHuberLoss(nn.Module):
    def __init__(self, delta: float = 1.0):
        super().__init__()
        self.delta = delta

    def forward(self, pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        diff = torch.abs(pred - target)
        quadratic = torch.minimum(diff, torch.tensor(self.delta, dtype=diff.dtype, device=diff.device))
        linear = diff - quadratic
        loss = 0.5 * quadratic ** 2 + self.delta * linear
        return (loss * weight).mean()


# -------------------------
# Gradient helper
# -------------------------
def get_spatial_gradient(image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Input:  (B, 1, H, W)
    Output: dx, dy each with shape (B, 1, H, W)
    """
    kernel_x = torch.tensor(
        [[-1, 0, 1],
         [-2, 0, 2],
         [-1, 0, 1]],
        dtype=image.dtype,
        device=image.device
    ).view(1, 1, 3, 3)

    kernel_y = torch.tensor(
        [[-1, -2, -1],
         [ 0,  0,  0],
         [ 1,  2,  1]],
        dtype=image.dtype,
        device=image.device
    ).view(1, 1, 3, 3)

    dx = F.conv2d(image, kernel_x, padding=1)
    dy = F.conv2d(image, kernel_y, padding=1)

    return dx / 8.0, dy / 8.0


def compute_losses(
    pred: torch.Tensor,
    yb: torch.Tensor,
    bgw_batch: torch.Tensor,
    cfg,
    loss_fn: nn.Module,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    gt_dt = yb

    fg = (gt_dt > 0).float()
    bg = 1.0 - fg

    if cfg.ignore_bg:
        w = cfg.fg_weight * fg
    else:
        w = cfg.fg_weight * fg + (cfg.bg_weight * bgw_batch) * bg

    loss_dt = loss_fn(pred, gt_dt, w)

    obj_mask = (gt_dt > 0).float()
    dt_dx, dt_dy = get_spatial_gradient(pred)
    pred_dt_mag = torch.sqrt(dt_dx ** 2 + dt_dy ** 2 + 1e-8)
    loss_eik_dt = F.mse_loss(pred_dt_mag * obj_mask, obj_mask)

    loss = cfg.w_dt * loss_dt + cfg.w_eik_dt * loss_eik_dt
    return loss, loss_dt, loss_eik_dt


def save_validation_preview(
    out_dir: Path,
    epoch: int,
    stem: str,
    xb: torch.Tensor,
    yb: torch.Tensor,
    pred: torch.Tensor,
) -> None:
    import cv2

    out_dir.mkdir(parents=True, exist_ok=True)

    x = xb[0, 0].detach().cpu().numpy()
    gt_dt = yb[0, 0].detach().cpu().numpy()
    pr_dt = pred[0, 0].detach().cpu().numpy()

    row = np.concatenate([
        _np_to_u8(x),
        _np_to_u8(gt_dt),
        _np_to_u8(pr_dt),
    ], axis=1)

    save_path = out_dir / f"epoch_{epoch:04d}_{stem}.png"
    cv2.imwrite(str(save_path), _np_to_u8(pr_dt))


# -------------------------
# Train
# -------------------------
@dataclass
class TrainCfg:
    nbase: List[int] = None
    nout: int = 1
    lr: float = 1e-3
    batch: int = 4
    epochs: int = 50
    crop: int = 512
    num_workers: int = 2
    w_dt: float = 1.0
    w_eik_dt: float = 10.0
    fg_weight: float = 5.0
    bg_weight: float = 1.0
    ignore_bg: bool = False
    amp: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True, help="dataset root with images/ and targets/")
    ap.add_argument("--out", type=str, required=True, help="output .pt (best state_dict) path")
    ap.add_argument("--checkpoint", type=str, default=None, help="optional checkpoint to initialize from")
    ap.add_argument("--start-epoch", type=int, default=1, help="starting epoch index (1-based)")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=9)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--crop", type=int, default=512)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--w-dt", type=float, default=1.0)
    ap.add_argument("--w-eik-dt", type=float, default=5.0)
    ap.add_argument("--huber-delta", type=float, default=1.0)

    ap.add_argument("--fg-weight", type=float, default=3.0, help="loss weight for foreground pixels")
    ap.add_argument("--bg-weight", type=float, default=1.0, help="global multiplier for background pixels")
    ap.add_argument("--ignore-bg", action="store_true", help="ignore background in supervised loss")

    ap.add_argument("--bg-weights-csv", type=str, default=None, help="CSV with columns: stem,bg_weight")
    ap.add_argument("--default-bg-weight", type=float, default=0.0, help="background weight used when a stem is missing from CSV")

    ap.add_argument("--image-template", type=str, default=None, help="image filename template with {stem}")
    ap.add_argument(
        "--target-template",
        type=str,
        default="{stem}_masks_{kind}.tif",
        help="target filename template with {stem} and {kind} (kind=dt)",
    )
    ap.add_argument("--targets-dir", type=str, default=None, help="folder with training targets (default: data/targets)")
    ap.add_argument("--logdir", type=str, default=None, help="tensorboard log dir (default: out_dir/runs)")
    ap.add_argument("--no-tensorboard", action="store_true", help="disable tensorboard logging")
    ap.add_argument("--no-amp", action="store_true")

    # validation
    ap.add_argument("--val-images-dir", type=str, default=None, help="validation images dir (default: data/val_images)")
    ap.add_argument("--val-targets-dir", type=str, default=None, help="validation targets dir (default: data/val_targets)")
    ap.add_argument("--val-batch", type=int, default=1)
    ap.add_argument("--val-preview-every", type=int, default=5, help="run preview inference on validation images every N epochs; 0 disables")
    ap.add_argument("--val-preview-count", type=int, default=4, help="how many validation samples to save per preview step")

    # periodic weight saving
    ap.add_argument("--save-every", type=int, default=20, help="save model weights every M epochs; 0 disables")
    ap.add_argument("--save-dir", type=str, default=None, help="directory for periodic weight snapshots (default: out_dir/checkpoints)")

    args = ap.parse_args()

    cfg = TrainCfg(
        nbase=[1, 32, 64, 128, 256],
        lr=args.lr,
        batch=args.batch,
        epochs=args.epochs,
        crop=args.crop,
        num_workers=args.workers,
        w_dt=args.w_dt,
        w_eik_dt=args.w_eik_dt,
        fg_weight=args.fg_weight,
        bg_weight=args.bg_weight,
        ignore_bg=args.ignore_bg,
        amp=not args.no_amp,
    )

    root = Path(args.data)
    images_dir = root / "images"
    targets_dir = Path(args.targets_dir) if args.targets_dir else (root / "targets")

    ds = DtTargetsDataset(
        images_dir=images_dir,
        targets_dir=targets_dir,
        target_template=args.target_template,
        image_template=args.image_template,
        crop=cfg.crop,
        augment=True,
        seed=args.seed,
        bg_weights_csv=args.bg_weights_csv,
        default_bg_weight=args.default_bg_weight,
    )

    dl = DataLoader(
        ds,
        batch_size=cfg.batch,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    val_dl = None
    val_images_dir = Path(args.val_images_dir) if args.val_images_dir else (root / "val_images")
    val_targets_dir = Path(args.val_targets_dir) if args.val_targets_dir else (root / "val_targets")

    if val_images_dir.exists() and val_targets_dir.exists():
        val_ds = DtTargetsDataset(
            images_dir=val_images_dir,
            targets_dir=val_targets_dir,
            target_template=args.target_template,
            image_template=args.image_template,
            crop=0,
            augment=False,
            seed=args.seed,
            bg_weights_csv=args.bg_weights_csv,
            default_bg_weight=args.default_bg_weight,
        )
        val_dl = DataLoader(
            val_ds,
            batch_size=args.val_batch,
            shuffle=False,
            num_workers=max(0, cfg.num_workers // 2),
            pin_memory=True,
            drop_last=False,
        )
        print(f"validation enabled: {len(val_ds)} images")
    else:
        print("validation disabled: val_images/ or val_targets/ not found")

    device = torch.device(cfg.device)

    model = ResUnet(
        nbase=cfg.nbase,
        nout=cfg.nout,
        sz=3,
        dilation=1,
        kernel_size=2,
        scale_factor=2,
        residual_on=True,
        style_on=True,
        concatenation=False,
        checkpoint=False,
        dropout=False,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=(cfg.amp and device.type == "cuda"))
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=(cfg.amp and device.type == "cuda"))

    start_epoch = max(1, int(args.start_epoch))
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device)
        if isinstance(ckpt, dict) and "model" in ckpt:
            model.load_state_dict(ckpt["model"])
            if "optimizer" in ckpt:
                opt.load_state_dict(ckpt["optimizer"])
            if "scaler" in ckpt:
                scaler.load_state_dict(ckpt["scaler"])
            if "epoch" in ckpt and args.start_epoch == 1:
                start_epoch = max(1, int(ckpt["epoch"]) + 1)
        else:
            model.load_state_dict(ckpt)

    loss_fn = WeightedHuberLoss(delta=args.huber_delta)

    best_train = float("inf")
    best_val = float("inf")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    save_dir = Path(args.save_dir) if args.save_dir else (out_path.parent / "checkpoints")
    preview_dir = out_path.parent / "val_previews"
    if args.save_every > 0:
        save_dir.mkdir(parents=True, exist_ok=True)

    logdir = Path(args.logdir) if args.logdir else (out_path.parent / "runs")
    writer = None
    if not args.no_tensorboard:
        writer = SummaryWriter(log_dir=str(logdir))

    def _fmt_seconds(sec: float) -> str:
        sec = max(0.0, float(sec))
        m, s = divmod(int(sec + 0.5), 60)
        h, m = divmod(m, 60)
        if h:
            return f"{h:d}h{m:02d}m{s:02d}s"
        if m:
            return f"{m:d}m{s:02d}s"
        return f"{s:d}s"

    global_step = 0
    t_start_all = time.time()

    for epoch in range(start_epoch, cfg.epochs + 1):
        model.train()
        t_start_epoch = time.time()

        running = 0.0
        running_dt = 0.0
        running_eik_dt = 0.0

        for batch_idx, (xb, yb, bgw_batch, stems) in enumerate(dl):
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            bgw_batch = bgw_batch.to(device, non_blocking=True).view(-1, 1, 1, 1)

            opt.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=device.type, enabled=scaler.is_enabled()):
                pred, _ = model(xb)
                loss, loss_dt, loss_eik_dt = compute_losses(
                    pred=pred,
                    yb=yb,
                    bgw_batch=bgw_batch,
                    cfg=cfg,
                    loss_fn=loss_fn,
                )

            if writer is not None and (epoch % 5 == 0) and batch_idx == 0:
                with torch.no_grad():
                    writer.add_image("samples/input", _to_01(xb[0]), epoch)
                    writer.add_image("samples/pred_dt", _to_01(pred[0]), epoch)
                    writer.add_image("samples/gt_dt", _to_01(yb[0]), epoch)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            running += float(loss.detach().cpu())
            running_dt += float(loss_dt.detach().cpu())
            running_eik_dt += float(loss_eik_dt.detach().cpu())

            if writer is not None:
                writer.add_scalar("train/loss", float(loss.detach().cpu()), global_step)
                writer.add_scalar("train/loss_dt", float(loss_dt.detach().cpu()), global_step)
                writer.add_scalar("train/loss_eik_dt", float(loss_eik_dt.detach().cpu()), global_step)

            global_step += 1

        denom = max(1, len(dl))
        epoch_loss = running / denom
        epoch_dt = running_dt / denom
        epoch_eik_dt = running_eik_dt / denom

        val_loss = None
        val_dt = None
        val_eik_dt = None

        if val_dl is not None:
            model.eval()
            val_running = 0.0
            val_running_dt = 0.0
            val_running_eik_dt = 0.0

            with torch.no_grad():
                for xb, yb, bgw_batch, stems in val_dl:
                    xb = xb.to(device, non_blocking=True)
                    yb = yb.to(device, non_blocking=True)
                    bgw_batch = bgw_batch.to(device, non_blocking=True).view(-1, 1, 1, 1)

                    with torch.amp.autocast(device_type=device.type, enabled=scaler.is_enabled()):
                        pred, _ = model(xb)
                        loss, loss_dt, loss_eik_dt = compute_losses(
                            pred=pred,
                            yb=yb,
                            bgw_batch=bgw_batch,
                            cfg=cfg,
                            loss_fn=loss_fn,
                        )

                    val_running += float(loss.detach().cpu())
                    val_running_dt += float(loss_dt.detach().cpu())
                    val_running_eik_dt += float(loss_eik_dt.detach().cpu())

            vden = max(1, len(val_dl))
            val_loss = val_running / vden
            val_dt = val_running_dt / vden
            val_eik_dt = val_running_eik_dt / vden

            if args.val_preview_every > 0 and (epoch % args.val_preview_every == 0):
                model.eval()
                with torch.no_grad():
                    saved = 0
                    for xb, yb, bgw_batch, stems in val_dl:
                        xb = xb.to(device, non_blocking=True)
                        yb = yb.to(device, non_blocking=True)

                        with torch.amp.autocast(device_type=device.type, enabled=scaler.is_enabled()):
                            pred, _ = model(xb)

                        bs = xb.shape[0]
                        for i in range(bs):
                            save_validation_preview(
                                out_dir=preview_dir,
                                epoch=epoch,
                                stem=stems[i],
                                xb=xb[i:i+1],
                                yb=yb[i:i+1],
                                pred=pred[i:i+1],
                            )
                            saved += 1
                            if saved >= args.val_preview_count:
                                break
                        if saved >= args.val_preview_count:
                            break

        lr_now = float(opt.param_groups[0]["lr"])
        epoch_sec = time.time() - t_start_epoch
        elapsed = time.time() - t_start_all
        epochs_done = max(1, epoch - start_epoch + 1)
        avg_epoch = elapsed / epochs_done
        remaining_epochs = max(0, cfg.epochs - epoch)
        eta_sec = remaining_epochs * avg_epoch

        msg = (
            f"epoch {epoch:03d}/{cfg.epochs}  "
            f"train_loss={epoch_loss:.6f}  "
            f"dt={epoch_dt:.6f}  "
            f"eik_dt={epoch_eik_dt:.6f}  "
        )
        if val_loss is not None:
            msg += (
                f"val_loss={val_loss:.6f}  "
                f"val_dt={val_dt:.6f}  "
                f"val_eik_dt={val_eik_dt:.6f}  "
            )
        msg += (
            f"lr={lr_now:.2e}  "
            f"time={_fmt_seconds(epoch_sec)}  "
            f"eta={_fmt_seconds(eta_sec)}"
        )
        print(msg)

        if writer is not None:
            writer.add_scalar("train/epoch_loss", float(epoch_loss), epoch)
            writer.add_scalar("train/epoch_dt", float(epoch_dt), epoch)
            writer.add_scalar("train/epoch_eik_dt", float(epoch_eik_dt), epoch)
            writer.add_scalar("train/lr", lr_now, epoch)
            if val_loss is not None:
                writer.add_scalar("val/epoch_loss", float(val_loss), epoch)
                writer.add_scalar("val/epoch_dt", float(val_dt), epoch)
                writer.add_scalar("val/epoch_eik_dt", float(val_eik_dt), epoch)

        metric_for_best = val_loss if val_loss is not None else epoch_loss
        if metric_for_best < (best_val if val_loss is not None else best_train):
            if val_loss is not None:
                best_val = metric_for_best
            else:
                best_train = metric_for_best
            torch.save(model.state_dict(), str(out_path))
            print(f"  saved best -> {out_path} (metric={metric_for_best:.6f})")

        if args.save_every > 0 and (epoch % args.save_every == 0):
            periodic_path = save_dir / f"{out_path.stem}_epoch_{epoch:04d}.pt"
            torch.save(model.state_dict(), str(periodic_path))
            print(f"  saved periodic -> {periodic_path}")

    if writer is not None:
        writer.close()

    print("done.")


if __name__ == "__main__":
    main()