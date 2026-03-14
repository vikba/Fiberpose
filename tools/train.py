#!/usr/bin/env python3
"""
train.py

Train your ResUnet (nout=3) with precomputed Omnipose-style targets:
  - flows = gradient-derived vector field (mu)
  - dt    = smooth distance field (T)

Channels:
  pred[:,0] -> flow_y
  pred[:,1] -> flow_x
  pred[:,2] -> dt (Omnipose T; background set to -dist_bg like Omnipose)

Requirements:
  pip install torch torchvision tifffile numpy opencv-python

Data layout (example):
  dataset/
    images/
      0001.tif
      0002.tif
      ...
    targets/
      0001_flow_y.tif
      0001_flow_x.tif
      0001_dt.tif
      0001_flow_y.png
      0001_flow_x.png
      0001_dt.png
      0001_targets.json
      ...

Run:
  python train.py --data dataset --out weights_fiberpose.pt

Optional filename template (match images to target stems):
  python train.py --data dataset --out weights.pt \
    --image-template "img_{stem}.tif"
"""

from __future__ import annotations

import argparse
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
# Model import (your project)
# -------------------------
# ResUnet lives in `sources/resunet.py`.
# Make sibling package imports work whether this script is run from repo root
# (`python tools/train.py`) or from inside `tools/`.
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
        # minimal fallback for png/jpg if you prefer; remove if not needed
        import cv2
        im = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if im is None:
            raise FileNotFoundError(f"Could not read image: {path}")
    im = np.asarray(im)

    # squeeze channels if needed
    if im.ndim == 3:
        # if RGB, convert to luminance-ish by average (simple)
        if im.shape[-1] in (3, 4):
            im = im[..., :3].mean(axis=-1)
        else:
            im = im[..., 0]

    im = im.astype(np.float32)
    # per-image min/max normalization (matches your inference behavior)
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


# -------------------------
# Augmentations (simple + safe)
# -------------------------
def random_flip_rotate(
    x: np.ndarray,
    flows: np.ndarray,
    dt: np.ndarray,
    rng: np.random.Generator
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    x:     (H,W)
    flows: (2,H,W) [y,x]
    dt:    (H,W)

    Applies random 90-degree rotations and flips, and updates flow components.
    """
    # random k*90 rotation
    k = int(rng.integers(0, 4))
    if k:
        x = np.rot90(x, k, axes=(0, 1))
        dt = np.rot90(dt, k, axes=(0, 1))
        fy = np.rot90(flows[0], k, axes=(0, 1))
        fx = np.rot90(flows[1], k, axes=(0, 1))

        # rotate vector components:
        # k=1: (fy,fx) -> (-fx, fy)
        # k=2: (fy,fx) -> (-fy,-fx)
        # k=3: (fy,fx) -> (fx, -fy)
        if k == 1:
            flows = np.stack([-fx, fy], axis=0)
        elif k == 2:
            flows = np.stack([-fy, -fx], axis=0)
        elif k == 3:
            flows = np.stack([fx, -fy], axis=0)
    else:
        flows = flows.copy()

    # horizontal flip (x-axis)
    if bool(rng.integers(0, 2)):
        x = np.flip(x, axis=1)
        dt = np.flip(dt, axis=1)
        flows = np.stack([np.flip(flows[0], axis=1), -np.flip(flows[1], axis=1)], axis=0)

    # vertical flip (y-axis)
    if bool(rng.integers(0, 2)):
        x = np.flip(x, axis=0)
        dt = np.flip(dt, axis=0)
        flows = np.stack([-np.flip(flows[0], axis=0), np.flip(flows[1], axis=0)], axis=0)

    return x, flows, dt


def random_crop(
    x: np.ndarray,
    flows: np.ndarray,
    dt: np.ndarray,
    crop: int,
    rng: np.random.Generator
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    H, W = x.shape
    if crop <= 0 or (H <= crop and W <= crop):
        return x, flows, dt

    ch = min(crop, H)
    cw = min(crop, W)
    y0 = int(rng.integers(0, H - ch + 1)) if H > ch else 0
    x0 = int(rng.integers(0, W - cw + 1)) if W > cw else 0

    x_c = x[y0:y0+ch, x0:x0+cw]
    dt_c = dt[y0:y0+ch, x0:x0+cw]
    flows_c = flows[:, y0:y0+ch, x0:x0+cw]
    return x_c, flows_c, dt_c


# -------------------------
# Dataset
# -------------------------
class OmniTargetsDataset(Dataset):
    def __init__(
        self,
        images_dir: Path,
        targets_dir: Path,
        target_template: str = "{stem}_{kind}.tif",
        image_template: str | None = None,
        crop: int = 512,
        augment: bool = True,
        seed: int = 0,
    ):
        self.images_dir = images_dir
        self.targets_dir = targets_dir
        self.target_template = target_template
        self.crop = crop
        self.augment = augment
        self.rng = np.random.default_rng(seed)

        self.image_template = image_template
        self._image_re = _template_to_regex(self.image_template) if self.image_template else None

        self.image_paths = sorted([p for p in images_dir.iterdir() if p.is_file()])
        if not self.image_paths:
            raise FileNotFoundError(f"No images found in: {images_dir}")

        self.target_stems: list[str] = []
        self.target_float_paths: list[Tuple[Path, Path, Path] | None] = []
        matched_images: list[Path] = []
        for ip in self.image_paths:
            if self._image_re is not None:
                m = self._image_re.match(ip.name)
                if not m:
                    continue
                stem = m.group("stem")
            else:
                stem = ip.stem
            fy_tif = targets_dir / self.target_template.format(stem=stem, kind="flow_y")
            fx_tif = targets_dir / self.target_template.format(stem=stem, kind="flow_x")
            dt_tif = targets_dir / self.target_template.format(stem=stem, kind="dt")
            if fy_tif.exists() and fx_tif.exists() and dt_tif.exists():
                self.target_float_paths.append((fy_tif, fx_tif, dt_tif))
            else:
                meta_path = targets_dir / f"{stem}_targets.json"
                fy_path = targets_dir / f"{stem}_flow_y.png"
                fx_path = targets_dir / f"{stem}_flow_x.png"
                dt_path = targets_dir / f"{stem}_dt.png"
                if not (meta_path.exists() and fy_path.exists() and fx_path.exists() and dt_path.exists()):
                    raise FileNotFoundError(
                        f"Missing targets for {ip.name}: expected {stem}_*.tif "
                        f"or {stem}_*.png and {stem}_targets.json"
                    )
                self.target_float_paths.append(None)
            matched_images.append(ip)
            self.target_stems.append(stem)

        self.image_paths = matched_images
        if not self.image_paths:
            msg = f"No images matched template {self.image_template} in {images_dir}" if self.image_template else f"No images found in: {images_dir}"
            raise FileNotFoundError(msg)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        ip = self.image_paths[idx]
        stem = self.target_stems[idx]
        float_paths = self.target_float_paths[idx]

        x = read_image_any(ip)             # (H,W) float32 [0,1]

        if float_paths is not None:
            fy_tif, fx_tif, dt_tif = float_paths
            flow_y = _load_float_tif(fy_tif)
            flow_x = _load_float_tif(fx_tif)
            flows = np.stack([flow_y, flow_x], axis=0)
            dt = _load_float_tif(dt_tif)
        else:
            meta_path = self.targets_dir / f"{stem}_targets.json"
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)

            flow_y_u16 = _load_u16_png(self.targets_dir / f"{stem}_flow_y.png")
            flow_x_u16 = _load_u16_png(self.targets_dir / f"{stem}_flow_x.png")
            dt_u16 = _load_u16_png(self.targets_dir / f"{stem}_dt.png")

            flows = np.stack([
                _denormalize_u16(flow_y_u16, meta["flow_y_min"], meta["flow_y_max"]),
                _denormalize_u16(flow_x_u16, meta["flow_x_min"], meta["flow_x_max"]),
            ], axis=0).astype(np.float32)
            dt = _denormalize_u16(dt_u16, meta["dt_min"], meta["dt_max"]).astype(np.float32)

        if self.augment:
            x, flows, dt = random_crop(x, flows, dt, self.crop, self.rng)
            x, flows, dt = random_flip_rotate(x, flows, dt, self.rng)
        else:
            x, flows, dt = random_crop(x, flows, dt, self.crop, self.rng)

        # build target (3,H,W): [flow_y, flow_x, dt]
        y = np.concatenate([flows, dt[None, ...]], axis=0).astype(np.float32)

        # torch tensors
        x_t = torch.from_numpy(np.ascontiguousarray(x[None, ...]))     # (1,H,W)
        y_t = torch.from_numpy(np.ascontiguousarray(y))                # (3,H,W)
        return x_t, y_t


# -------------------------
# Loss
# -------------------------
class WeightedSmoothL1(nn.Module):
    def __init__(self, beta: float = 1.0):
        super().__init__()
        self.beta = beta

    def forward(self, pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        # SmoothL1 with per-pixel weights
        # pred/target: (B,C,H,W) or (B,H,W)
        diff = torch.abs(pred - target)
        loss = torch.where(diff < self.beta, 0.5 * diff * diff / self.beta, diff - 0.5 * self.beta)
        return (loss * weight).mean()
    
# ---------------------------------------------------------
# Helper: Differentiable Sobel Filter for DT Gradient
# ---------------------------------------------------------
def get_spatial_gradient(image):
    """
    Computes dx, dy using Sobel filter.
    Input: (B, 1, H, W)
    Output: (B, 1, H, W) for dx, dy
    """
    kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                            dtype=image.dtype, device=image.device).view(1, 1, 3, 3)
    kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                            dtype=image.dtype, device=image.device).view(1, 1, 3, 3)
    
    # Padding=1 preserves dimensions
    dx = F.conv2d(image, kernel_x, padding=1)
    dy = F.conv2d(image, kernel_y, padding=1)
    
    # Sobel scale factor correction (usually divided by 8 for magnitude)
    return dx / 8.0, dy / 8.0


# -------------------------
# Train
# -------------------------
@dataclass
class TrainCfg:
    nbase: List[int] = None
    nout: int = 3
    lr: float = 1e-3
    batch: int = 4
    epochs: int = 50
    crop: int = 512
    num_workers: int = 2
    dist_bg: float = 5.0
    w_flow: float = 1.0
    w_dt: float = 1
    w_eik_flow: float = 0.5
    w_eik_dt: float = 0.5
    fg_weight: float = 5.0
    bg_weight: float = 1.0
    ignore_bg: bool = False
    amp: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True, help="dataset root with images/ and targets/")
    ap.add_argument("--out", type=str, required=True, help="output .pt (state_dict) path")
    ap.add_argument("--checkpoint", type=str, default=None, help="optional checkpoint to initialize from")
    ap.add_argument("--start-epoch", type=int, default=1, help="starting epoch index (1-based)")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=9)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--crop", type=int, default=512)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dist-bg", type=float, default=5.0)
    ap.add_argument("--w-flow", type=float, default=1.0)
    ap.add_argument("--w-dt", type=float, default=1.0)
    ap.add_argument("--fg-weight", type=float, default=3.0, help="loss weight for foreground pixels")
    ap.add_argument("--bg-weight", type=float, default=1.0, help="loss weight for background pixels")
    ap.add_argument("--ignore-bg", action="store_true", help="ignore background in loss (bg weight=0)")
    ap.add_argument("--image-template", type=str, default=None, help="image filename template with {stem}")
    ap.add_argument(
        "--target-template",
        type=str,
        default="{stem}_{kind}.tif",
        help="target filename template with {stem} and {kind} (kind=flow_y|flow_x|dt)",
    )
    ap.add_argument("--targets-dir", type=str, default=None, help="folder with precomputed targets (default: data/targets)")
    ap.add_argument("--logdir", type=str, default=None, help="tensorboard log dir (default: out_dir/runs)")
    ap.add_argument("--no-tensorboard", action="store_true", help="disable tensorboard logging")
    ap.add_argument("--no-amp", action="store_true")
    args = ap.parse_args()

    cfg = TrainCfg(
        nbase=[1, 32, 64, 128, 256],
        lr=args.lr,
        batch=args.batch,
        epochs=args.epochs,
        crop=args.crop,
        num_workers=args.workers,
        dist_bg=args.dist_bg,
        w_flow=args.w_flow,
        w_dt=args.w_dt,
        fg_weight=args.fg_weight,
        bg_weight=args.bg_weight,
        ignore_bg=args.ignore_bg,
        amp=not args.no_amp,
    )

    root = Path(args.data)
    images_dir = root / "images"
    targets_dir = Path(args.targets_dir) if args.targets_dir else (root / "targets")

    ds = OmniTargetsDataset(
        images_dir=images_dir,
        targets_dir=targets_dir,
        target_template=args.target_template,
        image_template=args.image_template,
        crop=cfg.crop,
        augment=True,
        seed=args.seed,
    )
    dl = DataLoader(
        ds,
        batch_size=cfg.batch,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    device = torch.device(cfg.device)

    # Instantiate your ResUnet with inference-compatible args
    model = ResUnet(
        nbase=cfg.nbase,
        nout=cfg.nout,
        sz=3,
        dilation=1,
        kernel_size=2,
        scale_factor=2,
        residual_on=True,
        style_on=True,
        concatenation=False,   # concat_style=False
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

    loss_fn = WeightedSmoothL1(beta=1.0)

    best = float("inf")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
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

    model.train()
    global_step = 0
    t_start_all = time.time()
    for epoch in range(start_epoch, cfg.epochs + 1):
        t_start_epoch = time.time()
        running = 0.0
        running_flow = 0.0
        running_dt = 0.0
        running_eik_flow = 0.0
        running_eik_dt = 0.0
        for batch_idx, (xb, yb) in enumerate(dl):
            xb = xb.to(device, non_blocking=True)        # (B,1,H,W)
            yb = yb.to(device, non_blocking=True)        # (B,3,H,W)

            # loss weights: control background vs foreground contribution
            # Here dt<=0 was clamped to -dist_bg, so "foreground-ish" can be dt > -dist_bg
            dt = yb[:, 2:3]
            fg = (dt > -cfg.dist_bg + 1e-6).float()
            bg = 1.0 - fg
            if cfg.ignore_bg:
                w = cfg.fg_weight * fg
            else:
                w = cfg.fg_weight * fg + cfg.bg_weight * bg
            

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=scaler.is_enabled()):
                pred, _ = model(xb)  # pred: (B,3,H,W)
                # separate weights for flows/dt if you want; reuse w for both
                loss_flow = loss_fn(pred[:, 0:2], yb[:, 0:2], w)
                loss_dt = loss_fn(pred[:, 2:3], yb[:, 2:3], w)

                # 2. Prepare Eikonal Calculation
                # We only enforce Eikonal INSIDE the objects (where GT distance > 0)
                # forcing grad=1 in background causes artifacts.
                obj_mask = (yb[:, 2:3] > 0).float()
                
                # -----------------------------------------------------
                # Eikonal A: Consistency of Predicted Flows
                # Check if the predicted flow vectors (Channels 0,1) have magnitude 1
                # -----------------------------------------------------
                pred_flow_mag = torch.sqrt(pred[:, 0]**2 + pred[:, 1]**2 + 1e-8).unsqueeze(1)
                # We want magnitude to be 1 inside the object
                loss_eik_flow = F.mse_loss(pred_flow_mag * obj_mask, obj_mask)

                # -----------------------------------------------------
                # Eikonal B: Physics of Predicted Distance
                # Calculate gradient of the predicted Distance Channel (Channel 2)
                # -----------------------------------------------------
                dt_dx, dt_dy = get_spatial_gradient(pred[:, 2:3])
                pred_dt_mag = torch.sqrt(dt_dx**2 + dt_dy**2 + 1e-8)

                # We want derived gradient magnitude to be 1 inside the object
                loss_eik_dt = F.mse_loss(pred_dt_mag * obj_mask, obj_mask)
                
                # We want derived gradient magnitude to be 1 inside the object
                loss = (cfg.w_flow * loss_flow + 
                    cfg.w_dt   * loss_dt + 
                    cfg.w_eik_flow * loss_eik_flow + 
                    cfg.w_eik_dt * loss_eik_dt)


            if writer is not None and (epoch % 5 == 0) and batch_idx == 0:
                with torch.no_grad():
                    writer.add_image("samples/input", _to_01(xb[0]), epoch)
                    writer.add_image("samples/pred_flow_y", _to_01(pred[0, 0:1]), epoch)
                    writer.add_image("samples/pred_flow_x", _to_01(pred[0, 1:2]), epoch)
                    writer.add_image("samples/pred_dt", _to_01(pred[0, 2:3]), epoch)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            running += float(loss.detach().cpu())
            running_flow += float(loss_flow.detach().cpu())
            running_dt += float(loss_dt.detach().cpu())
            running_eik_flow += float(loss_eik_flow.detach().cpu())
            running_eik_dt += float(loss_eik_dt.detach().cpu())
            if writer is not None:
                writer.add_scalar("train/loss", float(loss.detach().cpu()), global_step)
            global_step += 1

        denom = max(1, len(dl))
        epoch_loss = running / denom
        epoch_flow = running_flow / denom
        epoch_dt = running_dt / denom
        epoch_eik_flow = running_eik_flow / denom
        epoch_eik_dt = running_eik_dt / denom
        lr_now = float(opt.param_groups[0]["lr"])
        epoch_sec = time.time() - t_start_epoch
        elapsed = time.time() - t_start_all
        epochs_done = max(1, epoch - start_epoch + 1)
        avg_epoch = elapsed / epochs_done
        remaining_epochs = max(0, cfg.epochs - epoch)
        eta_sec = remaining_epochs * avg_epoch
        print(
            f"epoch {epoch:03d}/{cfg.epochs}  "
            f"loss={epoch_loss:.6f}  flow={epoch_flow:.6f}  dt={epoch_dt:.6f}  "
            f"eik_flow={epoch_eik_flow:.6f}  eik_dt={epoch_eik_dt:.6f}  "
            f"lr={lr_now:.2e}  "
            f"time={_fmt_seconds(epoch_sec)}  "
            f"eta={_fmt_seconds(eta_sec)}"
        )
        if writer is not None:
            writer.add_scalar("train/epoch_loss", float(epoch_loss), epoch)
            writer.add_scalar("train/lr", float(opt.param_groups[0]["lr"]), epoch)

        # save best
        if epoch_loss < best:
            best = epoch_loss
            torch.save(model.state_dict(), str(out_path))
            print(f"  saved best -> {out_path} (loss={best:.6f})")

    if writer is not None:
        writer.close()
    print("done.")


if __name__ == "__main__":
    main()
