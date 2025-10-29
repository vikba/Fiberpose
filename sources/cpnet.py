# fiberpose_cpnet.py
# One-file TRAIN + PREDICT script for CPnet

import os, glob, argparse, pathlib, random
from typing import List, Tuple, Optional
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from torch.amp import autocast
import torch.utils.checkpoint as cp

# ---------------- Optional GPU helpers ----------------

torch_GPU = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch_CPU = torch.device("cpu")
def empty_cache():
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

# -------------------- CPnet (your code; minimal-safe load tweak) --------------------

def dilation_list(x,N):
    return np.round(np.linspace(1, x, N)).astype(int).tolist()

def batchconv(in_channels, out_channels, kernel_size, dim, dilation, relu=True):
    BatchNorm = nn.BatchNorm2d if dim == 2 else nn.BatchNorm3d
    ConvND = nn.Conv2d if dim == 2 else nn.Conv3d
    padding = ((kernel_size - 1) * dilation) // 2
    layers = [BatchNorm(in_channels, eps=1e-5, momentum=0.05)]
    if relu:
        layers.append(nn.ReLU(inplace=True))
    layers.append(ConvND(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation,
                         padding_mode='reflect'))
    return nn.Sequential(*layers)

class resdown(nn.Module):
    def __init__(self, in_channels, out_channels, sz, dim, dilation):
        super().__init__()
        self.conv = nn.Sequential()
        self.proj = batchconv(in_channels, out_channels, 1, dim, 1, relu=False)
        for t in range(4):
            if t==0:
                self.conv.add_module('conv_%d'%t, batchconv(in_channels, out_channels, sz, dim, dilation))
            else:
                self.conv.add_module('conv_%d'%t, batchconv(out_channels, out_channels, sz, dim, dilation))
    def forward(self, x):
        x = self.proj(x) + self.conv[1](self.conv[0](x))
        x = x + self.conv[3](self.conv[2](x))
        return x

class convdown(nn.Module):
    def __init__(self, in_channels, out_channels, sz, dim, dilation):
        super().__init__()
        self.conv = nn.Sequential()
        for t in range(2):
            if t==0:
                self.conv.add_module('conv_%d'%t, batchconv(in_channels, out_channels, sz, dim, dilation))
            else:
                self.conv.add_module('conv_%d'%t, batchconv(out_channels, out_channels, sz, dim, dilation))
    def forward(self, x):
        x = self.conv[0](x)
        x = self.conv[1](x)
        return x

class batchconvstyle(nn.Module):
    def __init__(self, in_channels, out_channels, style_channels, sz, dim, dilation, concatenation=False):
        super().__init__()
        self.conv = batchconv(in_channels, out_channels, sz, dim, dilation)
        if concatenation:
            self.full = nn.Linear(style_channels, out_channels*2)
        else:
            self.full = nn.Linear(style_channels, out_channels)
        self.dim = dim
    def forward(self, style, x, mkldnn=False, y=None):
        if y is not None:
            x = x + y
        feat = self.full(style)
        for _ in range(self.dim):
            feat = feat.unsqueeze(-1)
        if mkldnn:
            x = x.to_dense()
            y = (x + feat).to_mkldnn()
        else:
            y = x + feat
        y = self.conv(y)
        return y

class resup(nn.Module):
    def __init__(self, in_channels, out_channels, style_channels, parent, dilation=None):
        super().__init__()
        sz = parent.sz
        concatenation = parent.concatenation
        dim = parent.dim
        dilation = parent.dilation if dilation is None else dilation
        self.conv = nn.Sequential()
        self.conv.add_module('conv_0', batchconv(in_channels, out_channels, sz, dim, dilation))
        self.conv.add_module('conv_1', batchconvstyle(out_channels, out_channels, style_channels, sz, dim, dilation, concatenation=concatenation))
        self.conv.add_module('conv_2', batchconvstyle(out_channels, out_channels, style_channels, sz, dim, dilation))
        self.conv.add_module('conv_3', batchconvstyle(out_channels, out_channels, style_channels, sz, dim, dilation))
        self.proj = batchconv(in_channels, out_channels, 1, dim, 1, relu=False)
    def forward(self, x, y, style, mkldnn=False):
        x = self.proj(x) + self.conv[1](style, self.conv[0](x) + y, mkldnn=mkldnn)
        x = x + self.conv[3](style, self.conv[2](style, x, mkldnn=mkldnn), mkldnn=mkldnn)
        return x

class convup(nn.Module):
    def __init__(self, in_channels, out_channels, style_channels, parent, dilation=None):
        super().__init__()
        sz = parent.sz
        concatenation = parent.concatenation
        dim = parent.dim
        dilation = parent.dilation if dilation is None else dilation
        self.conv = nn.Sequential()
        self.conv.add_module('conv_0', batchconv(in_channels, out_channels, sz, dim, dilation))
        self.conv.add_module('conv_1', batchconvstyle(out_channels, out_channels, style_channels, sz, dim, dilation, concatenation=concatenation))
    def forward(self, x, y, style):
        x = self.conv[1](style, self.conv[0](x) + y)
        return x

class downsample(nn.Module):
    def __init__(self, parent):
        super().__init__()
        nbase = parent.nbase
        sz = parent.sz
        residual_on = parent.residual_on
        dim = parent.dim
        dilation = parent.dilation
        kernel_size = parent.kernel_size
        scale_factor = parent.scale_factor
        self.checkpoint = parent.checkpoint
        self.down = nn.Sequential()
        maxpool = nn.MaxPool2d if dim == 2 else nn.MaxPool3d
        self.maxpool = maxpool(kernel_size=kernel_size, stride=scale_factor)
        N = len(nbase)-1
        dilations = dilation_list(dilation, N)
        for n, dil in enumerate(dilations):
            if residual_on:
                self.down.add_module('res_down_%d'%n, resdown(nbase[n], nbase[n+1], sz, dim, dil))
            else:
                self.down.add_module('conv_down_%d'%n, convdown(nbase[n], nbase[n+1], sz, dim, dil))
    def forward(self, x):
        xd = []
        for n in range(len(self.down)):
            y = self.maxpool(xd[n-1]) if n>0 else x
            xd.append(self.down[n](y))
        return xd

class upsample(nn.Module):
    def __init__(self, parent):
        super().__init__()
        nbase = parent.nbaseup
        self.upsampling = nn.Upsample(scale_factor=parent.scale_factor, mode='nearest')
        self.up = nn.Sequential()
        self.checkpoint = parent.checkpoint
        N = len(nbase)-1
        dilations = dilation_list(parent.dilation, N)
        for k, dil in enumerate(dilations):
            n = k+1
            if parent.residual_on:
                self.up.add_module('res_up_%d'%(n-1), resup(nbase[n], nbase[n-1], nbase[-1], parent, dilation=dil))
            else:
                self.up.add_module('conv_up_%d'%(n-1), convup(nbase[n], nbase[n-1], nbase[-1], parent, dilation=dil))
    def forward(self, style, xd, mkldnn=False):
        x = xd[-1]
        for n in range(len(self.up)):
            idx = -(n + 1)
            if n > 0:
                x = self.upsampling(x.to_dense()).to_mkldnn() if mkldnn else self.upsampling(x)
            x = self.up[idx](x, xd[idx], style, mkldnn=mkldnn) if "res_up" in self.up[idx]._get_name() else self.up[idx](x, xd[idx], style)
        return x

class make_style(nn.Module):
    def __init__(self, parent):
        super().__init__()
        self.dim = parent.dim
        self.flatten = nn.Flatten()
        self.avg_pool = F.avg_pool2d if self.dim==2 else F.avg_pool3d
    def forward(self, x0):
        style = self.avg_pool(x0, kernel_size=tuple(x0.shape[-self.dim:]))
        style = self.flatten(style)
        style = style / torch.sum(style**2, axis=1, keepdim=True)**.5
        return style

def _safe_torch_load(path, map_location, weights_only=True):
    try:
        return torch.load(path, map_location=map_location, weights_only=weights_only)
    except TypeError:
        return torch.load(path, map_location=map_location)

class CPnet(nn.Module):
    def __init__(self, nbase, nout, sz, residual_on=True,
                 style_on=True, concatenation=False, mkldnn=False, dim=2,
                 checkpoint=False, dropout=False, kernel_size=2, scale_factor=2, dilation=1):
        super(CPnet, self).__init__()
        self.checkpoint = checkpoint
        self.kernel_size = kernel_size
        self.scale_factor = scale_factor
        self.dilation = dilation
        self.nbase = nbase
        self.nout = nout
        self.sz = sz
        self.dim = dim
        self.residual_on = residual_on
        self.style_on = style_on
        self.concatenation = concatenation
        self.mkldnn = mkldnn if mkldnn is not None else False
        self.downsample = downsample(self)
        nbaseup = nbase[1:]
        nbaseup.append(nbaseup[-1])
        self.nbaseup = nbaseup
        self.upsample = upsample(self)
        self.make_style = make_style(self)
        self.output = batchconv(nbaseup[0], nout, 1, self.dim, 1)
        self.style_on = style_on
        self.do_dropout = dropout
        if self.do_dropout:
            self.dropout = nn.Dropout(0.1)
    def forward(self, data):
        if self.mkldnn:
            data = data.to_mkldnn()
        T0 = self.downsample(data)
        style = self.make_style(T0[-1].to_dense()) if self.mkldnn else self.make_style(T0[-1])
        style0 = style
        if not self.style_on:
            style = style * 0
        T0 = self.upsample(style, T0, self.mkldnn)
        if self.do_dropout:
            T0 = self.dropout(T0)
        T0 = self.output(T0)
        if self.mkldnn:
            T0 = T0.to_dense()
        return T0, style0
    def save_model(self, filename):
        torch.save(self.state_dict(), filename)
    def load_model(self, filename, cpu=False):
        if not cpu:
            state_dict = _safe_torch_load(filename, map_location=torch_GPU, weights_only=True)
            self.load_state_dict(state_dict)
        else:
            self.__init__(self.nbase, self.nout, self.sz, self.residual_on, self.style_on,
                          self.concatenation, self.mkldnn, self.dim, self.checkpoint,
                          self.do_dropout, self.kernel_size, self.scale_factor, self.dilation)
            state_dict = _safe_torch_load(filename, map_location=torch_CPU, weights_only=True)
            self.load_state_dict(state_dict, strict=False)

# ---------------------------- Data utilities ----------------------------

def list_images(folder: str) -> List[str]:
    exts = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp")
    return sorted([str(p) for p in pathlib.Path(folder).glob("*") if p.suffix.lower() in exts])

def load_image_as_chw(path: str, channels: int) -> np.ndarray:
    img = Image.open(path)
    if channels == 1:
        img = img.convert("F")
        arr = np.asarray(img, dtype=np.float32)  # (H, W)
        arr = arr[None, ...]
    elif channels == 3:
        img = img.convert("RGB")
        arr = np.asarray(img, dtype=np.float32)  # (H, W, 3)
        arr = np.transpose(arr, (2,0,1))
    else:
        # Keep as-is if multi-channel formats are passed
        arr = np.asarray(img, dtype=np.float32)
        if arr.ndim == 2 and channels == 1:
            arr = arr[None, ...]
        elif arr.ndim == 3 and arr.shape[-1] == channels:
            arr = np.transpose(arr, (2,0,1))
        else:
            raise ValueError(f"Unsupported input for {path} with requested channels={channels}")
    # normalize per-image to [0,1] if dynamic range > 0
    vmin, vmax = float(np.min(arr)), float(np.max(arr))
    if vmax > vmin:
        arr = (arr - vmin) / (vmax - vmin + 1e-12)
    return arr.astype(np.float32)

def save_tensor_as_pngs(out_base: str, y: torch.Tensor):
    arr = y.detach().cpu().numpy()  # (B,C,H,W)
    np.save(out_base + ".npy", arr)
    b, c, h, w = arr.shape
    for k in range(c):
        ch = arr[0, k]
        lo, hi = float(ch.min()), float(ch.max())
        ch8 = np.zeros_like(ch, dtype=np.uint8) if hi<=lo else ((ch - lo)/(hi-lo)*255.0).astype(np.uint8)
        Image.fromarray(ch8).save(out_base + f"_ch{k}.png")

class PairedFolderDataset(Dataset):
    """
    Expects two folders with matching filenames (img vs gt).
    e.g., imgs/xxx.png and gt/xxx.png
    """
    def __init__(self, in_dir: str, gt_dir: str, in_channels: int, out_channels: int,
                 crop_size: Optional[int]=None, random_hflip=True, random_vflip=True):
        self.in_dir = in_dir
        self.gt_dir = gt_dir
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.random_hflip = random_hflip
        self.random_vflip = random_vflip
        self.crop_size = crop_size

        self.in_files = list_images(in_dir)
        if not self.in_files:
            raise RuntimeError(f"No images in {in_dir}")
        # build paired list by basename
        gt_map = {pathlib.Path(p).stem: p for p in list_images(gt_dir)}
        pairs = []
        for p in self.in_files:
            stem = pathlib.Path(p).stem
            if stem in gt_map:
                pairs.append((p, gt_map[stem]))
        if not pairs:
            raise RuntimeError("No matched image/gt pairs by stem name.")
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def _random_crop(self, a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray,np.ndarray]:
        # a:(C,H,W), b:(C2,H,W)
        if self.crop_size is None:
            return a, b
        C,H,W = a.shape
        if H < self.crop_size or W < self.crop_size:
            # pad reflect
            pad_h = max(0, self.crop_size - H)
            pad_w = max(0, self.crop_size - W)
            a = np.pad(a, ((0,0),(pad_h//2, pad_h - pad_h//2),(pad_w//2, pad_w - pad_w//2)), mode="reflect")
            b = np.pad(b, ((0,0),(pad_h//2, pad_h - pad_h//2),(pad_w//2, pad_w - pad_w//2)), mode="reflect")
            C,H,W = a.shape
        y = random.randint(0, H - self.crop_size)
        x = random.randint(0, W - self.crop_size)
        return a[:, y:y+self.crop_size, x:x+self.crop_size], b[:, y:y+self.crop_size, x:x+self.crop_size]

    def __getitem__(self, idx):
        ipath, gpath = self.pairs[idx]
        x = load_image_as_chw(ipath, self.in_channels)
        y = load_image_as_chw(gpath, self.out_channels)

        # optional augment
        if self.crop_size:
            x, y = self._random_crop(x, y)
        if self.random_hflip and random.random() < 0.5:
            x = x[:, :, ::-1].copy(); y = y[:, :, ::-1].copy()
        if self.random_vflip and random.random() < 0.5:
            x = x[:, ::-1, :].copy(); y = y[:, ::-1, :].copy()

        return torch.from_numpy(x), torch.from_numpy(y)

# ---------------------------- Training ----------------------------

def build_model_from_args(args, device):
    nbase = [int(z) for z in args.nbase.split(",") if z.strip()]
    model = CPnet(
        nbase=nbase,
        nout=args.nout,
        sz=args.sz,
        residual_on=not args.no_residual,
        style_on=not args.no_style_off,
        concatenation=args.concat_style,
        mkldnn=False,
        dim=2,
        checkpoint=args.checkpointing,
        dropout=args.dropout,
        kernel_size=args.pool_kernel,
        scale_factor=args.scale,
        dilation=args.dilation,
    ).to(device)
    return model

def make_optimizer(args, model):
    if args.opt.lower() == "adam":
        return torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd, betas=(0.9, 0.999))
    if args.opt.lower() == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd, betas=(0.9, 0.999))
    if args.opt.lower() == "sgd":
        return torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd, nesterov=True)
    raise ValueError(f"Unknown optimizer {args.opt}")

def make_scheduler(args, optimizer):
    if args.sched == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr_min)
    if args.sched == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    return None

def make_loss(args):
    if args.loss.lower() == "l1":
        return nn.L1Loss()
    if args.loss.lower() == "mse":
        return nn.MSELoss()
    # Huber as a robust option
    if args.loss.lower() in ("huber","smoothl1"):
        return nn.SmoothL1Loss(beta=args.huber_beta)
    raise ValueError(f"Unknown loss {args.loss}")

def train_one_epoch(model, loader, criterion, optimizer, device, scaler, amp_dtype, clip_grad=None):
    model.train()
    running, n = 0.0, 0
    for x,y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            with autocast(device_type="cuda", dtype=amp_dtype):
                yhat, _ = model(x)
                loss = criterion(yhat, y)
            scaler.scale(loss).backward()
            if clip_grad is not None:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            scaler.step(optimizer)
            scaler.update()
        else:
            yhat, _ = model(x)
            loss = criterion(yhat, y)
            loss.backward()
            if clip_grad is not None:
                nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()
        bs = x.size(0)
        running += float(loss.item()) * bs
        n += bs
    return running / max(1,n)

@torch.no_grad()
def evaluate(model, loader, criterion, device, amp_dtype, use_amp):
    model.eval()
    running, n = 0.0, 0
    for x,y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        if use_amp:
            with autocast(device_type="cuda", dtype=amp_dtype):
                yhat,_ = model(x)
                loss = criterion(yhat, y)
        else:
            yhat,_ = model(x)
            loss = criterion(yhat, y)
        bs = x.size(0)
        running += float(loss.item()) * bs
        n += bs
    return running / max(1,n)

def save_ckpt(outdir, name, model, optimizer=None, epoch=None, best_val=None):
    path = os.path.join(outdir, name)
    os.makedirs(outdir, exist_ok=True)
    # Save pure state_dict for compatibility with your loader
    torch.save(model.state_dict(), path)
    # Optionally, save a trainer snapshot next to it
    snap = dict(epoch=epoch, best_val=best_val,
                optimizer=(optimizer.state_dict() if optimizer else None))
    torch.save(snap, os.path.join(outdir, name.replace(".pt","_trainer.pt")))
    return path

def cmd_train(args):
    device = torch_GPU if (torch.cuda.is_available() and not args.cpu) else torch_CPU
    nbase = [int(z) for z in args.nbase.split(",") if z.strip()]
    in_ch, out_ch = nbase[0], args.nout

    # Datasets
    train_ds = PairedFolderDataset(args.train_in, args.train_gt, in_ch, out_ch,
                                   crop_size=args.crop, random_hflip=not args.no_hflip,
                                   random_vflip=not args.no_vflip)
    val_ds = PairedFolderDataset(args.val_in, args.val_gt, in_ch, out_ch,
                                 crop_size=None, random_hflip=False, random_vflip=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=args.val_batch_size, shuffle=False,
                              num_workers=args.workers, pin_memory=True)

    model = build_model_from_args(args, device)
    criterion = make_loss(args).to(device)
    optimizer = make_optimizer(args, model)
    scheduler = make_scheduler(args, optimizer)

    # AMP
    use_amp = (device.type == "cuda") and args.amp
    if device.type == "cuda":
        amp_dtype = torch.float16 if args.fp16 else (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
    else:
        amp_dtype = torch.float32
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and (amp_dtype is torch.float16))


    # Resume weights (pure state_dict) if provided
    if args.resume:
        print(f"Resuming weights from {args.resume}")
        state_dict = _safe_torch_load(args.resume, map_location=device, weights_only=True)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print("Resume missing keys:", missing)
        print("Resume unexpected keys:", unexpected)

    # Train loop
    best_val = float("inf")
    os.makedirs(args.outdir, exist_ok=True)
    for epoch in range(1, args.epochs+1):
        tr_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler,
                                  amp_dtype, clip_grad=args.clip_grad)
        va_loss = evaluate(model, val_loader, criterion, device, amp_dtype, use_amp)
        if scheduler is not None:
            scheduler.step()

        print(f"Epoch {epoch:03d}: train {tr_loss:.6f} | val {va_loss:.6f} | lr {optimizer.param_groups[0]['lr']:.3e}")

        # Save last and best
        save_ckpt(args.outdir, "last.pt", model, optimizer, epoch, best_val)
        if va_loss < best_val - 1e-7:
            best_val = va_loss
            save_ckpt(args.outdir, "best.pt", model, optimizer, epoch, best_val)

    print(f"Done. Best val loss: {best_val:.6f}")

# ---------------------------- Predict ----------------------------

def parse_nbase(s: str):
    vals = [int(x.strip()) for x in s.split(",") if x.strip()]
    if len(vals) < 2:
        raise ValueError("nbase must have ≥2 entries.")
    return vals

def run_predict(args):
    device = torch_GPU if (torch.cuda.is_available() and not args.cpu) else torch_CPU
    nbase = parse_nbase(args.nbase)
    in_channels = nbase[0]

    model = CPnet(
        nbase=nbase, nout=args.nout, sz=args.sz,
        residual_on=not args.no_residual,
        style_on=not args.no_style_off,
        concatenation=args.concat_style,
        mkldnn=False, dim=2, checkpoint=args.checkpointing, dropout=args.dropout,
        kernel_size=args.pool_kernel, scale_factor=args.scale, dilation=args.dilation,
    ).to(device)

    model.load_model(args.ckpt, cpu=(device.type=="cpu"))
    model.eval()

    # Collect inputs
    if any(ch in args.input for ch in "*?[]"):
        files = sorted(glob.glob(args.input))
    else:
        p = pathlib.Path(args.input)
        if p.is_dir():
            files = list_images(str(p))
        else:
            files = [str(p)]
    if not files:
        print(f"No inputs found: {args.input}")
        return

    os.makedirs(args.output, exist_ok=True)
    use_amp = device.type == "cuda" and args.amp
    if device.type == "cuda":
        amp_dtype = torch.float16 if args.fp16 else (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
    else:
        amp_dtype = torch.float32


    for f in files:
        try:
            arr = load_image_as_chw(f, in_channels)  # (C,H,W)
        except Exception as e:
            print(f"[WARN] skip {f}: {e}")
            continue
        x = torch.from_numpy(arr[None, ...]).to(device)
        with torch.no_grad():
            if use_amp:
                with autocast(device_type="cuda", dtype=amp_dtype):
                    y,_ = model(x)
            else:
                y,_ = model(x)
        stem = pathlib.Path(f).stem
        out_base = os.path.join(args.output, stem)
        save_tensor_as_pngs(out_base, y)
        print(f"[OK] {f} -> {out_base}.npy/_ch*.png, shape={tuple(y.shape)}")
    empty_cache()

# ---------------------------- CLI ----------------------------

def build_parser():
    p = argparse.ArgumentParser(description="CPnet: train & predict")
    sub = p.add_subparsers(dest="cmd", required=True)

    # Shared arch args
    def add_arch_args(ap):
        ap.add_argument("--nbase", type=str, default="1,32,64,128,256",
                        help="Comma list of channels per scale (first is input channels)")
        ap.add_argument("--nout", type=int, default=3)
        ap.add_argument("--sz", type=int, default=3, help="kernel size inside residual/conv blocks")
        ap.add_argument("--pool_kernel", type=int, default=2)
        ap.add_argument("--scale", type=int, default=2)
        ap.add_argument("--dilation", type=int, default=1)
        ap.add_argument("--no_residual", action="store_true")
        ap.add_argument("--no_style_off", action="store_true",
                        help="If set, turns style OFF; by default style is ON. (Flag name kept for backward compat)")
        ap.add_argument("--concat_style", action="store_true",
                        help="Use concatenation mode in batchconvstyle (rarely needed)")
        ap.add_argument("--checkpointing", action="store_true")
        ap.add_argument("--dropout", action="store_true")

        ap.add_argument("--cpu", action="store_true")
        ap.add_argument("--amp", action="store_true", help="Enable AMP on CUDA")
        ap.add_argument("--fp16", action="store_true", help="Use fp16 for AMP (default bfloat16)")

    # TRAIN
    t = sub.add_parser("train", help="Train CPnet")
    add_arch_args(t)
    t.add_argument("--train_in", type=str, required=True)
    t.add_argument("--train_gt", type=str, required=True)
    t.add_argument("--val_in", type=str, required=True)
    t.add_argument("--val_gt", type=str, required=True)
    t.add_argument("--outdir", type=str, required=True)
    t.add_argument("--resume", type=str, default=None, help="Path to state_dict to warm-start")

    t.add_argument("--epochs", type=int, default=100)
    t.add_argument("--batch_size", type=int, default=8)
    t.add_argument("--val_batch_size", type=int, default=8)
    t.add_argument("--workers", type=int, default=4)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--lr_min", type=float, default=1e-5)
    t.add_argument("--opt", type=str, default="adam", choices=["adam","adamw","sgd"])
    t.add_argument("--wd", type=float, default=0.0)
    t.add_argument("--sched", type=str, default="cosine", choices=["none","cosine","step"])
    t.add_argument("--step_size", type=int, default=30)
    t.add_argument("--gamma", type=float, default=0.5)
    t.add_argument("--loss", type=str, default="l1", choices=["l1","mse","huber","smoothl1"])
    t.add_argument("--huber_beta", type=float, default=1.0)
    t.add_argument("--clip_grad", type=float, default=None, help="e.g., 1.0 to clip; None to disable")
    t.add_argument("--crop", type=int, default=256, help="Random crop size for training; 0 to disable")
    t.add_argument("--no_hflip", action="store_true")
    t.add_argument("--no_vflip", action="store_true")

    # PREDICT
    pr = sub.add_parser("predict", help="Run inference")
    add_arch_args(pr)
    pr.add_argument("--ckpt", type=str, required=True)
    pr.add_argument("--input", type=str, required=True, help="File/dir/glob")
    pr.add_argument("--output", type=str, required=True)

    return p

def main():
    args = build_parser().parse_args()
    if args.cmd == "train":
        if args.crop == 0:
            args.crop = None
        cmd_train(args)
    elif args.cmd == "predict":
        run_predict(args)
    else:
        raise SystemExit("Unknown command")

if __name__ == "__main__":
    main()
