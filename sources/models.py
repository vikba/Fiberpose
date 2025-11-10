# app/nn_predictor.py
import logging
import os
from typing import Optional
import numpy as np
import torch
from torch.amp import autocast

from .resunet import ResUnet, parse_nbase, _safe_torch_load, torch_GPU, torch_CPU

logger = logging.getLogger(__name__)

class FiberposeModel:
    def __init__(
        self,
        ckpt_path: str,
        nbase: str = "1,32,64,128,256",
        nout: int = 3,
        sz: int = 3,
        pool_kernel: int = 2,
        scale: int = 2,
        dilation: int = 1,
        residual_on: bool = True,
        style_on: bool = True,
        concat_style: bool = False,
        checkpointing: bool = False,
        dropout: bool = False,
        cpu: bool = False,            # kept for back-compat; ignored if `device` given
        amp: bool = True,
        fp16: bool = False,
        dt_channel: int = 0,
        device: Optional[torch.device | str] = None,  # NEW: 'cuda', 'cuda:1', 'cpu', etc.
    ):
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        # ---- resolve device (prefers explicit `device`, else legacy `cpu`, else auto) ----
        def _resolve_device(dev_arg, legacy_cpu_flag):
            if dev_arg is not None:
                d = torch.device(dev_arg) if not isinstance(dev_arg, torch.device) else dev_arg
                if d.type == "cuda" and not torch.cuda.is_available():
                    print("[WARN] Requested CUDA but no CUDA available; falling back to CPU.")
                    return torch_CPU
                return d
            if legacy_cpu_flag:
                return torch_CPU
            return torch_GPU if torch.cuda.is_available() else torch_CPU

        self.device = _resolve_device(device, cpu)

        # ---- AMP config (only meaningful on CUDA) ----
        self.use_amp = (self.device.type == "cuda") and amp
        if self.device.type == "cuda":
            if fp16:
                self.amp_dtype = torch.float16
            else:
                self.amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            self.amp_dtype = torch.float32  # not used unless you later enable CPU autocast

        self.nbase_str = nbase
        self.nbase = parse_nbase(nbase)
        self.nout = int(nout)
        self.dt_channel = int(dt_channel)

        self.model = ResUnet(
            nbase=self.nbase, nout=self.nout, sz=sz,
            residual_on=residual_on, style_on=style_on, concatenation=concat_style,
            mkldnn=False, dim=2, checkpoint=checkpointing, dropout=dropout,
            kernel_size=pool_kernel, scale_factor=scale, dilation=dilation,
        ).to(self.device)

        try:
            self.model.load_model(ckpt_path, cpu=(self.device.type == "cpu"))
        except Exception as exc:
            logger.exception("Failed to load checkpoint '%s'", ckpt_path)
            raise
        logger.info("Loaded checkpoint '%s' on device '%s'", ckpt_path, self.device)
        self.model.eval()

    @torch.no_grad()
    def predict(self, img: np.ndarray) -> np.ndarray:
        in_ch = self.nbase[0]
        arr = self._prep_input(img, in_ch)
        x = torch.from_numpy(arr[None, ...]).to(self.device)

        if self.use_amp:
            with autocast(device_type="cuda", dtype=self.amp_dtype):
                y, _ = self.model(x)
        else:
            y, _ = self.model(x)

        y = y.detach().cpu().numpy()
        dt = y[0, 2, ...].astype(np.float32)
        return dt


    @staticmethod
    def _prep_input(x: np.ndarray, in_channels: int) -> np.ndarray:
        if x.ndim == 2:
            arr = x.astype(np.float32)[None, ...]  # (1,H,W)
        elif x.ndim == 3:
            if x.shape[2] == in_channels:
                arr = np.transpose(x, (2, 0, 1)).astype(np.float32)
            else:
                if in_channels == 1:
                    if x.shape[2] >= 3:
                        r,g,b = x[...,0], x[...,1], x[...,2]
                        arr = (0.2126*r + 0.7152*g + 0.0722*b).astype(np.float32)[None, ...]
                    else:
                        arr = x[...,0:1].astype(np.float32).transpose(2,0,1)
                elif in_channels == 3:
                    if x.shape[2] == 1:
                        arr = np.repeat(x, 3, axis=2).astype(np.float32).transpose(2,0,1)
                    else:
                        arr = x[..., :3].astype(np.float32).transpose(2,0,1)
                else:
                    raise ValueError(f"Unsupported channel adapt; in_channels={in_channels}, x.shape={x.shape}")
        else:
            raise ValueError(f"Unsupported input shape {x.shape}")

        vmin, vmax = float(np.min(arr)), float(np.max(arr))
        if vmax > vmin:
            arr = (arr - vmin) / (vmax - vmin + 1e-12)
        else:
            arr = np.zeros_like(arr, dtype=np.float32)
        return arr.astype(np.float32)
