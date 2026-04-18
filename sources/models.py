"""Inference wrapper for Fiberpose ResUnet checkpoints."""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

import numpy as np
import torch
from torch.amp import autocast

from .resunet import ResUnet, parse_nbase, torch_GPU, torch_CPU

logger = logging.getLogger(__name__)


def _safe_torch_load(path: str, map_location: Any, weights_only: bool = True) -> Any:
    """Load a PyTorch object while remaining compatible with older torch versions."""
    try:
        return torch.load(path, map_location=map_location, weights_only=weights_only)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _extract_state_dict(payload: Any) -> dict[str, Any]:
    """Extract state dict from either a raw state dict or a checkpoint dict."""
    if isinstance(payload, dict):
        model_payload = payload.get("model")
        if isinstance(model_payload, dict):
            return model_payload
        if all(isinstance(k, str) for k in payload.keys()):
            return payload
    raise ValueError("Checkpoint does not contain a recognizable state_dict payload.")


def _infer_nout_from_state_dict(state_dict: dict[str, Any]) -> int | None:
    """Infer output channels from checkpoint tensors."""
    primary_key = "output.2.weight"
    primary = state_dict.get(primary_key)
    if hasattr(primary, "ndim") and int(primary.ndim) >= 3 and hasattr(primary, "shape"):
        return int(primary.shape[0])

    indexed_candidates: list[tuple[int, Any]] = []
    loose_candidates: list[tuple[str, Any]] = []
    for key, value in state_dict.items():
        if not (isinstance(key, str) and key.startswith("output.") and key.endswith(".weight")):
            continue
        if hasattr(value, "ndim") and int(value.ndim) >= 3 and hasattr(value, "shape"):
            parts = key.split(".")
            if len(parts) >= 3 and parts[1].isdigit():
                indexed_candidates.append((int(parts[1]), value))
            else:
                loose_candidates.append((key, value))

    if indexed_candidates:
        indexed_candidates.sort(key=lambda item: item[0])
        return int(indexed_candidates[-1][1].shape[0])

    if not loose_candidates:
        return None
    loose_candidates.sort(key=lambda item: item[0])
    return int(loose_candidates[-1][1].shape[0])


def _infer_nout_from_checkpoint(ckpt_path: str) -> int | None:
    payload = _safe_torch_load(ckpt_path, map_location=torch_CPU, weights_only=True)
    state_dict = _extract_state_dict(payload)
    return _infer_nout_from_state_dict(state_dict)


def _resolve_device(
    device: Optional[torch.device | str] = None,
    cpu: bool = False,
) -> torch.device:
    if device is not None:
        dev = torch.device(device) if not isinstance(device, torch.device) else device
        if dev.type == "cuda" and not torch.cuda.is_available():
            logger.warning("Requested CUDA but no CUDA device is available; falling back to CPU.")
            return torch_CPU
        return dev
    if cpu:
        return torch_CPU
    return torch_GPU if torch.cuda.is_available() else torch_CPU


def _resolve_amp_dtype(device: torch.device, fp16: bool) -> torch.dtype:
    if device.type != "cuda":
        return torch.float32
    if fp16:
        return torch.float16
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


class FiberposeModel:
    """Thin inference wrapper around ResUnet."""

    def __init__(
        self,
        ckpt_path: str,
        nbase: str = "1,32,64,128,256",
        nout: int | None = None,
        sz: int = 3,
        pool_kernel: int = 2,
        scale: int = 2,
        dilation: int = 1,
        residual_on: bool = True,
        style_on: bool = True,
        concat_style: bool = False,
        checkpointing: bool = False,
        dropout: bool = False,
        cpu: bool = False,
        amp: bool = True,
        fp16: bool = False,
        device: Optional[torch.device | str] = None,
    ):
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        self.device = _resolve_device(device=device, cpu=cpu)
        self.use_amp = (self.device.type == "cuda") and bool(amp)
        self.amp_dtype = _resolve_amp_dtype(self.device, fp16=fp16)

        self.nbase_str = nbase
        self.nbase = parse_nbase(nbase)

        inferred_nout = _infer_nout_from_checkpoint(ckpt_path)
        if nout is None:
            if inferred_nout is None:
                raise ValueError(
                    f"Could not infer nout from checkpoint '{ckpt_path}'. "
                    "Pass nout explicitly or use a compatible checkpoint."
                )
            resolved_nout = int(inferred_nout)
        else:
            resolved_nout = int(nout)
            if resolved_nout <= 0:
                raise ValueError(f"nout must be positive; got {resolved_nout}")
            if inferred_nout is not None and resolved_nout != int(inferred_nout):
                raise ValueError(
                    "Explicit nout does not match checkpoint output channels: "
                    f"nout={resolved_nout}, inferred={int(inferred_nout)} "
                    f"from '{ckpt_path}'."
                )
        self.nout = resolved_nout

        self.model = ResUnet(
            nbase=self.nbase,
            nout=self.nout,
            sz=sz,
            residual_on=residual_on,
            style_on=style_on,
            concatenation=concat_style,
            mkldnn=False,
            dim=2,
            checkpoint=checkpointing,
            dropout=dropout,
            kernel_size=pool_kernel,
            scale_factor=scale,
            dilation=dilation,
        ).to(self.device)

        try:
            self.model.load_model(ckpt_path, map_location=self.device, strict=True)
        except Exception:
            logger.exception("Failed to load checkpoint '%s'", ckpt_path)
            raise

        self.model.eval()
        logger.info("Resolved nout=%d for checkpoint '%s'", self.nout, ckpt_path)
        logger.info("Loaded checkpoint '%s' on device '%s'", ckpt_path, self.device)

    @torch.no_grad()
    def predict(self, img: np.ndarray, return_style: bool = False):
        arr = self._prep_input(img, self.nbase[0])
        x = torch.from_numpy(arr[None, ...]).to(self.device)

        if self.use_amp:
            with autocast(device_type="cuda", dtype=self.amp_dtype):
                y, style = self.model(x)
        else:
            y, style = self.model(x)

        y_np = y.detach().cpu().numpy()
        dt_channel = self._dt_channel_index()
        if y_np.shape[1] <= dt_channel:
            raise ValueError(
                "Predicted tensor channel count is incompatible with DT channel selection: "
                f"got shape={y_np.shape}, selected_channel={dt_channel}"
            )
        dt = y_np[0, dt_channel].astype(np.float32)

        if not return_style:
            return dt

        style_vec = style.detach().cpu().numpy()
        if style_vec.ndim > 1:
            style_vec = style_vec[0]
        return dt, np.asarray(style_vec, dtype=np.float32, copy=False)

    @staticmethod
    def _prep_input(x: np.ndarray, in_channels: int) -> np.ndarray:
        if x.ndim == 2:
            arr = x.astype(np.float32)[None, ...]

        elif x.ndim == 3:
            if x.shape[2] == in_channels:
                arr = np.transpose(x, (2, 0, 1)).astype(np.float32)

            elif in_channels == 1:
                if x.shape[2] >= 3:
                    r, g, b = x[..., 0], x[..., 1], x[..., 2]
                    gray = 0.2126 * r + 0.7152 * g + 0.0722 * b
                    arr = gray.astype(np.float32)[None, ...]
                else:
                    arr = x[..., 0:1].astype(np.float32).transpose(2, 0, 1)

            elif in_channels == 3:
                if x.shape[2] == 1:
                    arr = np.repeat(x, 3, axis=2).astype(np.float32).transpose(2, 0, 1)
                else:
                    arr = x[..., :3].astype(np.float32).transpose(2, 0, 1)

            else:
                raise ValueError(
                    f"Unsupported channel adaptation: in_channels={in_channels}, x.shape={x.shape}"
                )
        else:
            raise ValueError(f"Unsupported input shape: {x.shape}")

        vmin = float(np.min(arr))
        vmax = float(np.max(arr))
        if vmax > vmin:
            arr = (arr - vmin) / (vmax - vmin + 1e-12)
        else:
            arr = np.zeros_like(arr, dtype=np.float32)

        return arr.astype(np.float32, copy=False)

    def _dt_channel_index(self) -> int:
        if self.nout >= 3:
            return 2
        if self.nout == 1:
            return 0
        raise ValueError(
            f"Unsupported DT channel mapping for nout={self.nout}. "
            "Expected nout == 1 or nout >= 3."
        )
