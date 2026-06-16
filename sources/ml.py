"""ONNX Runtime inference wrapper for Fiberpose models."""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

import numpy as np
import onnxruntime as ort

logger = logging.getLogger(__name__)


def get_available_providers() -> list[str]:
    """Return providers reported by the local ONNX Runtime build."""
    try:
        return list(ort.get_available_providers())
    except Exception:
        logger.exception("Failed to query ONNX Runtime providers")
        return ["CPUExecutionProvider"]


def onnx_cuda_available() -> bool:
    """Return whether ONNX Runtime exposes CUDAExecutionProvider."""
    return "CUDAExecutionProvider" in get_available_providers()


def _resolve_providers(device: Optional[str]) -> list[str]:
    requested = str(device or "cpu").strip().lower()
    available = get_available_providers()

    if requested == "cuda":
        if "CUDAExecutionProvider" in available:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        logger.warning("Requested CUDA but CUDAExecutionProvider is unavailable; falling back to CPU.")
    return ["CPUExecutionProvider"]


class OnnxFiberposeModel:
    """ONNX-only inference wrapper used by runtime analysis."""

    def __init__(
        self,
        model_path: str,
        *,
        device: str = "cpu",
        nout: int | None = None,
        nbase: str = "1,32,64,128,256",
        stride: int | None = None,
        **_: Any,
    ):
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")

        providers = _resolve_providers(device)
        self.device = "cuda" if providers and providers[0] == "CUDAExecutionProvider" else "cpu"
        self.providers = providers
        self.session = ort.InferenceSession(model_path, providers=providers)

        inputs = self.session.get_inputs()
        if not inputs:
            raise ValueError(f"ONNX model has no inputs: {model_path}")
        self.input_name = str(inputs[0].name)
        self.input_channels = self._infer_input_channels(inputs[0].shape)

        self.output_name = str(self.session.get_outputs()[0].name)
        self.nout = int(nout) if nout is not None else None

        self.nbase_str = nbase
        self.stride = int(stride) if stride is not None else self._infer_stride_from_nbase(nbase)

        logger.info(
            "Loaded ONNX model '%s' with providers=%s input='%s' output='%s' stride=%d",
            model_path,
            providers,
            self.input_name,
            self.output_name,
            self.stride,
        )

    @staticmethod
    def _infer_input_channels(shape: Any) -> int:
        if isinstance(shape, (list, tuple)) and len(shape) >= 2:
            ch = shape[1]
            if isinstance(ch, int) and ch > 0:
                return int(ch)
        return 1

    @staticmethod
    def _infer_stride_from_nbase(nbase: str) -> int:
        try:
            levels = [int(p.strip()) for p in str(nbase).split(",") if p.strip()]
            downs = max(0, len(levels) - 1)
            return int(2 ** downs)
        except Exception:
            return 16

    def predict(self, img: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        """Return `(dt, foreground_prob)`, with foreground only for `nout == 2`."""
        arr = self._prep_input(img, self.input_channels)
        x = arr[None, ...].astype(np.float32, copy=False)
        outputs = self.session.run([self.output_name], {self.input_name: x})
        if not outputs:
            raise RuntimeError("ONNX Runtime returned no outputs.")
        y = np.asarray(outputs[0])
        if y.ndim != 4:
            raise ValueError(f"Unexpected ONNX output shape: {y.shape}")

        resolved_nout = int(self.nout) if self.nout is not None else int(y.shape[1])
        if resolved_nout <= 0:
            raise ValueError(f"Unsupported output channel count: nout={resolved_nout}")
        dt_channel = 1 if resolved_nout == 2 else resolved_nout - 1
        if y.shape[1] <= dt_channel:
            raise ValueError(
                "Predicted tensor channel count is incompatible with DT channel selection: "
                f"got shape={y.shape}, selected_channel={dt_channel}"
            )

        foreground_prob = None
        if resolved_nout == 2:
            fg_logit = np.clip(np.asarray(y[0, 0], dtype=np.float32), -60.0, 60.0)
            foreground_prob = (1.0 / (1.0 + np.exp(-fg_logit))).astype(np.float32, copy=False)

        return np.asarray(y[0, dt_channel], dtype=np.float32), foreground_prob

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
