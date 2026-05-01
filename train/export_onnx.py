#!/usr/bin/env python3
"""
Export Fiberpose ResUNet checkpoint to ONNX.

Example:
    python export_onnx.py \
        --checkpoint checkpoints/fiber_2000_epoch.pt \
        --output models/fiberpose_resunet.onnx \
        --height 512 \
        --width 512 \
        --test-onnxruntime

Notes:
    - Input must be float32 tensor in NCHW format: [1, C, H, W].
    - The exported ONNX model returns the full network output [1, nout, H, W].
    - For current Fiberpose checkpoints:
        nout == 1  -> DT channel is 0
        nout >= 3  -> DT channel is 2
    - Padding/cropping should be handled outside ONNX, as FiberAnalyzer_2D already does.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from resunet import ResUnet, parse_nbase


def safe_torch_load(path: str, map_location: Any = "cpu", weights_only: bool = True) -> Any:
    """Load checkpoint with compatibility for older PyTorch versions."""
    try:
        return torch.load(path, map_location=map_location, weights_only=weights_only)
    except TypeError:
        return torch.load(path, map_location=map_location)


def extract_state_dict(payload: Any) -> dict[str, Any]:
    """Extract state_dict from either raw state_dict or checkpoint dictionary."""
    if isinstance(payload, dict):
        if isinstance(payload.get("model"), dict):
            return payload["model"]

        # Raw state_dict case.
        if all(isinstance(k, str) for k in payload.keys()):
            return payload

    raise ValueError("Checkpoint does not contain a recognizable state_dict.")


def infer_nout_from_state_dict(state_dict: dict[str, Any]) -> int:
    """
    Infer number of output channels from the output layer weights.
    Matches the logic used in FiberposeModel.
    """
    primary_key = "output.2.weight"
    primary = state_dict.get(primary_key)

    if hasattr(primary, "ndim") and int(primary.ndim) >= 3 and hasattr(primary, "shape"):
        return int(primary.shape[0])

    candidates: list[tuple[str, Any]] = []
    for key, value in state_dict.items():
        if key.startswith("output.") and key.endswith(".weight"):
            if hasattr(value, "ndim") and int(value.ndim) >= 3 and hasattr(value, "shape"):
                candidates.append((key, value))

    if not candidates:
        raise ValueError("Could not infer nout from checkpoint output weights.")

    candidates.sort(key=lambda item: item[0])
    return int(candidates[-1][1].shape[0])


def dt_channel_index(nout: int) -> int:
    """Fiberpose DT channel convention."""
    if nout >= 3:
        return 2
    if nout == 1:
        return 0
    raise ValueError(f"Unsupported nout={nout}. Expected nout == 1 or nout >= 3.")


class OutputOnlyWrapper(nn.Module):
    """
    ResUnet.forward returns (output, style).
    ONNX Runtime user inference usually needs only output.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y, _style = self.model(x)
        return y


def build_model(args: argparse.Namespace, state_dict: dict[str, Any], nout: int) -> nn.Module:
    """Build ResUnet with the same parameters used by FiberposeModel."""
    model = ResUnet(
        nbase=parse_nbase(args.nbase),
        nout=nout,
        sz=args.kernel_size,
        residual_on=not args.no_residual,
        style_on=not args.no_style,
        concatenation=args.concat_style,
        mkldnn=False,
        dim=2,
        checkpoint=False,
        dropout=False,
        kernel_size=args.pool_kernel,
        scale_factor=args.scale,
        dilation=args.dilation,
    )

    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    if missing:
        print("[warning] Missing keys:")
        for key in missing:
            print(f"  {key}")

    if unexpected:
        print("[warning] Unexpected keys:")
        for key in unexpected:
            print(f"  {key}")

    if missing or unexpected:
        print("[warning] Check architecture parameters: nbase, kernel size, style, dilation, etc.")

    model.eval()
    return model


def export_onnx(args: argparse.Namespace) -> None:
    checkpoint_path = Path(args.checkpoint)
    output_path = Path(args.output)

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = safe_torch_load(str(checkpoint_path), map_location="cpu", weights_only=True)
    state_dict = extract_state_dict(payload)

    nout = args.nout if args.nout is not None else infer_nout_from_state_dict(state_dict)
    dt_ch = dt_channel_index(nout)

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Output ONNX: {output_path}")
    print(f"nbase: {args.nbase}")
    print(f"nout: {nout}")
    print(f"DT channel convention: {dt_ch}")
    print(f"Dummy input: [1, {args.in_channels}, {args.height}, {args.width}]")

    model = build_model(args, state_dict, nout=nout)
    wrapped = OutputOnlyWrapper(model)

    dummy = torch.randn(
        1,
        args.in_channels,
        args.height,
        args.width,
        dtype=torch.float32,
    )

    dynamic_shapes = None
    if not args.static:
        from torch.export import Dim

        dynamic_shapes = (
            {
                0: Dim("batch", min=1),
                2: Dim("height", min=16),
                3: Dim("width", min=16),
            },
        )

    with torch.no_grad():
        torch.onnx.export(
            wrapped,
            (dummy,),
            str(output_path),
            export_params=True,
            opset_version=args.opset,
            do_constant_folding=False,
            input_names=["input"],
            output_names=["output"],
            dynamic_shapes=dynamic_shapes,
            dynamo=True,
        )

    print("ONNX export finished.")

    if args.check:
        import onnx

        onnx_model = onnx.load(str(output_path))
        onnx.checker.check_model(onnx_model)
        print("ONNX checker passed.")

    if args.test_onnxruntime:
        import onnxruntime as ort
        import numpy as np

        session = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
        smoke_shapes = [(args.height, args.width), (args.smoke_height, args.smoke_width)]
        for h, w in smoke_shapes:
            sample = np.random.randn(1, args.in_channels, h, w).astype(np.float32)
            y = session.run(None, {"input": sample})[0]
            print(f"ONNX Runtime output shape for {h}x{w}: {y.shape}")
            print(f"DT output shape for {h}x{w}: {y[:, dt_ch:dt_ch + 1].shape}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Fiberpose ResUNet checkpoint to ONNX.")

    parser.add_argument("--checkpoint", required=True, help="Path to PyTorch checkpoint .pt")
    parser.add_argument("--output", required=True, help="Path to output .onnx file")

    parser.add_argument("--nbase", default="1,32,64,128,256", help="Encoder channel widths")
    parser.add_argument("--nout", type=int, default=None, help="Output channels. If omitted, inferred from checkpoint.")
    parser.add_argument("--in-channels", type=int, default=1, help="Input channels")
    parser.add_argument("--height", type=int, default=512, help="Dummy input height")
    parser.add_argument("--width", type=int, default=512, help="Dummy input width")

    parser.add_argument("--kernel-size", type=int, default=3, help="Convolution kernel size")
    parser.add_argument("--pool-kernel", type=int, default=2, help="Pooling kernel size")
    parser.add_argument("--scale", type=int, default=2, help="Down/up scaling factor")
    parser.add_argument("--dilation", type=int, default=1, help="Maximum dilation")
    parser.add_argument("--concat-style", action="store_true", help="Use concatenated style mode")
    parser.add_argument("--no-style", action="store_true", help="Disable style modulation")
    parser.add_argument("--no-residual", action="store_true", help="Disable residual blocks")

    parser.add_argument("--opset", type=int, default=18, help="ONNX opset version")
    parser.add_argument(
        "--static",
        action="store_true",
        help="Export fixed H/W axes from --height/--width (dynamic is default).",
    )
    parser.add_argument("--smoke-height", type=int, default=240, help="Second ONNX Runtime smoke-test height")
    parser.add_argument("--smoke-width", type=int, default=320, help="Second ONNX Runtime smoke-test width")
    parser.add_argument("--check", action="store_true", help="Validate exported ONNX with onnx.checker")
    parser.add_argument(
        "--test-onnxruntime",
        action="store_true",
        help="Run ONNX Runtime smoke tests for two input sizes.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    export_onnx(parse_args())
