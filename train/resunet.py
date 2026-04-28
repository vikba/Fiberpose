# resunet.py
"""ResUnet architecture and checkpoint utilities for Fiberpose."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = [
    "torch_GPU",
    "torch_CPU",
    "empty_cache",
    "dilation_list",
    "batchconv",
    "parse_nbase",
    "_safe_torch_load",
    "resdown",
    "convdown",
    "batchconvstyle",
    "resup",
    "convup",
    "downsample",
    "upsample",
    "make_style",
    "ResUnet",
    "build_resunet",
]


# -----------------------------------------------------------------------------
# Device helpers
# -----------------------------------------------------------------------------

torch_GPU = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch_CPU = torch.device("cpu")


def empty_cache() -> None:
    """Release cached CUDA memory if CUDA is available."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# -----------------------------------------------------------------------------
# Small utilities
# -----------------------------------------------------------------------------

def dilation_list(x: int | float, N: int) -> list[int]:
    """Return N integer dilation values linearly spaced from 1 to x."""
    if N <= 0:
        raise ValueError("N must be positive.")
    return np.round(np.linspace(1, x, N)).astype(int).tolist()


def parse_nbase(s: str | Sequence[int]) -> list[int]:
    """
    Parse encoder channel widths.

    Accepts either a comma-separated string like "1,32,64,128,256"
    or an iterable of integers.
    """
    if isinstance(s, str):
        vals = [int(x.strip()) for x in s.split(",") if x.strip()]
    else:
        vals = [int(x) for x in s]

    if len(vals) < 2:
        raise ValueError("nbase must have >= 2 entries.")
    if any(v <= 0 for v in vals):
        raise ValueError("All nbase entries must be positive.")
    return vals


def _safe_torch_load(path: str, map_location: Any, weights_only: bool = True) -> Any:
    """
    Load a PyTorch object while remaining compatible with older torch versions.
    """
    try:
        return torch.load(path, map_location=map_location, weights_only=weights_only)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _batchnorm_cls(dim: int) -> type[nn.Module]:
    if dim == 2:
        return nn.BatchNorm2d
    if dim == 3:
        return nn.BatchNorm3d
    raise ValueError(f"Unsupported dim={dim}. Expected 2 or 3.")


def _conv_cls(dim: int) -> type[nn.Module]:
    if dim == 2:
        return nn.Conv2d
    if dim == 3:
        return nn.Conv3d
    raise ValueError(f"Unsupported dim={dim}. Expected 2 or 3.")


def _maxpool_cls(dim: int) -> type[nn.Module]:
    if dim == 2:
        return nn.MaxPool2d
    if dim == 3:
        return nn.MaxPool3d
    raise ValueError(f"Unsupported dim={dim}. Expected 2 or 3.")


def _dropout_cls(dim: int) -> type[nn.Module]:
    if dim == 2:
        return nn.Dropout2d
    if dim == 3:
        return nn.Dropout3d
    raise ValueError(f"Unsupported dim={dim}. Expected 2 or 3.")


def batchconv(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    dim: int,
    dilation: int,
    relu: bool = True,
) -> nn.Sequential:
    """
    BatchNorm -> optional ReLU -> ConvNd with reflect padding.
    """
    BatchNorm = _batchnorm_cls(dim)
    ConvND = _conv_cls(dim)
    padding = ((kernel_size - 1) * dilation) // 2

    layers: list[nn.Module] = [BatchNorm(in_channels, eps=1e-5, momentum=0.05)]
    if relu:
        layers.append(nn.ReLU(inplace=True))
    layers.append(
        ConvND(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
            padding_mode="reflect",
        )
    )
    return nn.Sequential(*layers)


# -----------------------------------------------------------------------------
# Building blocks
# -----------------------------------------------------------------------------

class resdown(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, sz: int, dim: int, dilation: int):
        super().__init__()
        self.conv = nn.Sequential()
        self.proj = batchconv(in_channels, out_channels, 1, dim, 1, relu=False)
        for t in range(4):
            if t == 0:
                self.conv.add_module(
                    f"conv_{t}",
                    batchconv(in_channels, out_channels, sz, dim, dilation),
                )
            else:
                self.conv.add_module(
                    f"conv_{t}",
                    batchconv(out_channels, out_channels, sz, dim, dilation),
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x) + self.conv[1](self.conv[0](x))
        x = x + self.conv[3](self.conv[2](x))
        return x


class convdown(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, sz: int, dim: int, dilation: int):
        super().__init__()
        self.conv = nn.Sequential()
        for t in range(2):
            if t == 0:
                self.conv.add_module(
                    f"conv_{t}",
                    batchconv(in_channels, out_channels, sz, dim, dilation),
                )
            else:
                self.conv.add_module(
                    f"conv_{t}",
                    batchconv(out_channels, out_channels, sz, dim, dilation),
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv[0](x)
        x = self.conv[1](x)
        return x


class batchconvstyle(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        style_channels: int,
        sz: int,
        dim: int,
        dilation: int,
        concatenation: bool = False,
    ):
        super().__init__()
        self.conv = batchconv(in_channels, out_channels, sz, dim, dilation)
        if concatenation:
            self.full = nn.Linear(style_channels, out_channels * 2)
        else:
            self.full = nn.Linear(style_channels, out_channels)
        self.dim = dim

    def forward(
        self,
        style: torch.Tensor,
        x: torch.Tensor,
        mkldnn: bool = False,
        y: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if y is not None:
            x = x + y

        feat = self.full(style)
        for _ in range(self.dim):
            feat = feat.unsqueeze(-1)

        if mkldnn:
            x = x.to_dense()
            y_out = (x + feat).to_mkldnn()
        else:
            y_out = x + feat

        y_out = self.conv(y_out)
        return y_out


class resup(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        style_channels: int,
        parent: "ResUnet",
        dilation: int | None = None,
    ):
        super().__init__()
        sz = parent.sz
        concatenation = parent.concatenation
        dim = parent.dim
        dilation = parent.dilation if dilation is None else dilation

        self.conv = nn.Sequential()
        self.conv.add_module("conv_0", batchconv(in_channels, out_channels, sz, dim, dilation))
        self.conv.add_module(
            "conv_1",
            batchconvstyle(
                out_channels,
                out_channels,
                style_channels,
                sz,
                dim,
                dilation,
                concatenation=concatenation,
            ),
        )
        self.conv.add_module(
            "conv_2",
            batchconvstyle(out_channels, out_channels, style_channels, sz, dim, dilation),
        )
        self.conv.add_module(
            "conv_3",
            batchconvstyle(out_channels, out_channels, style_channels, sz, dim, dilation),
        )
        self.proj = batchconv(in_channels, out_channels, 1, dim, 1, relu=False)

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        style: torch.Tensor,
        mkldnn: bool = False,
    ) -> torch.Tensor:
        x = self.proj(x) + self.conv[1](style, self.conv[0](x) + y, mkldnn=mkldnn)
        x = x + self.conv[3](style, self.conv[2](style, x, mkldnn=mkldnn), mkldnn=mkldnn)
        return x


class convup(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        style_channels: int,
        parent: "ResUnet",
        dilation: int | None = None,
    ):
        super().__init__()
        sz = parent.sz
        concatenation = parent.concatenation
        dim = parent.dim
        dilation = parent.dilation if dilation is None else dilation

        self.conv = nn.Sequential()
        self.conv.add_module("conv_0", batchconv(in_channels, out_channels, sz, dim, dilation))
        self.conv.add_module(
            "conv_1",
            batchconvstyle(
                out_channels,
                out_channels,
                style_channels,
                sz,
                dim,
                dilation,
                concatenation=concatenation,
            ),
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        x = self.conv[1](style, self.conv[0](x) + y)
        return x


class downsample(nn.Module):
    def __init__(self, parent: "ResUnet"):
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

        MaxPool = _maxpool_cls(dim)
        self.maxpool = MaxPool(kernel_size=kernel_size, stride=scale_factor)

        N = len(nbase) - 1
        dilations = dilation_list(dilation, N)
        for n, dil in enumerate(dilations):
            if residual_on:
                self.down.add_module(f"res_down_{n}", resdown(nbase[n], nbase[n + 1], sz, dim, dil))
            else:
                self.down.add_module(f"conv_down_{n}", convdown(nbase[n], nbase[n + 1], sz, dim, dil))

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        xd: list[torch.Tensor] = []
        for n in range(len(self.down)):
            y = self.maxpool(xd[n - 1]) if n > 0 else x
            xd.append(self.down[n](y))
        return xd


class upsample(nn.Module):
    def __init__(self, parent: "ResUnet"):
        super().__init__()
        nbase = parent.nbaseup
        self.upsampling = nn.Upsample(scale_factor=parent.scale_factor, mode="nearest")
        self.up = nn.Sequential()
        self.checkpoint = parent.checkpoint

        N = len(nbase) - 1
        dilations = dilation_list(parent.dilation, N)
        for k, dil in enumerate(dilations):
            n = k + 1
            if parent.residual_on:
                self.up.add_module(
                    f"res_up_{n - 1}",
                    resup(nbase[n], nbase[n - 1], nbase[-1], parent, dilation=dil),
                )
            else:
                self.up.add_module(
                    f"conv_up_{n - 1}",
                    convup(nbase[n], nbase[n - 1], nbase[-1], parent, dilation=dil),
                )

    def forward(
        self,
        style: torch.Tensor,
        xd: list[torch.Tensor],
        mkldnn: bool = False,
    ) -> torch.Tensor:
        x = xd[-1]
        for n in range(len(self.up)):
            idx = -(n + 1)
            if n > 0:
                x = self.upsampling(x.to_dense()).to_mkldnn() if mkldnn else self.upsampling(x)

            block = self.up[idx]
            if "res_up" in block._get_name():
                x = block(x, xd[idx], style, mkldnn=mkldnn)
            else:
                x = block(x, xd[idx], style)
        return x


class make_style(nn.Module):
    def __init__(self, parent: "ResUnet"):
        super().__init__()
        self.dim = parent.dim
        self.flatten = nn.Flatten()
        self.avg_pool = F.avg_pool2d if self.dim == 2 else F.avg_pool3d

    def forward(self, x0: torch.Tensor) -> torch.Tensor:
        style = self.avg_pool(x0, kernel_size=tuple(x0.shape[-self.dim :]))
        style = self.flatten(style)
        style = style / torch.sum(style**2, axis=1, keepdim=True).sqrt()
        return style


# -----------------------------------------------------------------------------
# Main model
# -----------------------------------------------------------------------------

class ResUnet(nn.Module):
    """
    Residual U-Net with optional style modulation.

    Parameter names are intentionally kept unchanged for compatibility with the
    rest of the Fiberpose codebase and existing checkpoints/call sites.
    """

    def __init__(
        self,
        nbase,
        nout,
        sz,
        residual_on: bool = True,
        style_on: bool = True,
        concatenation: bool = False,
        mkldnn: bool = False,
        dim: int = 2,
        checkpoint: bool = False,
        dropout: bool = False,
        kernel_size: int = 2,
        scale_factor: int = 2,
        dilation: int = 1,
    ):
        super().__init__()

        if dim not in (2, 3):
            raise ValueError(f"Unsupported dim={dim}. Expected 2 or 3.")

        self.checkpoint = checkpoint
        self.kernel_size = kernel_size
        self.scale_factor = scale_factor
        self.dilation = dilation
        self.nbase = list(nbase)
        self.nout = int(nout)
        self.sz = int(sz)
        self.dim = dim
        self.residual_on = bool(residual_on)
        self.style_on = bool(style_on)
        self.concatenation = bool(concatenation)
        self.mkldnn = bool(mkldnn) if mkldnn is not None else False

        self.downsample = downsample(self)

        nbaseup = self.nbase[1:].copy()
        nbaseup.append(nbaseup[-1])
        self.nbaseup = nbaseup

        self.upsample = upsample(self)
        self.make_style = make_style(self)
        self.output = batchconv(nbaseup[0], self.nout, 1, self.dim, 1)

        self.do_dropout = bool(dropout)
        if self.do_dropout:
            Dropout = _dropout_cls(self.dim)
            self.dropout = Dropout(0.1)

    def forward(self, data: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.mkldnn:
            data = data.to_mkldnn()

        # Encoder
        T0 = self.downsample(data)

        # Style vector from bottleneck
        style = self.make_style(T0[-1].to_dense()) if self.mkldnn else self.make_style(T0[-1])
        style0 = style
        if not self.style_on:
            style = style * 0

        # Dropout at bottleneck only
        if self.do_dropout:
            if self.mkldnn:
                T0[-1] = self.dropout(T0[-1].to_dense()).to_mkldnn()
            else:
                T0[-1] = self.dropout(T0[-1])

        # Decoder
        T0 = self.upsample(style, T0, self.mkldnn)

        # Output head
        T0 = self.output(T0)

        if self.mkldnn:
            T0 = T0.to_dense()

        return T0, style0

    def save_model(self, filename: str) -> None:
        """Save model weights as a state dict."""
        torch.save(self.state_dict(), filename)

    def load_model(
        self,
        filename: str,
        map_location: Any = None,
        strict: bool = True,
        cpu: bool | None = None,
    ) -> "ResUnet":
        """
        Load model weights into the current instance.

        Parameters
        ----------
        filename
            Checkpoint path.
        map_location
            Passed to torch.load. Defaults to CPU if not specified.
        strict
            Passed to load_state_dict.
        cpu
            Backward-compatible alias. If provided, overrides map_location.
        """
        if cpu is not None:
            map_location = torch_CPU if cpu else torch_GPU
        if map_location is None:
            map_location = torch_CPU

        state_dict = _safe_torch_load(filename, map_location=map_location, weights_only=True)
        self.load_state_dict(state_dict, strict=strict)
        return self


# -----------------------------------------------------------------------------
# Convenience constructor
# -----------------------------------------------------------------------------

def build_resunet(
    nbase,
    nout: int = 3,
    sz: int = 3,
    residual_on: bool = True,
    style_on: bool = True,
    concatenation: bool = False,
    mkldnn: bool = False,
    dim: int = 2,
    checkpoint: bool = False,
    dropout: bool = False,
    kernel_size: int = 2,
    scale_factor: int = 2,
    dilation: int = 1,
) -> ResUnet:
    """
    Convenience wrapper around ResUnet with the same parameter names.
    """
    return ResUnet(
        nbase=nbase,
        nout=nout,
        sz=sz,
        residual_on=residual_on,
        style_on=style_on,
        concatenation=concatenation,
        mkldnn=mkldnn,
        dim=dim,
        checkpoint=checkpoint,
        dropout=dropout,
        kernel_size=kernel_size,
        scale_factor=scale_factor,
        dilation=dilation,
    )