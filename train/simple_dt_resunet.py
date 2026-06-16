"""Simple residual U-Net for DT + foreground prediction."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _valid_group_count(channels: int, max_groups: int = 8) -> int:
    """Return the largest group count <= max_groups that divides channels."""
    groups = min(max_groups, channels)
    while channels % groups != 0:
        groups -= 1
    return groups


class ConvGNReLU(nn.Module):
    """Conv -> GroupNorm -> ReLU."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, groups: int = 8):
        super().__init__()
        padding = kernel_size // 2
        groups = _valid_group_count(out_ch, groups)
        self.block = nn.Sequential(
            nn.Conv2d(
                in_ch,
                out_ch,
                kernel_size=kernel_size,
                padding=padding,
                padding_mode="reflect",
                bias=False,
            ),
            nn.GroupNorm(groups, out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualBlock(nn.Module):
    """Residual block with optional spatial dropout."""

    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        self.conv1 = ConvGNReLU(in_ch, out_ch)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                out_ch,
                out_ch,
                kernel_size=3,
                padding=1,
                padding_mode="reflect",
                bias=False,
            ),
            nn.GroupNorm(_valid_group_count(out_ch), out_ch),
        )
        self.shortcut = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False) if in_ch != out_ch else nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv1(x)
        y = self.dropout(y)
        y = self.conv2(y)
        return self.relu(y + self.shortcut(x))


class DownBlock(nn.Module):
    """MaxPool -> ResidualBlock."""

    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.res = ResidualBlock(in_ch, out_ch, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.res(self.pool(x))


class UpBlock(nn.Module):
    """Nearest upsample -> concatenate skip -> ResidualBlock."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.res = ResidualBlock(in_ch + skip_ch, out_ch, dropout=dropout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        diff_y = skip.size(2) - x.size(2)
        diff_x = skip.size(3) - x.size(3)
        if diff_x != 0 or diff_y != 0:
            x = F.pad(
                x,
                [
                    diff_x // 2,
                    diff_x - diff_x // 2,
                    diff_y // 2,
                    diff_y - diff_y // 2,
                ],
            )
        return self.res(torch.cat([skip, x], dim=1))


class SimpleDTResUNet(nn.Module):
    """
    Simplified residual U-Net for DT + foreground prediction.

    Output convention when out_channels=2:
        pred[:, 0:1] -> foreground logit
        pred[:, 1:2] -> DT prediction
    Output convention when out_channels=1:
        pred[:, 0:1] -> DT prediction only
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 2,
        base_channels: int = 16,
        dropout: float = 0.10,
        final_activation: str | None = None,
    ):
        super().__init__()
        if out_channels not in (1, 2):
            raise ValueError("SimpleDTResUNet supports out_channels=1 or out_channels=2.")

        c = base_channels
        self.out_channels = int(out_channels)
        self.final_activation = final_activation

        self.inc = ResidualBlock(in_channels, c, dropout=dropout)
        self.down1 = DownBlock(c, c * 2, dropout=dropout)
        self.down2 = DownBlock(c * 2, c * 4, dropout=dropout)
        self.down3 = DownBlock(c * 4, c * 8, dropout=dropout)
        self.bottleneck = DownBlock(c * 8, c * 16, dropout=dropout)

        self.up1 = UpBlock(c * 16, c * 8, c * 8, dropout=dropout)

        self.dt_up2 = UpBlock(c * 8, c * 4, c * 4, dropout=dropout)
        self.dt_up3 = UpBlock(c * 4, c * 2, c * 2, dropout=dropout)
        self.dt_up4 = UpBlock(c * 2, c, c, dropout=dropout)
        self.dt_head = nn.Conv2d(c, 1, kernel_size=1)

        if self.out_channels == 2:
            self.fg_up2 = UpBlock(c * 8, c * 4, c * 4, dropout=dropout)
            self.fg_up3 = UpBlock(c * 4, c * 2, c * 2, dropout=dropout)
            self.fg_up4 = UpBlock(c * 2, c, c, dropout=dropout)
            self.fg_head = nn.Conv2d(c, 1, kernel_size=1)

    def _apply_dt_activation(self, dt: torch.Tensor) -> torch.Tensor:
        if self.final_activation == "softplus":
            return F.softplus(dt)
        if self.final_activation is None:
            return dt
        raise ValueError(f"Unsupported final_activation={self.final_activation!r}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.bottleneck(x4)

        shared = self.up1(x5, x4)

        dt = self.dt_up2(shared, x3)
        dt = self.dt_up3(dt, x2)
        dt = self.dt_up4(dt, x1)
        dt = self.dt_head(dt)
        dt = self._apply_dt_activation(dt)

        if self.out_channels == 1:
            return dt

        fg = self.fg_up2(shared, x3)
        fg = self.fg_up3(fg, x2)
        fg = self.fg_up4(fg, x1)
        fg_logit = self.fg_head(fg)

        return torch.cat([fg_logit, dt], dim=1)
