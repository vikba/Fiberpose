# fiberpose_style.py

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


# -----------------------
# Semantic color palette
# -----------------------

FIBERPOSE_COLORS = {
    "manual": "#222222",
    "fiberpose": "#2F5D7C",        # deep muted blue
    "fiberpose_strict": "#E69F00", # orange
    "diameterj": "#009E73",        # green
    "gordium": "#CC79A7",          # purple/pink
    "gift": "#D55E00",             # vermillion
    "orientationj": "#D28B3C",     # muted orange
    "directionality": "#777777",   # neutral grey
    "cellpose_sam": "#56B4E9",     # sky blue
    "sam2": "#6A3D9A",             # purple
    "otsu": "#666666",
    "srm": "#7F7F7F",
    "gt": "#000000",
}


# Dataset encoding is deliberately neutral so method colors remain semantic.
DATASET_COLORS = {
    "easy": "#DCE3E8",
    "hard": "#59636B",
}

DATASET_HATCHES = {
    "easy": "",
    "hard": "///",
}

DATASET_MARKERS = {
    "dataset1": "o",
    "dataset2": "s",
    "easy": "o",
    "challenging": "s",
}

DT_CMAP = "magma"
GRAY_CMAP = "gray"


# -----------------------
# Global style
# -----------------------

def set_fiberpose_style(context: str = "paper") -> None:
    """
    Apply publication-style Matplotlib defaults for Fiberpose figures.

    context:
        "paper"  - regular manuscript panels
        "small"  - dense multipanel figures
        "talk"   - larger labels for slides
    """
    if context == "small":
        font_size = 6.5
        label_size = 7
        tick_size = 6
        lw = 0.8
    elif context == "talk":
        font_size = 10
        label_size = 11
        tick_size = 9
        lw = 1.4
    else:
        font_size = 8
        label_size = 8
        tick_size = 7
        lw = 1.0

    mpl.rcParams.update({
        # Output
        "figure.dpi": 150,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "savefig.transparent": False,

        # Fonts
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
        "font.size": font_size,
        "axes.labelsize": label_size,
        "axes.titlesize": label_size,
        "xtick.labelsize": tick_size,
        "ytick.labelsize": tick_size,
        "legend.fontsize": tick_size,

        # Lines / markers
        "lines.linewidth": lw,
        "lines.markersize": 4,
        "patch.linewidth": lw,

        # Axes
        "axes.linewidth": 0.8,
        "axes.edgecolor": "#222222",
        "axes.labelcolor": "#222222",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,

        # Ticks
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.color": "#222222",
        "ytick.color": "#222222",

        # Legend
        "legend.frameon": False,
        "legend.handlelength": 1.5,
        "legend.borderaxespad": 0.2,

        # SVG/PDF text remains editable
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,

        # Images
        "image.cmap": GRAY_CMAP,
        "image.interpolation": "nearest",
    })


# -----------------------
# Figure helpers
# -----------------------

def cm_to_inch(*values: float):
    return tuple(v / 2.54 for v in values)


def nature_width(kind: str = "single") -> float:
    """
    Approximate Nature-style figure widths in inches.
    """
    if kind == "single":
        return 8.9 / 2.54   # 89 mm
    if kind == "double":
        return 18.0 / 2.54  # 180 mm
    if kind == "full":
        return 18.3 / 2.54
    raise ValueError(f"Unknown width kind: {kind}")


def despine(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", which="both", direction="out")
    return ax


def light_ygrid(ax):
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.6)
    ax.set_axisbelow(True)
    return ax


def panel_label(ax, label: str, x: float = -0.12, y: float = 1.08):
    ax.text(
        x, y, label,
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=9,
        fontweight="bold",
        color="#111111",
    )


def image_panel(ax, image, cmap="gray", vmin=None, vmax=None):
    ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    return ax


def add_scalebar(
    ax,
    length_px: float,
    label: str,
    loc: str = "lower right",
    color: str = "white",
    lw: float = 2.0,
    pad_frac: float = 0.06,
):
    """
    Simple image scalebar in pixel coordinates.
    """
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    width = abs(xlim[1] - xlim[0])
    height = abs(ylim[1] - ylim[0])

    pad_x = width * pad_frac
    pad_y = height * pad_frac

    if "right" in loc:
        x1 = max(xlim) - pad_x
        x0 = x1 - length_px
    else:
        x0 = min(xlim) + pad_x
        x1 = x0 + length_px

    if "lower" in loc:
        y = max(ylim) - pad_y
        text_y = y - height * 0.035
        va = "bottom"
    else:
        y = min(ylim) + pad_y
        text_y = y + height * 0.035
        va = "top"

    ax.plot([x0, x1], [y, y], color=color, lw=lw, solid_capstyle="butt")
    ax.text(
        (x0 + x1) / 2,
        text_y,
        label,
        color=color,
        ha="center",
        va=va,
        fontsize=7,
    )


# -----------------------
# Plot primitives
# -----------------------

def identity_line(ax, xlim=None, ylim=None, **kwargs):
    if xlim is None:
        xlim = ax.get_xlim()
    if ylim is None:
        ylim = ax.get_ylim()

    lo = min(xlim[0], ylim[0])
    hi = max(xlim[1], ylim[1])

    defaults = dict(color="#444444", lw=0.9, ls="--", zorder=0)
    defaults.update(kwargs)
    ax.plot([lo, hi], [lo, hi], **defaults)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)


def style_boxplot(bp, color="#222222", fill="#F2F2F2"):
    for box in bp["boxes"]:
        box.set(facecolor=fill, edgecolor=color, linewidth=0.9)
    for key in ["whiskers", "caps", "medians"]:
        for item in bp[key]:
            item.set(color=color, linewidth=0.9)
    if "fliers" in bp:
        for flier in bp["fliers"]:
            flier.set(
                marker="o",
                markerfacecolor="white",
                markeredgecolor=color,
                markersize=3,
                alpha=0.8,
            )


def save_figure(fig, path: str):
    """
    Save vector/raster cleanly. Use .svg for editable plots,
    .pdf for manuscript assembly, .png for quick preview.
    """
    fig.savefig(path)