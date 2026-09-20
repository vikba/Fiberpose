#!/usr/bin/env python3
"""Stage 2: plot cumulative recall vs equivalent GT aspect ratio from per-GT TSV files.

Aspect ratio is estimated from circularity by approximating each GT object as a rectangle.

For a rectangle with aspect ratio AR = a / b >= 1:
    C = pi * AR / (AR + 1)^2

Therefore:
    AR = [pi - 2C + sqrt(pi * (pi - 4C))] / (2C)

For C > pi/4 (more circular than a square), AR is set to 1.
For C <= 0, AR is treated as infinity.

For every cutoff t:
    recall = matched GT with AR >= t / all GT with AR >= t

Matching remains fixed. Undefined recall (no eligible GT) is plotted as a gap.
TSV match_iou stores each assigned pair's IoU; match_threshold stores the cutoff.
Circularity values are read from the TSV and converted to rectangular-equivalent AR.
Dependencies: numpy, matplotlib.

Example:
    python plot_recall_vs_aspect_ratio.py fiberpose.tsv cellpose.tsv sam2.tsv
"""

from __future__ import annotations

import argparse
import csv
from decimal import Decimal
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Save figures without opening a GUI or requiring a display.
import matplotlib.pyplot as plt
import numpy as np

from fiberpose_style import (
    FIBERPOSE_COLORS,
    despine,
    light_ygrid,
    nature_width,
    panel_label as add_panel_label,
    save_figure,
    set_fiberpose_style,
)


VERSION = "2026.09.18-equivalent-aspect-ratio-wilson-fiberpose-style"
# (sample, GT id) -> (GT circularity, matched 0/1)
InstanceResults = dict[tuple[str, int], tuple[float, int]]


def read_results(path: Path) -> tuple[InstanceResults, float]:
    results: InstanceResults = {}
    thresholds: set[float] = set()
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"sample", "gt_instance", "circularity", "matched", "match_iou", "match_threshold"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(
                f"{path}: expected match_iou and match_threshold columns; "
                "regenerate the TSV with the updated evaluate_recall_vs_circularity.py"
            )
        for line, row in enumerate(reader, start=2):
            key = (row["sample"], int(row["gt_instance"]))
            circularity = float(row["circularity"])
            matched = int(row["matched"])
            match_iou = float(row["match_iou"])
            match_threshold = float(row["match_threshold"])
            if (
                not key[0] or key[1] < 1 or key in results
                or not np.isfinite(circularity) or circularity < 0
                or matched not in (0, 1) or not 0 <= match_threshold <= 1
                or (matched == 1 and not match_threshold <= match_iou <= 1)
                or (matched == 0 and not np.isnan(match_iou))
            ):
                raise ValueError(f"{path}: invalid or duplicate GT row at line {line}")
            results[key] = (circularity, matched)
            thresholds.add(match_threshold)
    if not results:
        raise ValueError(f"{path}: no GT instances; recall is undefined")
    if len(thresholds) != 1:
        raise ValueError(f"{path}: inconsistent matching IoU thresholds")
    return results, thresholds.pop()


def validate_comparison(
    results: InstanceResults, baseline: InstanceResults, path: Path
) -> None:
    if results.keys() != baseline.keys():
        raise ValueError(f"{path}: different GT samples/instances; use the same GT set")
    if not all(
        np.isclose(results[key][0], baseline[key][0], rtol=0, atol=1e-12)
        for key in baseline
    ):
        raise ValueError(f"{path}: GT circularity differs from the first file")


def aspect_ratio_cutoffs(start: float, stop: float, step: float) -> np.ndarray:
    # Decimal arithmetic keeps user-specified cutoffs exact.
    first, last, spacing = (Decimal(str(value)) for value in (start, stop, step))
    count = int((last - first) // spacing) + 1
    values = [float(first + index * spacing) for index in range(count)]
    if values[-1] < stop:
        values.append(stop)
    return np.asarray(values)


def circularity_to_aspect_ratio(circularities: np.ndarray) -> np.ndarray:
    """Convert circularity to rectangular-equivalent aspect ratio (AR >= 1)."""
    c = np.asarray(circularities, dtype=float)
    ar = np.full(c.shape, np.inf, dtype=float)

    positive = c > 0
    # Maximum circularity attainable by a rectangle is pi/4 (a square).
    # Shapes more circular than that are assigned the limiting value AR = 1.
    c_eff = np.minimum(c[positive], np.pi / 4.0)
    discriminant = np.maximum(0.0, np.pi * (np.pi - 4.0 * c_eff))
    ar[positive] = (
        np.pi - 2.0 * c_eff + np.sqrt(discriminant)
    ) / (2.0 * c_eff)
    return ar



def wilson_interval(tp: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Return the Wilson 95% confidence interval for a binomial proportion."""
    if n <= 0:
        return float("nan"), float("nan")

    p = tp / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    half_width = (
        z
        * np.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n))
        / denom
    )
    return max(0.0, center - half_width), min(1.0, center + half_width)


def recall_curve(
    results: InstanceResults,
    cutoffs: np.ndarray,
) -> tuple[list[float], list[float], list[float]]:
    circularities, matched = np.asarray(list(results.values())).T
    aspect_ratios = circularity_to_aspect_ratio(circularities)

    recalls: list[float] = []
    lower: list[float] = []
    upper: list[float] = []

    for cutoff in cutoffs:
        included = aspect_ratios >= cutoff
        gt_count = int(included.sum())

        if gt_count == 0:
            recalls.append(float("nan"))
            lower.append(float("nan"))
            upper.append(float("nan"))
            continue

        tp = int(matched[included].sum())
        recall = tp / gt_count
        lo, hi = wilson_interval(tp, gt_count)

        recalls.append(recall)
        lower.append(lo)
        upper.append(hi)

    return recalls, lower, upper




def method_color(label: str) -> str:
    """Map benchmark labels to manuscript-wide semantic method colors."""
    token = "".join(ch for ch in label.lower() if ch.isalnum())
    if token.startswith("fp") or "fiberpose" in token:
        return FIBERPOSE_COLORS["fiberpose"]
    if "cellpose" in token:
        return FIBERPOSE_COLORS["cellpose_sam"]
    if "sam2" in token:
        return FIBERPOSE_COLORS["sam2"]
    return "#555555"

def plot_curves(
    cutoffs: np.ndarray,
    curves: list[tuple[str, list[float], list[float], list[float]]],
    output: Path,
    reference_cutoff: float,
    match_threshold: float,
    panel: str | None = None,
) -> None:
    """Plot recall curves using the shared Fiberpose manuscript style."""
    set_fiberpose_style("small")
    output.parent.mkdir(parents=True, exist_ok=True)

    width = nature_width("single")
    figure, axis = plt.subplots(
        figsize=(width, width * 0.72),
        constrained_layout=True,
    )

    for label, recalls, lower, upper in curves:
        recalls_arr = np.asarray(recalls, dtype=float)
        lower_arr = np.asarray(lower, dtype=float)
        upper_arr = np.asarray(upper, dtype=float)

        yerr = np.vstack(
            [
                recalls_arr - lower_arr,
                upper_arr - recalls_arr,
            ]
        )

        color = method_color(label)
        axis.errorbar(
            cutoffs,
            recalls_arr,
            yerr=yerr,
            color=color,
            marker="o",
            markerfacecolor=color,
            markeredgecolor=color,
            markersize=3.2,
            linewidth=1.0,
            elinewidth=0.7,
            capsize=2.0,
            capthick=0.7,
            label=label,
            zorder=2,
        )

    if cutoffs[0] <= reference_cutoff <= cutoffs[-1]:
        axis.axvline(
            reference_cutoff,
            color="#777777",
            linestyle="--",
            linewidth=0.8,
            zorder=0,
        )

    axis.set_xlabel("Equivalent aspect ratio threshold")
    axis.set_ylabel(f"Instance recall (IoU ≥ {match_threshold:g})")
    axis.set_ylim(0.0, 1.02)
    axis.set_yticks(np.linspace(0.0, 1.0, 5))

    if cutoffs[0] < cutoffs[-1]:
        axis.set_xlim(cutoffs[0], cutoffs[-1])

    despine(axis)
    light_ygrid(axis)

    axis.legend(
        title="Method",
        loc="lower left",
        frameon=False,
    )

    if panel:
        add_panel_label(axis, panel, x=-0.14, y=1.06)

    save_figure(figure, str(output))
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot recall vs rectangular-equivalent GT aspect ratio from one or more per-GT TSV files."
    )
    parser.add_argument("--version", action="version", version=VERSION)
    parser.add_argument("results", nargs="+", type=Path, help="One TSV/text file per method")
    parser.add_argument("--labels", nargs="+", help="Legend labels in input-file order")
    parser.add_argument("-o", "--output", type=Path, default=Path("recall-vs-aspect-ratio.png"))
    parser.add_argument("--min-ar", type=float, default=1.0)
    parser.add_argument("--max-ar", type=float, default=20.0)
    parser.add_argument("--step", type=float, default=1.0)
    parser.add_argument(
        "--reference-ar", type=float, default=3.0,
        help="Aspect-ratio cutoff marked by a dashed line (default: 3).",
    )
    parser.add_argument(
        "--panel-label", type=str, default=None,
        help="Optional manuscript panel label, e.g. C or F.",
    )
    args = parser.parse_args()
    if not (
        np.isfinite(args.min_ar)
        and np.isfinite(args.max_ar)
        and 1 <= args.min_ar <= args.max_ar
    ):
        parser.error("aspect-ratio limits must be finite and satisfy 1 <= min <= max")
    if not np.isfinite(args.step) or args.step <= 0:
        parser.error("--step must be finite and positive")
    if not np.isfinite(args.reference_ar) or args.reference_ar < 1:
        parser.error("--reference-ar must be finite and >= 1")
    if args.labels and len(args.labels) != len(args.results):
        parser.error("--labels must contain one label per result file")
    if not args.output.suffix:
        args.output = args.output.with_suffix(".png")
    if args.output.suffix.lower() not in {".png", ".svg", ".pdf"}:
        parser.error("--output must be .png, .svg or .pdf")
    return args


def main() -> None:
    try:
        args = parse_args()
        cutoffs = aspect_ratio_cutoffs(args.min_ar, args.max_ar, args.step)
        labels = args.labels or [path.stem.removesuffix("-recall") for path in args.results]
        curves: list[tuple[str, list[float], list[float], list[float]]] = []
        baseline, match_threshold = read_results(args.results[0])
        for index, (path, label) in enumerate(zip(args.results, labels)):
            results, threshold = (baseline, match_threshold) if index == 0 else read_results(path)
            if not np.isclose(threshold, match_threshold, rtol=0, atol=1e-12):
                raise ValueError(f"{path}: matching IoU threshold differs from the first file")
            validate_comparison(results, baseline, path)
            recalls, lower, upper = recall_curve(results, cutoffs)
            curves.append((label, recalls, lower, upper))
        plot_curves(cutoffs, curves, args.output, args.reference_ar, match_threshold, args.panel_label)
        print(f"Wrote recall plot to {args.output}")
    except Exception as exc:
        raise SystemExit(f"ERROR: {exc}") from exc


if __name__ == "__main__":
    main()
