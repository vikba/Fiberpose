#!/usr/bin/env python3
"""Match all GT once, select by circularity, then evaluate fiber instances.

Use evaluation_common.py beside this script. RGB/16-bit masks and SAM2
*_raw_proposals stacks follow the same decoding and matching as the recall TSV.
Only the original GT ignore domain is removed. Shape selection never rematches
or changes the masks used for IoU, NSD and HD95.

Precision excludes predictions matched to excluded GT; all globally unmatched
predictions remain FP. prediction_count is the evaluated denominator (TP + FP),
not the raw number of loaded predictions. CSV columns are preserved; floats
are written to 3 decimals, with all calculations performed before rounding.

Dependencies: numpy, scipy, pillow.
Example: python evaluate_instance.py GT FP2 --max-circularity 0.6
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion, distance_transform_edt

from evaluation_common import (
    VALIDATION_DIR,
    VERSION,
    circularity,
    existing_path,
    match_pair,
    paired_inputs,
    sample_id,
)


def surface(mask: np.ndarray) -> np.ndarray:
    return mask & ~binary_erosion(mask, structure=np.ones((3, 3)), border_value=0)


def surface_metrics(
    reference: np.ndarray,
    prediction: np.ndarray,
    evaluable_surface: np.ndarray,
    tolerance: float,
) -> tuple[float, float]:
    ref_surface = surface(reference) & evaluable_surface
    pred_surface = surface(prediction) & evaluable_surface
    if not ref_surface.any() or not pred_surface.any():
        return float("nan"), float("nan")

    ref_to_pred = distance_transform_edt(~pred_surface)[ref_surface]
    pred_to_ref = distance_transform_edt(~ref_surface)[pred_surface]
    nsd = (
        np.count_nonzero(ref_to_pred <= tolerance)
        + np.count_nonzero(pred_to_ref <= tolerance)
    ) / (len(ref_to_pred) + len(pred_to_ref))
    hd95 = max(np.percentile(ref_to_pred, 95), np.percentile(pred_to_ref, 95))
    return float(nsd), float(hd95)


def mean_or_nan(values: list[float]) -> float:
    finite = [value for value in values if np.isfinite(value)]
    return float(np.mean(finite)) if finite else float("nan")


def evaluate_pair(
    gt_path: Path,
    prediction_path: Path,
    max_circularity: float = 0.6,
    match_iou: float = 0.5,
    surface_tolerance: float = 2.0,
) -> dict[str, object]:
    gt, prediction, valid, overlaps, matches = match_pair(
        gt_path, prediction_path, match_iou
    )
    gt_total_count = len(gt)
    selected = np.array(
        [circularity(instance) < max_circularity for instance in gt], dtype=bool
    )
    selected_matches = [(g, p) for g, p in matches if selected[g]]

    gt_count = int(selected.sum())
    tp = len(selected_matches)
    # Matches to excluded GT leave the denominator; all unmatched predictions are FP.
    fp = len(prediction) - len(matches)
    prediction_count = tp + fp
    fn = gt_count - tp
    precision = tp / prediction_count if prediction_count else float("nan")
    recall = tp / gt_count if gt_count else float("nan")

    # Ignore boundaries created artificially where an object meets an ignored area.
    evaluable_surface = ~binary_dilation(~valid, structure=np.ones((3, 3)))
    ious, nsds, hd95s = [], [], []
    for ref_index, pred_index in selected_matches:
        ious.append(float(overlaps[ref_index, pred_index]))
        nsd, hd95 = surface_metrics(
            gt[ref_index],
            prediction[pred_index],
            evaluable_surface,
            surface_tolerance,
        )
        nsds.append(nsd)
        hd95s.append(hd95)

    return {
        "sample": sample_id(gt_path),
        "reference": str(gt_path),
        "prediction": str(prediction_path),
        "gt_total_count": gt_total_count,
        "gt_count": gt_count,
        "gt_excluded_count": gt_total_count - gt_count,
        "prediction_count": prediction_count,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "recall": recall,
        "precision": precision,
        "iou": mean_or_nan(ious),
        "nsd": mean_or_nan(nsds),
        "hd95": mean_or_nan(hd95s),
    }


def default_output_path(prediction: Path) -> Path:
    name = prediction.stem if prediction.is_file() else prediction.name
    return VALIDATION_DIR / f"{name}-eval.csv"


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(
            {key: f"{value:.3f}" if isinstance(value, float) else value
             for key, value in row.items()}
            for row in rows
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Match all GT once, then select by circularity and report metrics. GT: "
            "values/colors=instances, maximum/white=background, zero/black=ignore. "
            "Ordinary predictions: values/colors=instances, zero/black=background. "
            "A *_raw_proposals TIFF is split first by layer and then by "
            "8-connectivity."
        )
    )
    parser.add_argument("--version", action="version", version=VERSION)
    parser.add_argument("ground_truth", help="GT mask file or folder")
    parser.add_argument("prediction", help="Prediction mask file or folder")
    parser.add_argument("-o", "--output", type=Path)
    parser.add_argument(
        "--max-circularity",
        type=float,
        default=0.5,
        help="After matching all GT, report instances with circularity strictly below this value.",
    )
    parser.add_argument(
        "--match-iou",
        type=float,
        default=0.5,
        help="Minimum IoU for an accepted one-to-one match.",
    )
    parser.add_argument(
        "--surface-tolerance",
        type=float,
        default=2.0,
        help="Boundary tolerance in pixels for normalized surface Dice.",
    )
    args = parser.parse_args()
    if not 0 <= args.max_circularity <= 1:
        parser.error("--max-circularity must be between 0 and 1")
    if not 0 <= args.match_iou <= 1:
        parser.error("--match-iou must be between 0 and 1")
    if args.surface_tolerance < 0:
        parser.error("--surface-tolerance must be non-negative")
    return args


def main() -> None:
    try:
        args = parse_args()
        gt = existing_path(args.ground_truth)
        prediction = existing_path(args.prediction)
        rows = [
            evaluate_pair(
                gt_path,
                prediction_path,
                args.max_circularity,
                args.match_iou,
                args.surface_tolerance,
            )
            for gt_path, prediction_path in paired_inputs(gt, prediction)
        ]
        output = args.output or default_output_path(prediction)
        if args.output and not output.is_absolute() and output.parent == Path("."):
            output = VALIDATION_DIR / output
        write_csv(rows, output)
        print(f"Wrote {len(rows)} comparison(s) to {output}")
    except Exception as exc:
        raise SystemExit(f"ERROR: {exc}") from exc


if __name__ == "__main__":
    main()
