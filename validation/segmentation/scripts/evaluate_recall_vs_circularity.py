#!/usr/bin/env python3
"""Stage 1: match all GT once per image and save per-instance data as TSV.

Keep evaluation_common.py beside this script. Shared mask decoding, circularity
and one-to-one matching are also used by evaluate_instance.py. Only original
GT zero/black pixels are ignored; no circularity filtering occurs here.
Stage 2 selects GT rows without rematching. TSV fields and formatting are
unchanged: match_iou is the assigned IoU (nan for unmatched GT), match_threshold
is the acceptance threshold, and floating-point values use 3 decimals.

Dependencies: numpy, scipy, pillow.
Example: python evaluate_recall_vs_circularity.py GT FP2
Output: FP2.tsv beside the FP2 folder.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from evaluation_common import (
    VERSION,
    circularity,
    existing_path,
    match_pair,
    paired_inputs,
    sample_id,
)


FIELDS = ("sample", "gt_instance", "circularity", "matched", "match_iou", "match_threshold")


def evaluate_pair(
    gt_path: Path, prediction_path: Path, match_iou: float = 0.5
) -> list[dict[str, object]]:
    """Match once on the full GT population; retain every GT, including misses."""
    gt, _prediction, _valid, overlaps, matches = match_pair(
        gt_path, prediction_path, match_iou
    )
    matched_ious = {
        ref_index: float(overlaps[ref_index, pred_index])
        for ref_index, pred_index in matches
    }
    return [
        {
            "sample": sample_id(gt_path),
            "gt_instance": index + 1,
            "circularity": circularity(instance),
            "matched": int(index in matched_ious),
            "match_iou": matched_ious.get(index, float("nan")),
            "match_threshold": match_iou,
        }
        for index, instance in enumerate(gt)
    ]


def write_results(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(
            {key: f"{value:.3f}" if isinstance(value, float) else value
             for key, value in row.items()}
            for row in rows
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Match all GT instances once and save circularity/matched per GT to TSV. "
            "GT: white/maximum=background, black/zero=ignore. Predictions: "
            "black/zero=background. RGB/16-bit: value then 8-connectivity. "
            "Only *_raw_proposals TIFF files: layer then 8-connectivity."
        )
    )
    parser.add_argument("--version", action="version", version=VERSION)
    parser.add_argument("ground_truth", help="GT mask file or folder")
    parser.add_argument("prediction", help="One method's mask file or folder")
    parser.add_argument(
        "-o", "--output", type=Path,
        help="TSV/text file; default: <prediction-name>.tsv beside predictions.",
    )
    parser.add_argument(
        "--match-iou", type=float, default=0.5,
        help="Minimum IoU for an accepted one-to-one match (default: 0.5).",
    )
    args = parser.parse_args()
    if not 0 <= args.match_iou <= 1:
        parser.error("--match-iou must be between 0 and 1")
    if args.output and args.output.suffix.lower() not in {"", ".tsv", ".txt"}:
        parser.error("--output must be a .tsv or .txt file (not the evaluation CSV)")
    return args


def main() -> None:
    try:
        args = parse_args()
        gt = existing_path(args.ground_truth)
        prediction = existing_path(args.prediction)
        pairs = paired_inputs(gt, prediction)
        name = prediction.stem if prediction.is_file() else prediction.name
        output = args.output or prediction.parent / f"{name}.tsv"
        if not output.suffix:
            output = output.with_suffix(".tsv")

        rows: list[dict[str, object]] = []
        for index, (gt_file, prediction_file) in enumerate(pairs, start=1):
            rows.extend(evaluate_pair(gt_file, prediction_file, args.match_iou))
            print(f"[{index}/{len(pairs)}] Matched {gt_file.name}", flush=True)
        write_results(rows, output)
        print(f"Wrote {len(rows)} GT instances to {output}")
    except Exception as exc:
        raise SystemExit(f"ERROR: {exc}") from exc


if __name__ == "__main__":
    main()
