#!/usr/bin/env python3
"""Summarize image metrics and optionally compare a reference against other methods.

Pair observations by sample ID, using independent images as experimental units.
The two-sided paired permutation test uses mean(reference - comparison); under
the null, method labels must be exchangeable within each image. Holm correction
covers all requested method/metric comparisons in this run. No best method is
selected from observed scores. IoU/NSD/HD95 compare each method's own recovered
instances, not necessarily the same GT objects. Missing metrics are omitted
pairwise and their counts reported; fewer than two pairs gives no test result.

Summary columns stay unchanged. Decimal rounding is applied only on output;
p-values use significant digits so small probabilities never print as zero.
Tests require numpy and scipy >= 1.15. Without --reference, only summarize.

Example:
    python collate_evaluation_metrics_VB.py --evaluation-dir validation \\
        --reference FP2 --compare Cellpose-SAM SAM2 --test-metrics recall
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from statistics import mean, stdev


METRICS = ("recall", "precision", "iou", "nsd", "hd95")
DEFAULT_VALIDATION_DIR = Path(__file__).resolve().parent.parent / "validation"


def evaluation_files(folder: Path, algorithms: list[str] | None) -> list[Path]:
    if algorithms:
        paths = [folder / f"{algorithm}-eval.csv" for algorithm in algorithms]
    else:
        paths = sorted(folder.glob("*-eval.csv"))
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing evaluation CSV: {missing[0]}")
    if not paths:
        raise FileNotFoundError(f"No *-eval.csv files found in {folder}")
    return paths


def load_metrics(path: Path) -> dict[str, dict[str, float]]:
    values: dict[str, dict[str, float]] = {}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        missing = [key for key in ("sample", *METRICS) if key not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{path} is missing: {', '.join(missing)}")
        for row in reader:
            sample = row["sample"].strip()
            if not sample or sample in values:
                raise ValueError(f"{path}: empty or duplicate sample ID {sample!r}")
            values[sample] = {metric: float(row[metric]) for metric in METRICS}
    return values


def summarize(algorithm: str, values: dict[str, dict[str, float]]) -> dict[str, object]:
    row: dict[str, object] = {"algorithm": algorithm}
    for metric in METRICS:
        samples = [sample[metric] for sample in values.values() if math.isfinite(sample[metric])]
        row[metric] = mean(samples) if samples else float("nan")
        row[f"{metric}_std"] = stdev(samples) if len(samples) > 1 else float("nan")
    return row


def compare_methods(data, reference: str, comparisons: list[str], metrics: list[str]):
    import numpy as np
    from scipy.stats import permutation_test

    rows = []
    baseline = data[reference]
    for comparison in comparisons:
        other = data[comparison]
        if baseline.keys() != other.keys():
            raise ValueError(
                f"{reference} and {comparison} have different sample IDs: "
                f"{len(baseline.keys() - other.keys())} missing in {comparison}, "
                f"{len(other.keys() - baseline.keys())} missing in {reference}. "
                "Evaluate the same images before testing."
            )
        for metric in metrics:
            differences = np.array([
                baseline[sample][metric] - other[sample][metric]
                for sample in sorted(baseline)
                if math.isfinite(baseline[sample][metric])
                and math.isfinite(other[sample][metric])
            ])
            n = len(differences)
            p_value = float("nan")
            test = "insufficient_pairs"
            if n >= 2:
                result = permutation_test(
                    (differences,), np.mean, permutation_type="samples",
                    alternative="two-sided", n_resamples=99_999,
                    vectorized=True, batch=1024, rng=0,
                )
                p_value = float(result.pvalue)
                test = "paired_permutation_exact" if n <= 16 else "paired_permutation_mc"
            rows.append({
                "reference": reference,
                "comparison": comparison,
                "metric": metric,
                "n_pairs": n,
                "n_excluded": len(baseline) - n,
                "mean_difference": float(differences.mean()) if n else float("nan"),
                "test": test,
                "p_value": p_value,
                "p_holm": float("nan"),
                "significant_0_05": "",
            })

    # Count every requested test in the family, including untestable comparisons.
    ordered = sorted(
        (row for row in rows if math.isfinite(row["p_value"])),
        key=lambda row: row["p_value"],
    )
    adjusted = 0.0
    for rank, row in enumerate(ordered):
        adjusted = max(adjusted, (len(rows) - rank) * row["p_value"])
        row["p_holm"] = min(1.0, adjusted)
        row["significant_0_05"] = row["p_holm"] < 0.05
    return rows


def write_csv(rows, path: Path, fieldnames: list[str], decimals: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: (f"{value:.3g}" if key in {"p_value", "p_holm"}
                      else f"{value:.{decimals}f}")
                if isinstance(value, float) else value
                for key, value in row.items()
            })


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write mean/SD across images; optionally run paired permutation tests."
    )
    parser.add_argument(
        "--evaluation-dir",
        "--validation-dir",
        dest="evaluation_dir",
        type=Path,
        default=DEFAULT_VALIDATION_DIR,
    )
    parser.add_argument(
        "--algorithms",
        nargs="+",
        help="Algorithm names; by default, discover all *-eval.csv files.",
    )
    parser.add_argument("-o", "--output", type=Path)
    parser.add_argument("--decimals", type=int, default=3, help="Decimal places for metrics (default: 3).")
    parser.add_argument("--reference", help="Reference algorithm name, e.g. FP2; enables tests.")
    parser.add_argument("--compare", nargs="+", help="Comparators; default: all other loaded methods.")
    parser.add_argument(
        "--test-metrics", nargs="+", choices=METRICS, default=["recall"],
        help="Metrics to test; default: recall. Holm covers all requested tests together.",
    )
    args = parser.parse_args()
    if args.decimals < 0:
        parser.error("--decimals must be non-negative")
    if args.compare and not args.reference:
        parser.error("--compare requires --reference")
    if len(set(args.test_metrics)) != len(args.test_metrics):
        parser.error("Do not repeat --test-metrics entries")

    paths = evaluation_files(args.evaluation_dir, args.algorithms)
    data = {path.name.removesuffix("-eval.csv"): load_metrics(path) for path in paths}
    tests = []
    if args.reference:
        if args.reference not in data:
            parser.error(f"Reference {args.reference!r} is not among the loaded algorithms")
        comparisons = args.compare or [name for name in data if name != args.reference]
        if not comparisons or any(name not in data or name == args.reference for name in comparisons):
            parser.error("--compare must name loaded algorithms other than --reference")
        if len(set(comparisons)) != len(comparisons):
            parser.error("Do not repeat --compare entries")
        tests = compare_methods(data, args.reference, comparisons, args.test_metrics)

    rows = [summarize(name, values) for name, values in data.items()]
    output = args.output or args.evaluation_dir / "evaluation.csv"
    fieldnames = [
        "algorithm",
        *METRICS,
        *(f"{metric}_std" for metric in METRICS),
    ]
    test_output = output.with_name(f"{output.stem}-tests.csv")
    if any(path.resolve() in {output.resolve(), test_output.resolve()} for path in paths):
        parser.error("Output must not overwrite an input evaluation CSV")
    write_csv(rows, output, fieldnames, args.decimals)
    print(f"Wrote {len(rows)} algorithm summary row(s) to {output}")
    if tests:
        write_csv(tests, test_output, list(tests[0]), args.decimals)
        print(f"Wrote {len(tests)} paired tests (one Holm family) to {test_output}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
