"""Shared mask decoding, circularity and fixed one-to-one instance matching.

GT: zero/black is ignored, maximum/white is background. Other values/colors
are split by 8-connectivity. Predictions keep independent overlapping masks;
only *_raw_proposals TIFF files are decoded as binary layer stacks.
Matching precedes all GT shape selection and uses only original GT ignore pixels.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageSequence
from scipy.ndimage import label as connected_components
from scipy.optimize import linear_sum_assignment


IMAGE_SUFFIXES = {".png", ".tif", ".tiff"}
VALIDATION_DIR = Path(__file__).resolve().parent.parent / "validation"
VERSION = "2026.09.20-shared-matching-before-selection"


def existing_path(text: str) -> Path:
    path = Path(text)
    if path.exists():
        return path
    candidate = VALIDATION_DIR / path
    if candidate.exists():
        return candidate
    raise FileNotFoundError(text)


def read_mask(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB") if image.mode == "P" else image)
    if array.ndim == 3 and array.shape[2] >= 3:
        array = array[..., :3]
    elif array.ndim != 2:
        raise ValueError(f"Expected an RGB or single-channel mask: {path}")
    if not np.issubdtype(array.dtype, np.integer):
        raise ValueError(f"Expected an integer-valued mask, got {array.dtype}: {path}")
    return array


def pixels_equal(mask: np.ndarray, value: np.ndarray | np.generic) -> np.ndarray:
    return mask == value if mask.ndim == 2 else np.all(mask == value, axis=-1)


def stack_instances(instances: list[np.ndarray], shape: tuple[int, int]) -> np.ndarray:
    if not instances:
        return np.zeros((0, *shape), dtype=bool)
    return np.stack(instances).astype(bool, copy=False)


def values_to_instances(mask: np.ndarray, excluded: np.ndarray) -> np.ndarray:
    """Split first by value/RGB color and then by 8-connectivity."""
    pixels = mask[~excluded]
    values = np.unique(pixels, axis=0) if mask.ndim == 3 else np.unique(pixels)
    instances: list[np.ndarray] = []
    for value in values:
        components, count = connected_components(
            pixels_equal(mask, value) & ~excluded,
            structure=np.ones((3, 3), dtype=bool),
        )
        instances.extend(components == index for index in range(1, count + 1))
    return stack_instances(instances, mask.shape[:2])


def zero_pixels(mask: np.ndarray) -> np.ndarray:
    if mask.ndim == 2:
        return mask == 0
    return pixels_equal(mask, np.zeros(mask.shape[2], mask.dtype))


def maximum_pixels(mask: np.ndarray) -> np.ndarray:
    maximum = np.iinfo(mask.dtype).max
    value = np.full(mask.shape[2], maximum, mask.dtype) if mask.ndim == 3 else maximum
    return pixels_equal(mask, value)


def load_gt(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """GT: colors/values are instances, maximum is background, zero is ignore."""
    mask = read_mask(path)
    ignore = zero_pixels(mask)
    background = maximum_pixels(mask)
    return values_to_instances(mask, ignore | background), ~ignore


def load_raw_proposals(path: Path) -> np.ndarray:
    """Split every binary TIFF layer independently by 8-connectivity."""
    if path.suffix.lower() not in {".tif", ".tiff"}:
        raise ValueError(f"Raw proposals must be a TIFF stack: {path}")

    instances: list[np.ndarray] = []
    shape: tuple[int, int] | None = None
    with Image.open(path) as image:
        for layer_number, frame in enumerate(ImageSequence.Iterator(image), start=1):
            layer = np.asarray(frame.copy())
            if layer.ndim == 3 and layer.shape[2] >= 3:
                foreground = np.any(layer[..., :3] != 0, axis=-1)
            elif layer.ndim == 2:
                foreground = layer != 0
            else:
                raise ValueError(f"Unsupported layer {layer_number} in {path}")

            if shape is None:
                shape = foreground.shape
            elif foreground.shape != shape:
                raise ValueError(f"Inconsistent layer shape in {path}")

            components, count = connected_components(
                foreground, structure=np.ones((3, 3), dtype=bool)
            )
            instances.extend(components == index for index in range(1, count + 1))

    if shape is None:
        raise ValueError(f"Empty TIFF stack: {path}")
    return stack_instances(instances, shape)


def load_prediction(path: Path) -> np.ndarray:
    """Decode an ordinary mask or a *_raw_proposals binary TIFF stack."""
    if path.stem.lower().endswith("_raw_proposals"):
        return load_raw_proposals(path)
    mask = read_mask(path)
    return values_to_instances(mask, zero_pixels(mask))


def apply_ignore(instances: np.ndarray, valid: np.ndarray) -> np.ndarray:
    masked = instances & valid[None, ...]
    return masked[np.any(masked, axis=(1, 2))]


def crofton_perimeter(mask: np.ndarray) -> float:
    padded = np.pad(mask, 1)
    horizontal = np.count_nonzero(padded[:, 1:] != padded[:, :-1])
    vertical = np.count_nonzero(padded[1:, :] != padded[:-1, :])
    diagonal_1 = np.count_nonzero(padded[1:, 1:] != padded[:-1, :-1])
    diagonal_2 = np.count_nonzero(padded[1:, :-1] != padded[:-1, 1:])
    return float(
        np.pi
        / 8
        * (horizontal + vertical + (diagonal_1 + diagonal_2) / np.sqrt(2))
    )


def circularity(mask: np.ndarray) -> float:
    perimeter = crofton_perimeter(mask)
    return float(4 * np.pi * np.count_nonzero(mask) / perimeter**2)


def iou_matrix(reference: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    """Find overlaps via disjoint GT labels; keep prediction proposals independent."""
    n_ref, n_pred = len(reference), len(prediction)
    iou = np.zeros((n_ref, n_pred), dtype=float)
    if not n_ref or not n_pred:
        return iou

    labels = np.zeros(reference.shape[1:], dtype=np.int32)
    for ref_id, mask in enumerate(reference, start=1):
        labels[mask] = ref_id
    ref_area = np.bincount(labels.ravel(), minlength=n_ref + 1)[1:]

    for pred_id, mask in enumerate(prediction):
        intersections = np.bincount(labels[mask], minlength=n_ref + 1)[1:]
        candidates = np.flatnonzero(intersections)
        intersection = intersections[candidates]
        union = ref_area[candidates] + np.count_nonzero(mask) - intersection
        iou[candidates, pred_id] = intersection / union
    return iou


def match_instances(iou: np.ndarray, threshold: float) -> list[tuple[int, int]]:
    """Match eligible overlaps: maximize match count first and total IoU second."""
    eligible = (iou > 0) & (iou >= threshold)
    refs = np.flatnonzero(eligible.any(axis=1))
    preds = np.flatnonzero(eligible.any(axis=0))
    if not len(refs) or not len(preds):
        return []

    subset = np.ix_(refs, preds)
    overlaps = iou[subset]
    score = eligible[subset] * (min(overlaps.shape) + 1 + overlaps)
    rows, cols = linear_sum_assignment(score, maximize=True)
    return [
        (int(refs[row]), int(preds[col]))
        for row, col in zip(rows, cols)
        if score[row, col] > 0
    ]


def sample_id(path: Path) -> str:
    stem = re.sub(
        r"(?i)(?:[_-](?:raw[_-]proposals?|labels?|masks?|instances?|pred(?:iction)?|gt))+$",
        "",
        path.stem,
    )
    return re.sub(r"(?i)^(?:ground[_-]?truth|gt|pred(?:iction)?)[_-]", "", stem)


def image_files(folder: Path) -> list[Path]:
    return sorted(
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def files_by_sample(folder: Path) -> dict[str, Path]:
    indexed: dict[str, Path] = {}
    for path in image_files(folder):
        identifier = sample_id(path)
        if identifier in indexed:
            raise ValueError(f"Duplicate sample name '{identifier}' in {folder}")
        indexed[identifier] = path
    return indexed


def paired_inputs(gt: Path, prediction: Path) -> list[tuple[Path, Path]]:
    if gt.is_file() and prediction.is_file():
        return [(gt, prediction)]
    if not gt.is_dir() or not prediction.is_dir():
        raise ValueError("Inputs must be either two files or two folders.")

    gt_files = files_by_sample(gt)
    prediction_files = files_by_sample(prediction)
    identifiers = sorted(gt_files.keys() & prediction_files.keys())
    if not identifiers:
        raise ValueError(f"No matching sample names in {gt} and {prediction}.")
    return [(gt_files[name], prediction_files[name]) for name in identifiers]


def match_pair(
    gt_path: Path, prediction_path: Path, match_iou: float = 0.5
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[tuple[int, int]]]:
    """Return GT, predictions, valid pixels, IoUs and assignments before selection."""
    gt, valid = load_gt(gt_path)
    prediction = load_prediction(prediction_path)
    if gt.shape[1:] != prediction.shape[1:]:
        raise ValueError(
            f"Shape mismatch for {gt_path} and {prediction_path}: "
            f"{gt.shape[1:]} vs {prediction.shape[1:]}"
        )
    prediction = apply_ignore(prediction, valid)
    overlaps = iou_matrix(gt, prediction)
    matches = match_instances(overlaps, match_iou)
    return gt, prediction, valid, overlaps, matches
