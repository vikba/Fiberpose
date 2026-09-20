#!/usr/bin/env python3
"""Resolve reused labels/colors by splitting disconnected components.

The ``ground-truth`` mode preserves exact black (ignore) and white
(verified-background) pixels while assigning every 8-connected colored
component a distinct, human-readable RGB value.

The ``sam2`` mode splits every nonzero SAM2 label into 8-connected components,
discards components smaller than 10 pixels, writes a new integer label map,
and retains the existing manual-review exports.

With no mode argument, both pipelines are run using their validation-folder
defaults.  Explicit modes are ``ground-truth``, ``sam2``, and ``all``.
"""

from __future__ import annotations

import argparse
import colorsys
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage


VALIDATION_DIR = Path(__file__).resolve().parent
EVALUATION_DIR = VALIDATION_DIR.parent / "evaluation"
if str(EVALUATION_DIR) not in sys.path:
    sys.path.insert(0, str(EVALUATION_DIR))

from export_sam_disconnected_inspection import _display_rgb, infer_overlay_color


BLACK = np.array((0, 0, 0), dtype=np.uint8)
WHITE = np.array((255, 255, 255), dtype=np.uint8)
EIGHT_CONNECTED = np.ones((3, 3), dtype=np.uint8)
DEFAULT_MIN_COMPONENT_AREA = 10
DEFAULT_REVIEW_LARGE_AREA = 1_000
OVERLAY_ALPHA = 0.45


@dataclass(frozen=True)
class GroundTruthComponent:
    source_rgb: tuple[int, int, int]
    source_component: int
    source_color_component_count: int
    mask: np.ndarray
    area_pixels: int
    bbox: tuple[int, int, int, int]


@dataclass(frozen=True)
class SamComponent:
    sample: str
    source_label: int
    source_component: int
    new_label: int | None
    area_pixels: int

    @property
    def retained(self) -> bool:
        return self.new_label is not None


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in (
        Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ):
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def _write_csv(
    path: Path,
    fieldnames: list[str],
    rows: list[dict[str, object]],
    *,
    allow_empty: bool = False,
) -> None:
    if not rows and not allow_empty:
        raise ValueError(f"Cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Ground-truth RGB processing


def _srgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """Convert an ``(..., 3)`` uint8/sRGB array to CIE L*a*b* (D65)."""

    srgb = rgb.astype(np.float64) / 255.0
    linear = np.where(
        srgb <= 0.04045,
        srgb / 12.92,
        ((srgb + 0.055) / 1.055) ** 2.4,
    )
    xyz = linear @ np.array(
        [
            [0.4124564, 0.2126729, 0.0193339],
            [0.3575761, 0.7151522, 0.1191920],
            [0.1804375, 0.0721750, 0.9503041],
        ]
    )
    xyz /= np.array((0.95047, 1.0, 1.08883))
    epsilon = 216 / 24389
    kappa = 24389 / 27
    transformed = np.where(
        xyz > epsilon,
        np.cbrt(xyz),
        (kappa * xyz + 16) / 116,
    )
    return np.stack(
        (
            116 * transformed[..., 1] - 16,
            500 * (transformed[..., 0] - transformed[..., 1]),
            200 * (transformed[..., 1] - transformed[..., 2]),
        ),
        axis=-1,
    )


def perceptually_spaced_palette(count: int) -> np.ndarray:
    """Select distinct, non-black/non-white colors by farthest-point sampling."""

    levels = np.arange(0, 256, 17, dtype=np.uint8)
    candidates = np.stack(
        np.meshgrid(levels, levels, levels, indexing="ij"), axis=-1
    ).reshape(-1, 3)
    lab = _srgb_to_lab(candidates)
    chroma = np.linalg.norm(lab[:, 1:], axis=1)
    readable = (lab[:, 0] >= 30) & (lab[:, 0] <= 88) & (chroma >= 28)
    candidates = candidates[readable]
    lab = lab[readable]
    if count > len(candidates):
        raise ValueError(f"Requested {count} colors from {len(candidates)} candidates")

    seed = np.array((238, 34, 74), dtype=np.uint8)
    seed_lab = _srgb_to_lab(seed)
    first = int(np.argmin(np.sum((lab - seed_lab) ** 2, axis=1)))
    selected = [first]
    minimum_squared_distance = np.sum((lab - lab[first]) ** 2, axis=1)
    minimum_squared_distance[first] = -1
    for _ in range(1, count):
        index = int(np.argmax(minimum_squared_distance))
        selected.append(index)
        distance = np.sum((lab - lab[index]) ** 2, axis=1)
        minimum_squared_distance = np.minimum(minimum_squared_distance, distance)
        minimum_squared_distance[selected] = -1
    return candidates[selected]


def split_ground_truth_components(rgb: np.ndarray) -> list[GroundTruthComponent]:
    """Split each exact foreground RGB value into 8-connected components."""

    colors = np.unique(rgb.reshape(-1, 3), axis=0)
    foreground_colors = [
        color
        for color in colors
        if not np.array_equal(color, BLACK) and not np.array_equal(color, WHITE)
    ]
    components: list[GroundTruthComponent] = []
    for color in foreground_colors:
        connected, count = ndimage.label(
            np.all(rgb == color, axis=2), structure=EIGHT_CONNECTED
        )
        objects = ndimage.find_objects(connected)
        for component_number in range(1, count + 1):
            component_mask = connected == component_number
            slices = objects[component_number - 1]
            if slices is None:
                continue
            y_slice, x_slice = slices
            components.append(
                GroundTruthComponent(
                    source_rgb=tuple(int(value) for value in color),
                    source_component=component_number,
                    source_color_component_count=count,
                    mask=component_mask,
                    area_pixels=int(np.count_nonzero(component_mask)),
                    bbox=(
                        int(x_slice.start),
                        int(y_slice.start),
                        int(x_slice.stop - 1),
                        int(y_slice.stop - 1),
                    ),
                )
            )
    return sorted(components, key=lambda item: (item.bbox[1], item.bbox[0], item.source_rgb))


def repair_ground_truth_image(
    source: Path, destination: Path
) -> list[dict[str, object]]:
    with Image.open(source) as image:
        array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] < 3:
        raise ValueError(f"Expected RGB(A) ground truth, got {array.shape}: {source}")
    rgb = array[..., :3]
    components = split_ground_truth_components(rgb)
    palette = perceptually_spaced_palette(len(components))
    component_areas = np.array([component.area_pixels for component in components])
    first_quartile, third_quartile = np.percentile(component_areas, (25, 75))
    large_component_cutoff = third_quartile + 1.5 * (third_quartile - first_quartile)

    repaired = rgb.copy()
    rows: list[dict[str, object]] = []
    for instance_id, (component, color) in enumerate(
        zip(components, palette, strict=True), start=1
    ):
        repaired[component.mask] = color
        rows.append(
            {
                "image": source.name,
                "instance_id": instance_id,
                "output_rgb": "-".join(str(int(value)) for value in color),
                "source_rgb": "-".join(str(value) for value in component.source_rgb),
                "source_color_component": component.source_component,
                "source_color_component_count": component.source_color_component_count,
                "area_pixels": component.area_pixels,
                "large_area_review_flag": component.area_pixels > large_component_cutoff,
                "bbox_x_min": component.bbox[0],
                "bbox_y_min": component.bbox[1],
                "bbox_x_max": component.bbox[2],
                "bbox_y_max": component.bbox[3],
            }
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(repaired, mode="RGB").save(destination)
    original_black = np.all(rgb == BLACK, axis=2)
    original_white = np.all(rgb == WHITE, axis=2)
    if not np.array_equal(np.all(repaired == BLACK, axis=2), original_black):
        raise AssertionError(f"Black ignore region changed in {source}")
    if not np.array_equal(np.all(repaired == WHITE, axis=2), original_white):
        raise AssertionError(f"White background region changed in {source}")
    output_colors = np.unique(repaired.reshape(-1, 3), axis=0)
    output_foreground_count = sum(
        not np.array_equal(color, BLACK) and not np.array_equal(color, WHITE)
        for color in output_colors
    )
    if output_foreground_count != len(components):
        raise AssertionError(
            f"Expected {len(components)} unique instance colors in {destination}, "
            f"found {output_foreground_count}"
        )
    return rows


def process_ground_truth(source_dir: Path, output_dir: Path) -> None:
    sources = sorted(source_dir.glob("*.png"))
    if not sources:
        raise FileNotFoundError(f"No PNG ground-truth masks found in {source_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for source in sources:
        rows = repair_ground_truth_image(source, output_dir / source.name)
        all_rows.extend(rows)
        areas = np.array([int(row["area_pixels"]) for row in rows])
        summary_rows.append(
            {
                "image": source.name,
                "recovered_instance_count": len(rows),
                "minimum_area_pixels": int(areas.min()),
                "median_area_pixels": float(np.median(areas)),
                "maximum_area_pixels": int(areas.max()),
            }
        )
        print(f"GT {source.name}: recovered {len(rows)} instances")

    _write_csv(
        output_dir / "component-report.csv",
        [
            "image",
            "instance_id",
            "output_rgb",
            "source_rgb",
            "source_color_component",
            "source_color_component_count",
            "area_pixels",
            "large_area_review_flag",
            "bbox_x_min",
            "bbox_y_min",
            "bbox_x_max",
            "bbox_y_max",
        ],
        all_rows,
    )
    _write_csv(
        output_dir / "summary.csv",
        [
            "image",
            "recovered_instance_count",
            "minimum_area_pixels",
            "median_area_pixels",
            "maximum_area_pixels",
        ],
        summary_rows,
    )


# ---------------------------------------------------------------------------
# SAM2 integer-label processing and inspection output


def _sample_name(path: Path) -> str:
    suffix = "_instances"
    return path.stem[: -len(suffix)] if path.stem.endswith(suffix) else path.stem


def _load_label_map(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        array = np.asarray(image)
    if array.ndim != 2:
        raise ValueError(f"Expected a 2-D SAM instance map, got {array.shape}: {path}")
    return array


def split_sam_label_map(
    labels: np.ndarray,
    sample: str,
    min_component_area: int,
) -> tuple[np.ndarray, list[SamComponent]]:
    output = np.zeros(labels.shape, dtype=np.uint16)
    records: list[SamComponent] = []
    new_label = 0
    for source_value in np.unique(labels):
        source_label = int(source_value)
        if source_label == 0:
            continue
        connected, count = ndimage.label(
            labels == source_label, structure=EIGHT_CONNECTED
        )
        areas = np.bincount(connected.ravel())[1:]
        for source_component in range(1, count + 1):
            area = int(areas[source_component - 1])
            retained_label: int | None = None
            if area >= min_component_area:
                new_label += 1
                if new_label > np.iinfo(np.uint16).max:
                    raise OverflowError("Too many instances for uint16 output")
                output[connected == source_component] = new_label
                retained_label = new_label
            records.append(
                SamComponent(
                    sample=sample,
                    source_label=source_label,
                    source_component=source_component,
                    new_label=retained_label,
                    area_pixels=area,
                )
            )
    return output, records


def _label_color(label: int) -> tuple[int, int, int]:
    hue = (label * 0.6180339887498949) % 1.0
    red, green, blue = colorsys.hsv_to_rgb(hue, 0.72, 1.0)
    return round(red * 255), round(green * 255), round(blue * 255)


def make_sam_output_overlay(original: np.ndarray, labels: np.ndarray) -> Image.Image:
    image = original.copy()
    for value in np.unique(labels):
        label = int(value)
        if label == 0:
            continue
        mask = labels == label
        color = np.asarray(_label_color(label), dtype=np.float32)
        image[mask] = np.round(
            (1.0 - OVERLAY_ALPHA) * image[mask].astype(np.float32)
            + OVERLAY_ALPHA * color
        ).astype(np.uint8)
    return Image.fromarray(image, mode="RGB")


def _component_outline(mask: np.ndarray) -> np.ndarray:
    return mask & ~ndimage.binary_erosion(mask, structure=EIGHT_CONNECTED)


def make_sam_review_cell(
    sample: str,
    source_label: int,
    source_labels: np.ndarray,
    original: np.ndarray,
    original_overlay: np.ndarray,
    retained_components: list[SamComponent],
) -> Image.Image:
    source_mask = source_labels == source_label
    sam_color, alpha, fit_error = infer_overlay_color(
        original, original_overlay, source_mask
    )
    color = np.asarray(sam_color, dtype=np.uint8)
    left = original.copy()
    left[source_mask] = np.round(
        (1.0 - alpha) * left[source_mask].astype(np.float32) + alpha * color
    ).astype(np.uint8)
    left[_component_outline(source_mask)] = color
    right = np.full_like(original, 20)

    connected, _ = ndimage.label(source_mask, structure=EIGHT_CONNECTED)
    annotations: list[tuple[int, int, str]] = []
    areas: list[int] = []
    for display_number, record in enumerate(
        sorted(retained_components, key=lambda item: item.area_pixels, reverse=True),
        start=1,
    ):
        mask = connected == record.source_component
        right[mask] = color
        right[_component_outline(mask)] = (255, 255, 255)
        ys, xs = np.nonzero(mask)
        if xs.size:
            annotations.append(
                (int(np.median(xs)), int(np.median(ys)), str(display_number))
            )
        areas.append(record.area_pixels)

    panel_size = 320
    left_image = Image.fromarray(left, mode="RGB").resize(
        (panel_size, panel_size), Image.Resampling.LANCZOS
    )
    right_image = Image.fromarray(right, mode="RGB").resize(
        (panel_size, panel_size), Image.Resampling.LANCZOS
    )
    header = 68
    margin = 8
    gap = 10
    cell = Image.new(
        "RGB",
        (margin * 2 + panel_size * 2 + gap, header + panel_size + margin),
        (25, 25, 29),
    )
    draw = ImageDraw.Draw(cell)
    draw.text(
        (margin, 7),
        f"{sample}: original SAM ID {source_label}",
        fill="white",
        font=_font(17),
    )
    draw.text(
        (margin, 30),
        f"Split into {len(retained_components)} retained components; areas: "
        + ", ".join(f"{area:,}" for area in areas),
        fill=(205, 207, 215),
        font=_font(12),
    )
    draw.text(
        (margin, 48),
        f"Original color RGB {sam_color}, alpha {alpha:.2f}, fit MAE {fit_error:.3f}",
        fill=(205, 207, 215),
        font=_font(12),
    )
    cell.paste(left_image, (margin, header))
    right_x = margin + panel_size + gap
    cell.paste(right_image, (right_x, header))

    scale_x = panel_size / source_labels.shape[1]
    scale_y = panel_size / source_labels.shape[0]
    label_font = _font(13)
    for x, y, number in annotations:
        center_x = right_x + round(x * scale_x)
        center_y = header + round(y * scale_y)
        box = draw.textbbox(
            (center_x, center_y), number, font=label_font, anchor="mm"
        )
        draw.rectangle(
            (box[0] - 2, box[1] - 1, box[2] + 2, box[3] + 1),
            fill=(0, 0, 0),
            outline=sam_color,
        )
        draw.text(
            (center_x, center_y),
            number,
            fill="white",
            font=label_font,
            anchor="mm",
        )
    return cell


def make_contact_sheet(cells: list[Image.Image], columns: int = 2) -> Image.Image:
    gap = 10
    rows = (len(cells) + columns - 1) // columns
    width = columns * cells[0].width + (columns + 1) * gap
    height = rows * cells[0].height + (rows + 1) * gap
    sheet = Image.new("RGB", (width, height), (230, 230, 234))
    for index, cell in enumerate(cells):
        x = gap + (index % columns) * (cell.width + gap)
        y = gap + (index // columns) * (cell.height + gap)
        sheet.paste(cell, (x, y))
    return sheet


def _find_original(folder: Path, sample: str) -> Path:
    candidates = sorted(folder.glob(f"{sample}.*"))
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"Expected one original for {sample}, found {candidates}"
        )
    return candidates[0]


def process_sam2(
    source_dir: Path,
    originals_dir: Path,
    output_dir: Path,
    inspection_dir: Path,
) -> None:
    sources = sorted(source_dir.glob("*_instances.tiff"))
    if not sources:
        raise FileNotFoundError(f"No *_instances.tiff masks found in {source_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    inspection_dir.mkdir(parents=True, exist_ok=True)

    component_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    review_rows: list[dict[str, object]] = []
    for source_path in sources:
        sample = _sample_name(source_path)
        source_labels = _load_label_map(source_path)
        split_labels, records = split_sam_label_map(
            source_labels, sample, DEFAULT_MIN_COMPONENT_AREA
        )
        Image.fromarray(split_labels).save(output_dir / source_path.name)

        original_path = _find_original(originals_dir, sample)
        with Image.open(original_path) as image:
            original = _display_rgb(np.asarray(image))
        make_sam_output_overlay(original, split_labels).save(
            output_dir / f"{sample}_overlay.png"
        )

        retained = [record for record in records if record.retained]
        discarded = [record for record in records if not record.retained]
        source_count = int(np.count_nonzero(np.unique(source_labels)))
        retained_by_source: dict[int, list[SamComponent]] = {}
        for record in retained:
            retained_by_source.setdefault(record.source_label, []).append(record)

        overlay_path = source_dir / f"{sample}_overlay.png"
        with Image.open(overlay_path) as image:
            original_overlay = np.asarray(image.convert("RGB"))
        review_cells: list[Image.Image] = []
        for source_label, source_records in retained_by_source.items():
            if len(source_records) < 2:
                continue
            large_count = sum(
                record.area_pixels >= DEFAULT_REVIEW_LARGE_AREA
                for record in source_records
            )
            review_rows.append(
                {
                    "sample": sample,
                    "source_label": source_label,
                    "retained_component_count": len(source_records),
                    "large_component_count_ge_1000px": large_count,
                    "retained_component_areas_desc": ";".join(
                        str(record.area_pixels)
                        for record in sorted(
                            source_records,
                            key=lambda item: item.area_pixels,
                            reverse=True,
                        )
                    ),
                    "contact_sheet_generated": large_count >= 2,
                }
            )
            if large_count >= 2:
                review_cells.append(
                    make_sam_review_cell(
                        sample,
                        source_label,
                        source_labels,
                        original,
                        original_overlay,
                        source_records,
                    )
                )
        if review_cells:
            make_contact_sheet(review_cells).save(
                inspection_dir / f"{sample}_large-split-review.png"
            )

        for record in records:
            component_rows.append(
                {
                    "sample": record.sample,
                    "source_label": record.source_label,
                    "source_component": record.source_component,
                    "new_label": "" if record.new_label is None else record.new_label,
                    "area_pixels": record.area_pixels,
                    "retained": record.retained,
                }
            )
        summary_rows.append(
            {
                "sample": sample,
                "source_instance_count": source_count,
                "retained_instance_count": len(retained),
                "discarded_component_count_lt_10px": len(discarded),
                "discarded_pixel_count": sum(
                    record.area_pixels for record in discarded
                ),
                "source_labels_split_into_multiple_retained_instances": sum(
                    len(source_records) >= 2
                    for source_records in retained_by_source.values()
                ),
                "source_labels_with_multiple_components_ge_1000px": sum(
                    sum(
                        record.area_pixels >= DEFAULT_REVIEW_LARGE_AREA
                        for record in source_records
                    )
                    >= 2
                    for source_records in retained_by_source.values()
                ),
            }
        )
        print(
            f"SAM2 {sample}: {source_count} source IDs -> {len(retained)} instances; "
            f"discarded {len(discarded)} components / "
            f"{sum(record.area_pixels for record in discarded)} px"
        )

    _write_csv(
        output_dir / "component-report.csv",
        [
            "sample",
            "source_label",
            "source_component",
            "new_label",
            "area_pixels",
            "retained",
        ],
        component_rows,
    )
    _write_csv(
        output_dir / "summary.csv",
        [
            "sample",
            "source_instance_count",
            "retained_instance_count",
            "discarded_component_count_lt_10px",
            "discarded_pixel_count",
            "source_labels_split_into_multiple_retained_instances",
            "source_labels_with_multiple_components_ge_1000px",
        ],
        summary_rows,
    )
    _write_csv(
        inspection_dir / "review.csv",
        [
            "sample",
            "source_label",
            "retained_component_count",
            "large_component_count_ge_1000px",
            "retained_component_areas_desc",
            "contact_sheet_generated",
        ],
        review_rows,
        allow_empty=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        nargs="?",
        choices=("ground-truth", "sam2", "all"),
        default="all",
        help="Pipeline to run (default: all).",
    )
    parser.add_argument(
        "--gt-source",
        type=Path,
        default=VALIDATION_DIR / "ground-truth-with-color-collisions",
    )
    parser.add_argument(
        "--gt-output",
        type=Path,
        default=VALIDATION_DIR / "ground-truth",
    )
    parser.add_argument(
        "--sam2-source",
        type=Path,
        default=VALIDATION_DIR / "sam2",
    )
    parser.add_argument(
        "--sam2-output",
        type=Path,
        default=VALIDATION_DIR / "sam2-disconnected",
    )
    parser.add_argument(
        "--originals-dir",
        type=Path,
        default=VALIDATION_DIR / "originals",
    )
    parser.add_argument(
        "--inspection",
        type=Path,
        default=VALIDATION_DIR / "SAM-inspection" / "disconnected-split-review",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode in ("ground-truth", "all"):
        process_ground_truth(args.gt_source, args.gt_output)
    if args.mode in ("sam2", "all"):
        process_sam2(
            args.sam2_source,
            args.originals_dir,
            args.sam2_output,
            args.inspection,
        )


if __name__ == "__main__":
    main()
