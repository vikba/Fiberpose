"""UI <-> configuration binding helpers for MainWindow."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .datastructures import AppConfig, SampleProperties


@dataclass
class InvalidationFlags:
    """Computed invalidation deltas after applying UI state to config."""

    prediction_changed: bool = False
    segmentation_changed: bool = False
    measurements_changed: bool = False
    invalidate_centerlines: bool = False
    sample_properties_changed: bool = False


class UiConfigBinder:
    """Owns synchronization logic between Qt widgets and configuration models."""

    @staticmethod
    def sync_ui_to_sample_properties(window, sample_props: SampleProperties) -> tuple[SampleProperties, bool]:
        prev_px = float(getattr(sample_props, "pixel_size_x", 1.0))
        prev_py = float(getattr(sample_props, "pixel_size_y", 1.0))
        prev_z = float(getattr(sample_props, "voxel_size_z", 1.0))
        prev_unit = str(getattr(sample_props, "unit", "px"))

        def _parse_positive(text: str, fallback: float) -> float:
            text = (text or "").strip()
            if not text:
                return float(fallback)
            try:
                value = float(text)
            except ValueError:
                return float(fallback)
            if value > 0 and np.isfinite(value):
                return float(value)
            return float(fallback)

        px = _parse_positive(
            window.le_pixelsize.text(),
            sample_props.effective_pixel_size_xy(),
        )
        z = _parse_positive(
            window.le_voxeldepth.text(),
            sample_props.voxel_size_z,
        )

        sample_props.pixel_size_x = float(px)
        sample_props.pixel_size_y = float(px)
        sample_props.voxel_size_z = float(z)
        unit_txt = window.cb_units.currentText().strip()
        sample_props.unit = unit_txt if unit_txt else "px"
        sample_props.source = "manual"

        changed = (
            not math.isclose(prev_px, float(sample_props.pixel_size_x), rel_tol=1e-9, abs_tol=1e-9)
            or not math.isclose(prev_py, float(sample_props.pixel_size_y), rel_tol=1e-9, abs_tol=1e-9)
            or not math.isclose(prev_z, float(sample_props.voxel_size_z), rel_tol=1e-9, abs_tol=1e-9)
            or prev_unit != str(sample_props.unit)
        )
        return sample_props, changed

    @staticmethod
    def sync_sample_properties_to_ui(window, sample_props: SampleProperties) -> None:
        px = sample_props.effective_pixel_size_xy()
        window.le_pixelsize.setText(f"{px:.6g}")
        window.le_voxeldepth.setText(f"{sample_props.voxel_size_z:.6g}")

        unit_txt = sample_props.unit if sample_props.unit else "px"
        window.cb_units.blockSignals(True)
        idx = window.cb_units.findText(unit_txt)
        if idx >= 0:
            window.cb_units.setCurrentIndex(idx)
        else:
            window.cb_units.setCurrentText(unit_txt)
        window.cb_units.blockSignals(False)

    @staticmethod
    def sync_ui_to_config(window, config: AppConfig) -> tuple[AppConfig, InvalidationFlags]:
        prev_model_path = str(getattr(config, "model_path", ""))
        prev_scale_label = str(getattr(config, "scale_label", "1x"))
        prev_max_angle = float(getattr(config, "max_angle_deg", 30.0))
        prev_ridge_sigma = float(getattr(config, "ridge_sigma", 1.0))
        prev_ridge_low = float(getattr(config, "ridge_low", 0.01))
        prev_ridge_high = float(getattr(config, "ridge_high", 0.4))
        prev_min_fragment_centerline_length_px = float(
            getattr(config, "min_fragment_centerline_length_px", 10.0)
        )
        prev_strict = bool(getattr(config, "strict_centerlines", False))
        prev_mode = str(getattr(config, "segmentation_mode", "watershed"))
        prev_merge_expansion_factor = float(getattr(config, "merge_expansion_factor", 0.7))
        prev_segmentation_expansion_factor = float(getattr(config, "segmentation_expansion_factor", 1.0))
        prev_threshold = int(getattr(config, "seg_threshold", 1))
        prev_spacing = float(getattr(config, "sample_spacing", 5.0))

        config.model_path = window.cb_model.currentText().strip()
        config.use_gpu = window.chb_usegpu.isChecked()
        config.scale_label = window.cb_scale.currentText().strip()

        if hasattr(window, "sp_max_cent_angle"):
            config.max_angle_deg = float(window.sp_max_cent_angle.value())
        else:
            config.max_angle_deg = float(getattr(config, "max_angle_deg", 30.0))
        config.ridge_sigma = float(getattr(config, "ridge_sigma", 1.0))
        config.ridge_low = float(getattr(config, "ridge_low", 0.01))
        config.ridge_high = float(getattr(config, "ridge_high", 0.4))
        config.min_fragment_centerline_length_px = float(
            getattr(config, "min_fragment_centerline_length_px", 10.0)
        )
        config.strict_centerlines = window.chb_strict.isChecked()

        if window.rb_ridge.isChecked():
            config.segmentation_mode = "radial_expansion"
        else:
            config.segmentation_mode = "watershed"

        config.seg_threshold = int(window.sp_segm_threshold.value())
        config.merge_expansion_factor = float(getattr(config, "merge_expansion_factor", 0.7))
        config.segmentation_expansion_factor = float(getattr(config, "segmentation_expansion_factor", 1.0))

        if hasattr(window, "le_samspacing"):
            text = window.le_samspacing.text().strip()
            try:
                spacing = float(text)
            except Exception:
                spacing = prev_spacing
            if spacing > 0 and np.isfinite(spacing):
                config.sample_spacing = float(spacing)
            else:
                config.sample_spacing = float(prev_spacing)
        else:
            config.sample_spacing = float(getattr(config, "sample_spacing", 5.0))

        prediction_changed = (
            prev_model_path != str(config.model_path)
            or prev_scale_label != str(config.scale_label)
        )
        max_angle_changed = not math.isclose(prev_max_angle, float(config.max_angle_deg), rel_tol=1e-9, abs_tol=1e-9)
        ridge_sigma_changed = not math.isclose(prev_ridge_sigma, float(config.ridge_sigma), rel_tol=1e-9, abs_tol=1e-9)
        ridge_low_changed = not math.isclose(prev_ridge_low, float(config.ridge_low), rel_tol=1e-9, abs_tol=1e-9)
        ridge_high_changed = not math.isclose(prev_ridge_high, float(config.ridge_high), rel_tol=1e-9, abs_tol=1e-9)
        min_fragment_centerline_length_changed = not math.isclose(
            prev_min_fragment_centerline_length_px,
            float(config.min_fragment_centerline_length_px),
            rel_tol=1e-9,
            abs_tol=1e-9,
        )
        segmentation_changed = (
            max_angle_changed
            or ridge_sigma_changed
            or ridge_low_changed
            or ridge_high_changed
            or min_fragment_centerline_length_changed
            or prev_strict != bool(config.strict_centerlines)
            or prev_mode != str(config.segmentation_mode)
            or not math.isclose(prev_merge_expansion_factor, float(config.merge_expansion_factor), rel_tol=1e-9, abs_tol=1e-9)
            or not math.isclose(
                prev_segmentation_expansion_factor,
                float(config.segmentation_expansion_factor),
                rel_tol=1e-9,
                abs_tol=1e-9,
            )
            or prev_threshold != int(config.seg_threshold)
        )
        measurements_changed = not math.isclose(prev_spacing, float(config.sample_spacing), rel_tol=1e-9, abs_tol=1e-9)

        invalidate_centerlines = (
            max_angle_changed
            or ridge_sigma_changed
            or ridge_low_changed
            or ridge_high_changed
            or min_fragment_centerline_length_changed
            or prev_strict != bool(config.strict_centerlines)
            or not math.isclose(prev_merge_expansion_factor, float(config.merge_expansion_factor), rel_tol=1e-9, abs_tol=1e-9)
        )

        flags = InvalidationFlags(
            prediction_changed=prediction_changed,
            segmentation_changed=segmentation_changed,
            measurements_changed=measurements_changed,
            invalidate_centerlines=invalidate_centerlines,
        )
        return config, flags

    @staticmethod
    def sync_config_to_ui(window, config: AppConfig) -> None:
        def set_combo_text(cb, text):
            cb.blockSignals(True)
            idx = cb.findText(text)
            if idx >= 0:
                cb.setCurrentIndex(idx)
            else:
                cb.setCurrentText(text)
            cb.blockSignals(False)

        set_combo_text(window.cb_model, config.model_path)

        window.chb_usegpu.blockSignals(True)
        window.chb_usegpu.setChecked(bool(config.use_gpu))
        window.chb_usegpu.blockSignals(False)

        set_combo_text(window.cb_scale, config.scale_label)

        if hasattr(window, "sp_max_cent_angle"):
            window.sp_max_cent_angle.blockSignals(True)
            window.sp_max_cent_angle.setValue(int(config.max_angle_deg))
            window.sp_max_cent_angle.blockSignals(False)

        window.chb_strict.blockSignals(True)
        window.chb_strict.setChecked(bool(config.strict_centerlines))
        window.chb_strict.blockSignals(False)

        window.rb_watershed.blockSignals(True)
        window.rb_ridge.blockSignals(True)
        if str(config.segmentation_mode).strip().lower() == "radial_expansion":
            window.rb_ridge.setChecked(True)
        else:
            window.rb_watershed.setChecked(True)
        window.rb_watershed.blockSignals(False)
        window.rb_ridge.blockSignals(False)

        window.sp_segm_threshold.blockSignals(True)
        window.sp_segm_threshold.setValue(int(config.seg_threshold))
        window.sp_segm_threshold.blockSignals(False)

        if hasattr(window, "le_samspacing"):
            window.le_samspacing.setText(str(config.sample_spacing))

        if hasattr(window, "_update_strict_checkbox_state"):
            window._update_strict_checkbox_state()
