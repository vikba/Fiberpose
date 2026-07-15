"""UI support classes for dialogs, config binding, view state/rendering, and interaction tools."""

from __future__ import annotations

# ---- merged from gui_dialogs.py ----

import json
import os
from dataclasses import fields
from typing import Any, Dict
import numpy as np

from PyQt6 import QtGui, QtWidgets
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QDialogButtonBox

import pyqtgraph as pg

from .domain import AppConfig


class PlotProfileDialog(QtWidgets.QDialog):
    """Render and manage the plot profile dialog workflow."""
    closed = pyqtSignal()
    clearRequested = pyqtSignal()

    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("Distance Transform Profile")
        self.setModal(False)
        self.resize(520, 420)

        self._empty_text = "Select two points on the image to plot the distance transform profile."

        layout = QtWidgets.QVBoxLayout(self)

        self.info_label = QtWidgets.QLabel(self._empty_text)
        self.info_label.setWordWrap(True)
        layout.addWidget(self.info_label)

        self.plot_widget = pg.PlotWidget()
        layout.addWidget(self.plot_widget, 1)
        self._style_plot()

        button_row = QtWidgets.QHBoxLayout()
        self.btn_clear = QtWidgets.QPushButton("Clear selection")
        self.btn_clear.clicked.connect(self._on_clear_clicked)
        button_row.addWidget(self.btn_clear)
        button_row.addStretch(1)
        btn_close = QtWidgets.QPushButton("Close")
        btn_close.clicked.connect(self.close)
        button_row.addWidget(btn_close)
        layout.addLayout(button_row)

    def _style_plot(self):
        self.plot_widget.setBackground("w")
        self.plot_widget.showGrid(x=True, y=True, alpha=0.25)
        self.plot_widget.setTitle("Distance transform profile", color="k", size="12pt")
        self.plot_widget.setLabel("bottom", "Distance (px)")
        self.plot_widget.setLabel("left", "DT value")

    def reset_plot(self):
        self.plot_widget.clear()
        self._style_plot()
        self.info_label.setText(self._empty_text)

    def update_profile(self, distances_px: np.ndarray, values: np.ndarray, summary: str):
        self.plot_widget.clear()
        self._style_plot()
        pen = pg.mkPen(color="#D9534F", width=2)
        self.plot_widget.plot(distances_px, values, pen=pen)
        self.info_label.setText(summary)

    def _on_clear_clicked(self):
        self.clearRequested.emit()

    def closeEvent(self, event: QtGui.QCloseEvent):
        self.closed.emit()
        super().closeEvent(event)


class SaveOptionsDialog(QtWidgets.QDialog):
    """Modal dialog for shared export options used by all save actions."""

    def __init__(
        self,
        parent: QtWidgets.QWidget | None = None,
        initial_options: Dict[str, Any] | None = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Save options")
        self.setModal(True)

        layout = QtWidgets.QVBoxLayout(self)
        info_label = QtWidgets.QLabel(
            "These options apply to all exports: single image, folder batch, and stack save."
        )
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        self._checkboxes: Dict[str, QtWidgets.QCheckBox] = {
            "dt": QtWidgets.QCheckBox("Distance transform (DT float32 TIFF)"),
            "foreground": QtWidgets.QCheckBox("Foreground probability (float32 TIFF)"),
            "labels": QtWidgets.QCheckBox("Segmentation labels (RGB visualization)"),
            "overlay": QtWidgets.QCheckBox("Overlay visualization"),
            "diameters": QtWidgets.QCheckBox("Diameter measurements overlay"),
            "orientations": QtWidgets.QCheckBox("Orientation measurements overlay"),
            "outline": QtWidgets.QCheckBox("Segmentation outline image"),
            "tables": QtWidgets.QCheckBox("Measurements table (CSV)"),
        }

        defaults = initial_options if isinstance(initial_options, dict) else {}
        default_states: Dict[str, bool] = {
            "dt": True,
            "foreground": True,
            "labels": True,
            "overlay": True,
            "diameters": True,
            "orientations": True,
            "outline": False,
            "tables": True,
        }
        for key, checkbox in self._checkboxes.items():
            checkbox.setChecked(defaults.get(key, default_states.get(key, True)))
            layout.addWidget(checkbox)

        image_format_row = QtWidgets.QHBoxLayout()
        image_format_row.addWidget(QtWidgets.QLabel("Image format:"))
        self.cb_image_format = QtWidgets.QComboBox()
        self.cb_image_format.addItems(["png", "tiff"])
        selected_image_format = str(defaults.get("image_format", "png")).strip().lower()
        idx_image_format = self.cb_image_format.findText(selected_image_format)
        self.cb_image_format.setCurrentIndex(idx_image_format if idx_image_format >= 0 else 0)
        image_format_row.addWidget(self.cb_image_format, 1)
        layout.addLayout(image_format_row)

        collision_row = QtWidgets.QHBoxLayout()
        collision_row.addWidget(QtWidgets.QLabel("Collision policy:"))
        self.cb_collision = QtWidgets.QComboBox()
        self.cb_collision.addItems(["suffix", "overwrite", "error"])
        selected_collision = str(defaults.get("collision", "suffix")).strip().lower()
        idx_collision = self.cb_collision.findText(selected_collision)
        self.cb_collision.setCurrentIndex(idx_collision if idx_collision >= 0 else 0)
        collision_row.addWidget(self.cb_collision, 1)
        layout.addLayout(collision_row)

        self.chb_write_manifest = QtWidgets.QCheckBox("Write export manifest (manifest.json)")
        self.chb_write_manifest.setChecked(bool(defaults.get("write_manifest", True)))
        layout.addWidget(self.chb_write_manifest)

        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def options(self) -> Dict[str, Any]:
        options: Dict[str, Any] = {key: checkbox.isChecked() for key, checkbox in self._checkboxes.items()}
        options["image_format"] = self.cb_image_format.currentText().strip().lower() or "png"
        options["collision"] = self.cb_collision.currentText().strip().lower() or "suffix"
        options["write_manifest"] = bool(self.chb_write_manifest.isChecked())
        return options


class ConfigEditorDialog(QtWidgets.QDialog):
    """Schema-driven editor for AppConfig with load/save-as support."""

    _FIELD_SPEC: Dict[str, Dict[str, Any]] = {
        "model_path": {
            "label": "Model path",
            "kind": "str",
            "tooltip": "Relative or absolute path to the ONNX model (.onnx).",
        },
        "use_gpu": {"label": "Use GPU", "kind": "bool"},
        "scale_label": {
            "label": "Scale",
            "kind": "enum",
            "options": ["0.5x", "1x", "2x", "4x"],
            "editable": True,
        },
        "max_angle_deg": {"label": "Max angle (deg)", "kind": "float", "min": 0.0, "max": 180.0, "step": 1.0},
        "ridge_sigma": {"label": "Ridge sigma", "kind": "float", "min": 0.0, "max": 1e6, "step": 0.1},
        "ridge_low": {"label": "Ridge low", "kind": "float", "min": 0.0, "max": 1e6, "step": 0.01},
        "ridge_high": {"label": "Ridge high", "kind": "float", "min": 0.0, "max": 1e6, "step": 0.01},
        "min_fragment_centerline_length_px": {
            "label": "Min fragment centerline length (px)",
            "kind": "float",
            "min": 0.0,
            "max": 1e6,
            "step": 1.0,
        },
        "strict_centerlines": {"label": "Strict centerlines", "kind": "bool"},
        "keep_segments_without_measurement_centerlines": {
            "label": "Keep segments without measurement centerlines",
            "kind": "bool",
        },
        "segmentation_mode": {
            "label": "Segmentation mode",
            "kind": "enum",
            "options": ["watershed", "radial_expansion"],
            "editable": False,
        },
        "merge_expansion_factor": {
            "label": "Merge expansion factor",
            "kind": "float",
            "min": 0.0,
            "max": 5.0,
            "step": 0.05,
        },
        "segmentation_expansion_factor": {
            "label": "Segmentation expansion factor",
            "kind": "float",
            "min": 0.0,
            "max": 5.0,
            "step": 0.05,
        },
        "seg_threshold": {"label": "Segmentation threshold", "kind": "int", "min": 0, "max": 255, "step": 1},
        "sample_spacing": {"label": "Sample spacing (px)", "kind": "float", "min": 0.1, "max": 1e9, "step": 1.0},
        "grad_quality_points": {
            "label": "Grad quality points (JSON list)",
            "kind": "json_list",
            "tooltip": "Signed-error breakpoints [p0, p1, p2, p3] for DT gradient quality view.",
        },
        "save_options": {
            "label": "Save options (JSON object)",
            "kind": "json",
            "tooltip": "Shared export options for single-image, folder, and stack saves.",
        },
    }

    def __init__(
        self,
        config: AppConfig,
        config_path: str,
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Edit configuration")
        self.setModal(True)
        self.resize(640, 620)

        self._config_path = os.path.abspath(config_path) if config_path else ""
        self._initial_config = AppConfig(**vars(config))
        self._result_config = AppConfig(**vars(config))
        self._widgets: Dict[str, QtWidgets.QWidget] = {}
        self._field_order = [f.name for f in fields(AppConfig)]

        layout = QtWidgets.QVBoxLayout(self)

        path_row = QtWidgets.QHBoxLayout()
        path_row.addWidget(QtWidgets.QLabel("Config file:"))
        self.lbl_path = QtWidgets.QLabel("")
        self.lbl_path.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.lbl_path.setWordWrap(True)
        path_row.addWidget(self.lbl_path, 1)
        self.btn_load = QtWidgets.QPushButton("Load...")
        self.btn_save_as = QtWidgets.QPushButton("Save As...")
        path_row.addWidget(self.btn_load)
        path_row.addWidget(self.btn_save_as)
        layout.addLayout(path_row)

        self.scroll = QtWidgets.QScrollArea()
        self.scroll.setWidgetResizable(True)
        form_host = QtWidgets.QWidget()
        self.form = QtWidgets.QFormLayout(form_host)
        self.form.setFieldGrowthPolicy(QtWidgets.QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        self.form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.scroll.setWidget(form_host)
        layout.addWidget(self.scroll, 1)

        self._build_form()
        self._set_config_path(self._config_path)
        self._set_widgets_from_config(config)

        button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        self.btn_apply = button_box.button(QtWidgets.QDialogButtonBox.StandardButton.Ok)
        if self.btn_apply is not None:
            self.btn_apply.setText("Apply && Save")
        button_box.accepted.connect(self._on_accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

        self.btn_load.clicked.connect(self._on_load_clicked)
        self.btn_save_as.clicked.connect(self._on_save_as_clicked)

    def result_config(self) -> AppConfig:
        return AppConfig(**vars(self._result_config))

    def result_config_path(self) -> str:
        return str(self._config_path)

    def _build_form(self) -> None:
        for key in self._field_order:
            spec = self._FIELD_SPEC.get(key, {})
            template_value = getattr(self._initial_config, key, None)
            widget = self._create_widget_for_field(key, spec, template_value)
            if widget is None:
                continue
            label = spec.get("label") or key.replace("_", " ").capitalize()
            tooltip = spec.get("tooltip")
            if tooltip:
                widget.setToolTip(str(tooltip))
            self.form.addRow(label, widget)
            self._widgets[key] = widget

    def _create_widget_for_field(
        self,
        key: str,
        spec: Dict[str, Any],
        template_value: Any,
    ) -> QtWidgets.QWidget | None:
        kind = spec.get("kind")
        if not kind:
            if isinstance(template_value, bool):
                kind = "bool"
            elif isinstance(template_value, int):
                kind = "int"
            elif isinstance(template_value, float):
                kind = "float"
            elif isinstance(template_value, dict):
                kind = "json"
            else:
                kind = "str"
        if kind == "bool":
            return QtWidgets.QCheckBox()
        if kind == "int":
            spin = QtWidgets.QSpinBox()
            spin.setMinimum(int(spec.get("min", -2_000_000_000)))
            spin.setMaximum(int(spec.get("max", 2_000_000_000)))
            spin.setSingleStep(int(spec.get("step", 1)))
            return spin
        if kind == "float":
            spin = QtWidgets.QDoubleSpinBox()
            spin.setDecimals(6)
            spin.setMinimum(float(spec.get("min", -1e12)))
            spin.setMaximum(float(spec.get("max", 1e12)))
            spin.setSingleStep(float(spec.get("step", 0.1)))
            return spin
        if kind == "enum":
            combo = QtWidgets.QComboBox()
            options = [str(v) for v in spec.get("options", [])]
            combo.addItems(options)
            combo.setEditable(bool(spec.get("editable", False)))
            return combo
        if kind == "json":
            text = QtWidgets.QPlainTextEdit()
            text.setTabChangesFocus(True)
            text.setMinimumHeight(120)
            return text
        if kind == "json_list":
            text = QtWidgets.QPlainTextEdit()
            text.setTabChangesFocus(True)
            text.setMinimumHeight(80)
            return text
        return QtWidgets.QLineEdit()

    def _set_widgets_from_config(self, cfg: AppConfig) -> None:
        for key, widget in self._widgets.items():
            value = getattr(cfg, key, None)
            if isinstance(widget, QtWidgets.QCheckBox):
                widget.setChecked(bool(value))
            elif isinstance(widget, QtWidgets.QSpinBox):
                try:
                    widget.setValue(int(value))
                except Exception:
                    widget.setValue(0)
            elif isinstance(widget, QtWidgets.QDoubleSpinBox):
                try:
                    widget.setValue(float(value))
                except Exception:
                    widget.setValue(0.0)
            elif isinstance(widget, QtWidgets.QComboBox):
                txt = str(value) if value is not None else ""
                idx = widget.findText(txt)
                if idx >= 0:
                    widget.setCurrentIndex(idx)
                else:
                    widget.setCurrentText(txt)
            elif isinstance(widget, QtWidgets.QPlainTextEdit):
                kind = self._FIELD_SPEC.get(key, {}).get("kind")
                if kind == "json_list":
                    payload = list(value) if isinstance(value, (list, tuple)) else []
                else:
                    payload = value if isinstance(value, dict) else {}
                widget.setPlainText(json.dumps(payload, indent=2, sort_keys=True))
            elif isinstance(widget, QtWidgets.QLineEdit):
                widget.setText("" if value is None else str(value))

    def _read_config_from_widgets(self) -> AppConfig:
        data: Dict[str, Any] = {}
        for key in self._field_order:
            widget = self._widgets.get(key)
            if widget is None:
                continue
            if isinstance(widget, QtWidgets.QCheckBox):
                data[key] = bool(widget.isChecked())
            elif isinstance(widget, QtWidgets.QSpinBox):
                data[key] = int(widget.value())
            elif isinstance(widget, QtWidgets.QDoubleSpinBox):
                data[key] = float(widget.value())
            elif isinstance(widget, QtWidgets.QComboBox):
                data[key] = widget.currentText().strip()
            elif isinstance(widget, QtWidgets.QPlainTextEdit):
                raw = widget.toPlainText().strip()
                kind = self._FIELD_SPEC.get(key, {}).get("kind")
                if not raw:
                    data[key] = [] if kind == "json_list" else {}
                else:
                    parsed = json.loads(raw)
                    if kind == "json_list":
                        if not isinstance(parsed, list):
                            raise ValueError(f"Field '{key}' must be a JSON list.")
                    else:
                        if not isinstance(parsed, dict):
                            raise ValueError(f"Field '{key}' must be a JSON object.")
                    data[key] = parsed
            elif isinstance(widget, QtWidgets.QLineEdit):
                data[key] = widget.text().strip()
        return AppConfig.from_dict(data)

    def _set_config_path(self, path: str) -> None:
        self._config_path = os.path.abspath(path) if path else ""
        self.lbl_path.setText(self._config_path or "(not set)")

    def _on_load_clicked(self) -> None:
        start_dir = os.path.dirname(self._config_path) if self._config_path else os.getcwd()
        selected_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select config file",
            start_dir,
            "JSON Files (*.json);;All Files (*)",
        )
        if not selected_path:
            return
        selected_path = os.path.abspath(selected_path)
        try:
            cfg = AppConfig.load_from_json(selected_path)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Load failed", str(exc))
            return
        self._set_widgets_from_config(cfg)
        self._set_config_path(selected_path)

    def _on_save_as_clicked(self) -> None:
        try:
            cfg = self._read_config_from_widgets()
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Invalid config", str(exc))
            return

        start_path = self._config_path or os.path.join(os.getcwd(), "config.json")
        selected_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save config as",
            start_path,
            "JSON Files (*.json);;All Files (*)",
        )
        if not selected_path:
            return
        selected_path = os.path.abspath(selected_path)
        try:
            parent_dir = os.path.dirname(selected_path)
            if parent_dir:
                os.makedirs(parent_dir, exist_ok=True)
            cfg.save_to_json(selected_path)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Save failed", str(exc))
            return

        self._set_config_path(selected_path)
        self._result_config = cfg
        QtWidgets.QMessageBox.information(self, "Saved", f"Config saved to:\n{selected_path}")

    def _on_accept(self) -> None:
        try:
            cfg = self._read_config_from_widgets()
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Invalid config", str(exc))
            return

        if not self._config_path:
            QtWidgets.QMessageBox.warning(
                self,
                "Config path missing",
                "Use 'Save As...' to select a config file path.",
            )
            return

        try:
            parent_dir = os.path.dirname(self._config_path)
            if parent_dir:
                os.makedirs(parent_dir, exist_ok=True)
            cfg.save_to_json(self._config_path)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Save failed", str(exc))
            return

        self._result_config = cfg
        self.accept()


# ---- merged from ui_config_binder.py ----


import math
from dataclasses import dataclass

import numpy as np

from .domain import AppConfig, SampleProperties


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
        prev_model_path = str(config.model_path)
        prev_scale_label = str(config.scale_label)
        prev_ridge_sigma = float(config.ridge_sigma)
        prev_ridge_low = float(config.ridge_low)
        prev_ridge_high = float(config.ridge_high)
        prev_min_fragment_centerline_length_px = float(
            config.min_fragment_centerline_length_px
        )
        prev_strict = bool(config.strict_centerlines)
        prev_mode = str(config.segmentation_mode)
        prev_merge_expansion_factor = float(config.merge_expansion_factor)
        prev_segmentation_expansion_factor = float(config.segmentation_expansion_factor)
        prev_threshold = int(config.seg_threshold)

        config.model_path = window.cb_model.currentText().strip()
        config.use_gpu = window.chb_usegpu.isChecked()
        config.scale_label = window.cb_scale.currentText().strip()

        config.ridge_sigma = float(config.ridge_sigma)
        config.ridge_low = float(config.ridge_low)
        config.ridge_high = float(config.ridge_high)
        config.min_fragment_centerline_length_px = float(
            config.min_fragment_centerline_length_px
        )
        config.strict_centerlines = window.chb_strict.isChecked()

        if window.rb_ridge.isChecked():
            config.segmentation_mode = "radial_expansion"
        else:
            config.segmentation_mode = "watershed"

        config.seg_threshold = int(window.sp_segm_threshold.value())
        config.merge_expansion_factor = float(config.merge_expansion_factor)
        config.segmentation_expansion_factor = float(config.segmentation_expansion_factor)

        prediction_changed = (
            prev_model_path != str(config.model_path)
            or prev_scale_label != str(config.scale_label)
        )
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
            ridge_sigma_changed
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
        invalidate_centerlines = (
            ridge_sigma_changed
            or ridge_low_changed
            or ridge_high_changed
            or min_fragment_centerline_length_changed
            or prev_strict != bool(config.strict_centerlines)
            or not math.isclose(prev_merge_expansion_factor, float(config.merge_expansion_factor), rel_tol=1e-9, abs_tol=1e-9)
        )

        flags = InvalidationFlags(
            prediction_changed=prediction_changed,
            segmentation_changed=segmentation_changed,
            measurements_changed=False,
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

        window._update_strict_checkbox_state()

# ---- merged from view_state.py ----


from dataclasses import dataclass
from enum import Enum


class ViewMode(str, Enum):
    """Canonical view mode labels matching the UI combobox text."""

    ORIGINAL = "Original"
    DT = "DT"
    FOREGROUND = "Foreground"
    LAPLACIAN = "Laplacian"
    CENTERLINES = "Centerlines"
    SEGMENTATION = "Segmentation"
    SEGMENTATION_OUTLINE = "Segmentation outline"
    OVERLAY = "Overlay"
    SAMPLING = "Sampling"
    MEASUREMENTS_OVERLAY = "Measurements overlay"
    GX = "gx"
    GY = "gy"
    DT_GRAD_ERROR = "DT grad error"


@dataclass
class ViewState:
    """UI-level view selection and overlay toggles."""

    mode: str = ViewMode.ORIGINAL.value
    show_pores: bool = False
    show_orientation: bool = False

# ---- merged from view_renderer.py ----


from typing import Callable

import numpy as np
from skimage.segmentation import find_boundaries

from .domain import Frame
from .utils import ensure_rgb_u8, labels_to_color, to_uint8_for_view



class ViewRenderer:
    """Select and build the image to show for a given view state."""

    def __init__(self, grad_error_builder: Callable[[Frame], np.ndarray | None]):
        self._grad_error_builder = grad_error_builder

    @staticmethod
    def _build_segmentation_outline_image(
        base_img: np.ndarray | None,
        labels: np.ndarray | None,
    ) -> np.ndarray | None:
        if labels is None:
            return None
        base_rgb = ensure_rgb_u8(base_img)
        if base_rgb is None or base_rgb.shape[:2] != labels.shape[:2]:
            return None

        out = base_rgb.copy()
        try:
            boundaries = find_boundaries(labels, mode="outer")
        except Exception:
            boundaries = None
        if boundaries is not None and boundaries.any():
            out[boundaries] = (255, 0, 0)
        return out

    @staticmethod
    def _centerline_view(img: np.ndarray | None, labels: np.ndarray | None) -> np.ndarray | None:
        out = ensure_rgb_u8(img)
        if out is None:
            return None
        if labels is None or out.shape[:2] != labels.shape[:2]:
            return out
        painted = out.copy()
        try:
            boundaries = find_boundaries(labels, mode="outer")
        except Exception:
            boundaries = None
        if boundaries is not None and boundaries.any():
            painted[boundaries] = (0, 0, 255)
        return painted

    @staticmethod
    def _labels_rgb(frame: Frame) -> np.ndarray | None:
        labels = frame.labels_2d
        if labels is None:
            return None
        cache_id, cache_img = frame.labels_rgb_cache
        current_id = id(labels)
        if cache_id != current_id:
            cache_img = labels_to_color(labels)
            frame.labels_rgb_cache = (current_id, cache_img)
        return cache_img

    def _segmentation_outline_view(self, frame: Frame) -> np.ndarray | None:
        labels = frame.labels_2d
        if labels is None:
            return None

        base_img = frame.orig_img if frame.scale_factor == 1.0 or frame.orig_img_scaled is None else frame.orig_img_scaled
        cache_id, cache_img = frame.segmentation_outline_cache
        current_id = id(labels)
        if cache_id != current_id or cache_img is None:
            cache_img = self._build_segmentation_outline_image(base_img, labels)
            frame.segmentation_outline_cache = (current_id, cache_img)
        return cache_img

    def render(self, frame: Frame, state: ViewState) -> np.ndarray | None:
        mode = state.mode
        labels_rgb = self._labels_rgb(frame)

        handlers: dict[str, Callable[[], np.ndarray | None]] = {
            ViewMode.ORIGINAL.value: lambda: ensure_rgb_u8(frame.orig_img),
            ViewMode.DT.value: lambda: to_uint8_for_view(frame.dt_normalized),
            ViewMode.FOREGROUND.value: lambda: to_uint8_for_view(frame.foreground_prob),
            ViewMode.LAPLACIAN.value: lambda: to_uint8_for_view(frame.lapl_img),
            ViewMode.CENTERLINES.value: lambda: self._centerline_view(frame.centerlines_comparison_overlay_img, frame.labels_2d),
            ViewMode.SEGMENTATION.value: lambda: labels_rgb,
            ViewMode.SEGMENTATION_OUTLINE.value: lambda: self._segmentation_outline_view(frame),
            ViewMode.OVERLAY.value: lambda: frame.overlay_img,
            ViewMode.SAMPLING.value: lambda: frame.sampling_overlay if frame.sampling_overlay is not None else ensure_rgb_u8(frame.dt_u8),
            ViewMode.GX.value: lambda: to_uint8_for_view(frame.gx),
            ViewMode.GY.value: lambda: to_uint8_for_view(frame.gy),
            ViewMode.DT_GRAD_ERROR.value: lambda: self._grad_error_builder(frame),
        }

        if mode == ViewMode.MEASUREMENTS_OVERLAY.value:
            fallback = frame.overlay_img if frame.overlay_img is not None else labels_rgb
            if state.show_pores:
                return frame.pores_overlay if frame.pores_overlay is not None else fallback
            if state.show_orientation:
                return frame.orientations_overlay if frame.orientations_overlay is not None else fallback
            return frame.diameters_overlay if frame.diameters_overlay is not None else fallback

        handler = handlers.get(mode)
        if handler is None:
            handler = handlers[ViewMode.ORIGINAL.value]
        return handler()

# ---- merged from interaction_tools.py ----


from dataclasses import dataclass, field

import numpy as np
from skimage.measure import profile_line


@dataclass
class ScalebarTool:
    """State machine for selecting a two-point scalebar."""

    length_value: float = 100.0
    unit: str = "um"
    pending: bool = False
    points: list[tuple[float, float]] = field(default_factory=list)

    def start(self, length_value: float, unit: str) -> None:
        self.length_value = float(length_value)
        self.unit = str(unit)
        self.pending = True
        self.points = []

    def cancel(self) -> None:
        self.pending = False
        self.points = []

    def add_point(self, x: float, y: float) -> float | None:
        if not self.pending:
            return None
        self.points.append((float(x), float(y)))
        if len(self.points) < 2:
            return None
        (x0, y0), (x1, y1) = self.points[:2]
        pixel_length = float(np.hypot(x1 - x0, y1 - y0))
        self.pending = False
        self.points = []
        if pixel_length <= 0:
            return None
        return float(self.length_value / pixel_length)


@dataclass
class ProfileTool:
    """Two-point line profile interaction helper."""

    active: bool = False
    points_scene: list[tuple[float, float]] = field(default_factory=list)
    points_img: list[tuple[float, float]] = field(default_factory=list)

    def start(self) -> None:
        self.active = True
        self.points_scene = []
        self.points_img = []

    def stop(self) -> None:
        self.active = False
        self.points_scene = []
        self.points_img = []

    def reset(self) -> None:
        self.points_scene = []
        self.points_img = []

    def add_point(self, scene_xy: tuple[float, float], img_xy: tuple[float, float]) -> bool:
        if len(self.points_scene) >= 2:
            self.reset()
        self.points_scene.append((float(scene_xy[0]), float(scene_xy[1])))
        self.points_img.append((float(img_xy[0]), float(img_xy[1])))
        return len(self.points_scene) == 2

    @staticmethod
    def compute_profile(
        dt: np.ndarray,
        p0_xy: tuple[float, float],
        p1_xy: tuple[float, float],
        *,
        scale_factor: float,
    ) -> tuple[np.ndarray, np.ndarray, str] | None:
        x0, y0 = p0_xy
        x1, y1 = p1_xy
        length_px = float(np.hypot(x1 - x0, y1 - y0))
        if length_px <= 0:
            return None

        profile_vals = profile_line(dt, (y0, x0), (y1, x1), mode="reflect", order=1)
        if profile_vals is None or profile_vals.size == 0:
            return None

        profile_vals = np.asarray(profile_vals, dtype=np.float32, copy=False)
        scale = float(scale_factor)
        if not np.isfinite(scale) or scale <= 0:
            scale = 1.0

        true_length_px = length_px / scale
        profile_vals_true = profile_vals / scale
        distances = np.linspace(0.0, true_length_px, profile_vals.size, dtype=np.float32)
        delta = float(profile_vals_true[-1] - profile_vals_true[0])
        summary = (
            f"Length: {true_length_px:.2f} px · ΔDT: {delta:+.2f} · "
            f"Min/Max: {float(profile_vals_true.min()):.2f} / {float(profile_vals_true.max()):.2f}"
        )
        return distances, profile_vals_true, summary
