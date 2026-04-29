"""Qt dialog widgets for plotting, configuration editing, and 3D preview controls."""

import json
import os
from dataclasses import fields
from typing import Any, Dict, Tuple
import numpy as np

from PyQt6 import QtGui, QtWidgets
from PyQt6.QtCore import QPointF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QPen, QPolygonF
from PyQt6.QtWidgets import QDialogButtonBox

import pyqtgraph as pg
from scipy.ndimage import convolve

from .datastructures import AppConfig, BatchResult
from .utils import label_to_rgb


class PlotProfileDialog(QtWidgets.QDialog):
    """Render and manage the plot profile dialog workflow."""
    closed = pyqtSignal()
    clearRequested = pyqtSignal()

    def __init__(self, parent: QtWidgets.QWidget | None = None):
        """Initialize plot profile dialog and establish required runtime state.
        
        Args:
            parent (QtWidgets.QWidget | None): Value for parent interpreted by this routine."""
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
        """Provide an internal helper for style plot."""
        self.plot_widget.setBackground("w")
        self.plot_widget.showGrid(x=True, y=True, alpha=0.25)
        self.plot_widget.setTitle("Distance transform profile", color="k", size="12pt")
        self.plot_widget.setLabel("bottom", "Distance (px)")
        self.plot_widget.setLabel("left", "DT value")

    def reset_plot(self):
        """Orchestrate reset plot within the current module workflow."""
        self.plot_widget.clear()
        self._style_plot()
        self.info_label.setText(self._empty_text)

    def update_profile(self, distances_px: np.ndarray, values: np.ndarray, summary: str):
        """Refresh profile after upstream state changes.
        
        Args:
            distances_px (np.ndarray): Value for distances px interpreted by this routine.
            values (np.ndarray): Value for values interpreted by this routine.
            summary (str): Value for summary interpreted by this routine."""
        self.plot_widget.clear()
        self._style_plot()
        pen = pg.mkPen(color="#D9534F", width=2)
        self.plot_widget.plot(distances_px, values, pen=pen)
        self.info_label.setText(summary)

    def _on_clear_clicked(self):
        """Handle the lear clicked event and propagate UI state updates."""
        self.clearRequested.emit()

    def closeEvent(self, event: QtGui.QCloseEvent):
        """Handle the Qt close event for this widget.
        
        Args:
            event (QtGui.QCloseEvent): Value for event interpreted by this routine."""
        self.closed.emit()
        super().closeEvent(event)


class SaveOptionsDialog(QtWidgets.QDialog):
    """Modal dialog for shared export options used by all save actions."""

    def __init__(
        self,
        parent: QtWidgets.QWidget | None = None,
        initial_options: Dict[str, Any] | None = None,
    ):
        """Initialize save options dialog and establish required runtime state.
        
        Args:
            parent (QtWidgets.QWidget | None): Value for parent interpreted by this routine.
            initial_options (Dict[str, Any] | None): Configuration values that control how initial options is processed."""
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
            "dt": QtWidgets.QCheckBox("Distance transform (DT PNG)"),
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
        """Orchestrate options within the current module workflow.
        
        Returns:
            Dict[str, Any]: Mapping containing named options outputs and metadata."""
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
        "render3d_max_display_voxels": {
            "label": "3D max display voxels",
            "kind": "int",
            "min": 1,
            "max": 2_000_000_000,
            "step": 1000,
        },
        "render3d_max_display_edge": {
            "label": "3D max display edge",
            "kind": "int",
            "min": 1,
            "max": 10000,
            "step": 1,
        },
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
        """Initialize config editor dialog and establish required runtime state.
        
        Args:
            config (AppConfig): Configuration values that control how config is processed.
            config_path (str): Filesystem location describing where config path is read or written.
            parent (QtWidgets.QWidget | None): Value for parent interpreted by this routine."""
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
        """Orchestrate result config within the current module workflow.
        
        Returns:
            AppConfig: Result object produced by the result config routine."""
        return AppConfig(**vars(self._result_config))

    def result_config_path(self) -> str:
        """Orchestrate result config path within the current module workflow.
        
        Returns:
            str: String value representing the normalized result config path outcome."""
        return str(self._config_path)

    def _build_form(self) -> None:
        """Provide an internal helper for build form.
        
        Returns:
            None: Result object produced by the build form routine."""
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
        """Provide an internal helper for create widget for field.
        
        Args:
            key (str): Value for key interpreted by this routine.
            spec (Dict[str, Any]): Value for spec interpreted by this routine.
            template_value (Any): Value for template value interpreted by this routine.
        
        Returns:
            QtWidgets.QWidget | None: Result object produced by the create widget for field routine."""
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
        """Provide an internal helper for set widgets from config.
        
        Args:
            cfg (AppConfig): Value for cfg interpreted by this routine.
        
        Returns:
            None: Result object produced by the set widgets from config routine."""
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
        """Provide an internal helper for read config from widgets.
        
        Returns:
            AppConfig: Result object produced by the read config from widgets routine.
        
        Raises:
            ValueError: If provided values violate validation or domain constraints."""
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
        return AppConfig.from_dict(data, strict=True)

    def _set_config_path(self, path: str) -> None:
        """Provide an internal helper for set config path.
        
        Args:
            path (str): Filesystem location describing where path is read or written.
        
        Returns:
            None: Result object produced by the set config path routine."""
        self._config_path = os.path.abspath(path) if path else ""
        self.lbl_path.setText(self._config_path or "(not set)")

    def _on_load_clicked(self) -> None:
        """Handle the load clicked event and propagate UI state updates.
        
        Returns:
            None: Result object produced by the on load clicked routine.
        
        Raises:
            ValueError: If provided values violate validation or domain constraints."""
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
            cfg = AppConfig.load_from_json(selected_path, strict=True)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Load failed", str(exc))
            return
        self._set_widgets_from_config(cfg)
        self._set_config_path(selected_path)

    def _on_save_as_clicked(self) -> None:
        """Handle the save as clicked event and propagate UI state updates.
        
        Returns:
            None: Result object produced by the on save as clicked routine."""
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
        """Handle the accept event and propagate UI state updates.
        
        Returns:
            None: Result object produced by the on accept routine."""
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


class RangeSlider(QtWidgets.QWidget):
    """Histogram-backed min/max range slider for contrast/brightness windowing."""

    rangeChanged = pyqtSignal(float, float)

    def __init__(self, parent=None):
        """Initialize range slider and establish required runtime state.
        
        Args:
            parent (Any): Value for parent interpreted by this routine."""
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setFixedHeight(56)
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed)

        self._low = 0.0
        self._high = 255.0
        self._min_val = 0.0
        self._max_val = 255.0
        self._hist_data: np.ndarray | None = None
        self.hist_polygon = QPolygonF()

        self.bar_padding_y = 10
        self.handle_radius = 6
        self.active_handle: str | None = None

    def set_histogram(self, hist):
        """Update histogram and synchronize dependent runtime state.
        
        Args:
            hist (Any): Value for hist interpreted by this routine."""
        if hist is None or len(hist) == 0:
            self._hist_data = None
            self.hist_polygon = QPolygonF()
            self.update()
            return
        self._hist_data = np.asarray(hist, dtype=float)
        self._rebuild_hist_polygon()
        self.update()

    def set_bounds(self, min_val: float, max_val: float):
        """Update bounds and keep dependent state synchronized.
        
        Args:
            min_val (float): Numeric bound applied during filtering or validation.
            max_val (float): Numeric bound applied during filtering or validation."""
        if max_val <= min_val:
            return
        self._min_val = float(min_val)
        self._max_val = float(max_val)
        self.set_range(self._low, self._high, emit_signal=False)

    def set_range(self, low: float, high: float, emit_signal: bool = False):
        """Update range and synchronize dependent runtime state.
        
        Args:
            low (float): Value for low interpreted by this routine.
            high (float): Value for high interpreted by this routine.
            emit_signal (bool): Value for emit signal interpreted by this routine."""
        self._low = float(low)
        self._high = float(high)
        self._enforce_constraints()
        self.update()
        if emit_signal:
            self.rangeChanged.emit(self._low, self._high)

    def get_range(self) -> tuple[float, float]:
        """Return range in the format expected by downstream consumers.
        
        Returns:
            tuple[float, float]: Tuple with values ordered for downstream processing."""
        return self._low, self._high

    def resizeEvent(self, event):
        """Handle the Qt resize event for this widget.
        
        Args:
            event (Any): Value for event interpreted by this routine."""
        super().resizeEvent(event)
        self._rebuild_hist_polygon()

    def paintEvent(self, event):
        """Handle the Qt paint event for this widget.
        
        Args:
            event (Any): Value for event interpreted by this routine."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        w = self.width()
        h = self.height()

        if not self.hist_polygon.isEmpty():
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(75, 75, 75))
            painter.drawPolygon(self.hist_polygon)

        y_center = h / 2.0
        painter.setPen(QPen(QColor(90, 90, 90), 2))
        painter.drawLine(0, int(y_center), w, int(y_center))

        x_low = self._val_to_x(self._low)
        x_high = self._val_to_x(self._high)
        painter.setPen(QPen(QColor(90, 210, 255), 3))
        painter.drawLine(int(x_low), int(y_center), int(x_high), int(y_center))

        self._draw_handle(painter, x_low, y_center, self.active_handle == "low")
        self._draw_handle(painter, x_high, y_center, self.active_handle == "high")

    def _draw_handle(self, painter: QPainter, x: float, y: float, is_active: bool):
        """Provide an internal helper for draw handle.
        
        Args:
            painter (QPainter): Value for painter interpreted by this routine.
            x (float): Coordinate component representing the x axis value.
            y (float): Coordinate component representing the y axis value.
            is_active (bool): Boolean flag that toggles is active behavior."""
        radius = self.handle_radius + (2 if is_active else 0)
        color = QColor(255, 255, 255) if is_active else QColor(210, 210, 210)
        painter.setBrush(color)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(QPointF(x, y), radius, radius)

    def mousePressEvent(self, event):
        """Handle the Qt mouse press event for this widget.
        
        Args:
            event (Any): Value for event interpreted by this routine."""
        val = self._x_to_val(event.position().x())
        if abs(self._low - val) < abs(self._high - val):
            self.active_handle = "low"
            self._low = val
        else:
            self.active_handle = "high"
            self._high = val
        self._enforce_constraints()
        self.update()
        self.rangeChanged.emit(self._low, self._high)

    def mouseMoveEvent(self, event):
        """Handle the Qt mouse move event for this widget.
        
        Args:
            event (Any): Value for event interpreted by this routine."""
        if self.active_handle is None:
            return
        val = self._x_to_val(event.position().x())
        if self.active_handle == "low":
            self._low = val
        else:
            self._high = val
        self._enforce_constraints()
        self.update()
        self.rangeChanged.emit(self._low, self._high)

    def mouseReleaseEvent(self, event):
        """Handle the Qt mouse release event for this widget.
        
        Args:
            event (Any): Value for event interpreted by this routine."""
        self.active_handle = None
        self.update()

    def _enforce_constraints(self):
        """Provide an internal helper for enforce constraints."""
        self._low = max(self._min_val, min(self._low, self._max_val))
        self._high = max(self._min_val, min(self._high, self._max_val))
        if self._low > self._high:
            if self.active_handle == "low":
                self._high = self._low
            else:
                self._low = self._high

    def _val_to_x(self, val: float) -> float:
        """Provide an internal helper for val to x.
        
        Args:
            val (float): Value for val interpreted by this routine.
        
        Returns:
            float: Result object produced by the val to x routine."""
        denom = self._max_val - self._min_val
        if denom <= 0:
            return 0.0
        ratio = (val - self._min_val) / denom
        ratio = max(0.0, min(1.0, ratio))
        return ratio * max(self.width(), 1)

    def _x_to_val(self, x: float) -> float:
        """Provide an internal helper for x to val.
        
        Args:
            x (float): Coordinate component representing the x axis value.
        
        Returns:
            float: Result object produced by the x to val routine."""
        width = max(self.width(), 1)
        ratio = max(0.0, min(1.0, x / width))
        return self._min_val + ratio * (self._max_val - self._min_val)

    def _rebuild_hist_polygon(self):
        """Provide an internal helper for rebuild hist polygon."""
        if self._hist_data is None or self._hist_data.size == 0:
            self.hist_polygon = QPolygonF()
            return
        h_data = self._hist_data.copy()
        max_val = float(np.max(h_data))
        if max_val > 0:
            h_data /= max_val

        w = max(self.width(), 1)
        h = self.height()
        available_h = max(1.0, h - 2 * self.bar_padding_y)
        n_bins = len(h_data)
        step_w = w / max(n_bins, 1)

        poly = QPolygonF()
        poly.append(QPointF(0.0, h - self.bar_padding_y))
        for i, val in enumerate(h_data):
            x = i * step_w
            y = (h - self.bar_padding_y) - (float(val) * available_h)
            poly.append(QPointF(x, y))
            poly.append(QPointF(x + step_w, y))
        poly.append(QPointF(w, h - self.bar_padding_y))
        self.hist_polygon = poly


class Stack3DWindow(QtWidgets.QDialog):
    """Coordinate UI state and rendering for the stack3 d window."""
    def __init__(
        self,
        batch_result: BatchResult,
        parent: QtWidgets.QWidget | None = None,
        voxel_spacing_zyx: tuple[float, float, float] | None = None,
        max_display_voxels: int = 8_000_000,
        max_display_edge: int = 320,
    ):
        """Initialize stack3 d window and establish required runtime state.
        
        Args:
            batch_result (BatchResult): Value for batch result interpreted by this routine.
            parent (QtWidgets.QWidget | None): Value for parent interpreted by this routine.
            voxel_spacing_zyx (tuple[float, float, float] | None): Value for voxel spacing zyx interpreted by this routine.
            max_display_voxels (int): Value for max display voxels interpreted by this routine.
            max_display_edge (int): Value for max display edge interpreted by this routine."""
        super().__init__(parent)
        self.setWindowTitle("3D Stack Viewer")
        self.resize(900, 700)

        self._batch_result: BatchResult | None = None
        self._gl = None
        self._gl_error: str | None = None
        try:
            import pyqtgraph.opengl as gl
            self._gl = gl
        except Exception as exc:
            self._gl_error = str(exc)
            self._gl = None

        self._volume_item = None
        self._scatter_item = None
        self._junction_scatter_item = None
        self._line_items = []
        self._volume_source_u8: np.ndarray | None = None
        self._display_shape_zyx: tuple[int, int, int] | None = None
        self._display_step_zyx: tuple[int, int, int] = (1, 1, 1)
        self._voxel_spacing_zyx: tuple[float, float, float] = (1.0, 1.0, 1.0)
        self._max_display_voxels = max(1, int(max_display_voxels))
        self._max_display_edge = max(1, int(max_display_edge))

        self._imaging_options = ["Original", "DT", "Semantic segmentation"]
        self._projection_options = ["Nearest Point", "Brightest Point", "Mean Value"]

        self._build_ui()
        self.update_batch_result(batch_result, voxel_spacing_zyx=voxel_spacing_zyx)

    def set_display_limits(self, max_display_voxels: int, max_display_edge: int) -> None:
        """Update display limits and synchronize dependent runtime state.
        
        Args:
            max_display_voxels (int): Value for max display voxels interpreted by this routine.
            max_display_edge (int): Value for max display edge interpreted by this routine.
        
        Returns:
            None: Result object produced by the set display limits routine."""
        self._max_display_voxels = max(1, int(max_display_voxels))
        self._max_display_edge = max(1, int(max_display_edge))

    def _build_ui(self):
        """Provide an internal helper for build ui."""
        layout = QtWidgets.QVBoxLayout(self)

        controls = QtWidgets.QHBoxLayout()
        controls.addWidget(QtWidgets.QLabel("Imaging:"))
        self.cb_imaging = QtWidgets.QComboBox()
        self.cb_imaging.addItems(self._imaging_options)
        self.cb_imaging.setCurrentText("Semantic segmentation")
        controls.addWidget(self.cb_imaging)

        controls.addWidget(QtWidgets.QLabel("Projection Mode:"))
        self.cb_projection = QtWidgets.QComboBox()
        self.cb_projection.addItems(self._projection_options)
        self.cb_projection.setCurrentText("Brightest Point")
        controls.addWidget(self.cb_projection)
        self.cb_semantic_centerlines = QtWidgets.QCheckBox("Overlay centerlines")
        self.cb_semantic_centerlines.setChecked(False)
        self.cb_semantic_centerlines.setEnabled(False)
        controls.addWidget(self.cb_semantic_centerlines)
        self.cb_semantic_junctions = QtWidgets.QCheckBox("Overlay junctions")
        self.cb_semantic_junctions.setChecked(False)
        self.cb_semantic_junctions.setEnabled(False)
        controls.addWidget(self.cb_semantic_junctions)
        controls.addStretch(1)
        layout.addLayout(controls)

        self.lbl_render_info = QtWidgets.QLabel("")
        self.lbl_render_info.setStyleSheet("color: #9A9A9A;")
        layout.addWidget(self.lbl_render_info)

        if self._gl is None:
            label = QtWidgets.QLabel(
                "3D view unavailable. OpenGL backend could not be loaded."
            )
            label.setWordWrap(True)
            layout.addWidget(label, 1)
        else:
            self.gl_view = self._gl.GLViewWidget()
            self.gl_view.setBackgroundColor("k")
            layout.addWidget(self.gl_view, 1)

        intensity_layout = QtWidgets.QVBoxLayout()
        intensity_layout.setContentsMargins(0, 0, 0, 0)
        intensity_layout.setSpacing(4)
        intensity_layout.addWidget(QtWidgets.QLabel("Contrast / Brightness"))

        self.range_slider = RangeSlider()
        intensity_layout.addWidget(self.range_slider)

        levels_row = QtWidgets.QHBoxLayout()
        levels_row.addWidget(QtWidgets.QLabel("Low:"))
        self.lbl_low = QtWidgets.QLabel("0")
        levels_row.addWidget(self.lbl_low)
        levels_row.addWidget(QtWidgets.QLabel("High:"))
        self.lbl_high = QtWidgets.QLabel("255")
        levels_row.addWidget(self.lbl_high)
        levels_row.addStretch(1)
        intensity_layout.addLayout(levels_row)
        layout.addLayout(intensity_layout)

        self.cb_imaging.currentIndexChanged.connect(self._on_imaging_mode_changed)
        self.cb_projection.currentIndexChanged.connect(self._update_view)
        self.cb_semantic_centerlines.toggled.connect(self._update_view)
        self.cb_semantic_junctions.toggled.connect(self._update_view)
        self.range_slider.rangeChanged.connect(self._on_range_slider_changed)
        self._on_imaging_mode_changed()

    def _on_imaging_mode_changed(self):
        """Handle the maging mode changed event and propagate UI state updates."""
        semantic_mode = self.cb_imaging.currentText() == "Semantic segmentation"
        self.cb_semantic_centerlines.setEnabled(semantic_mode)
        self.cb_semantic_junctions.setEnabled(semantic_mode)
        if (not semantic_mode) and self.cb_semantic_centerlines.isChecked():
            self.cb_semantic_centerlines.blockSignals(True)
            self.cb_semantic_centerlines.setChecked(False)
            self.cb_semantic_centerlines.blockSignals(False)
        if (not semantic_mode) and self.cb_semantic_junctions.isChecked():
            self.cb_semantic_junctions.blockSignals(True)
            self.cb_semantic_junctions.setChecked(False)
            self.cb_semantic_junctions.blockSignals(False)
        self._update_view()

    @staticmethod
    def _parse_voxel_spacing(
        voxel_spacing_zyx: tuple[float, float, float] | list[float] | np.ndarray,
    ) -> tuple[float, float, float]:
        """Provide an internal helper for parse voxel spacing.
        
        Args:
            voxel_spacing_zyx (tuple[float, float, float] | list[float] | np.ndarray): Value for voxel spacing zyx interpreted by this routine.
        
        Returns:
            tuple[float, float, float]: Tuple grouping the parse voxel spacing outputs in a stable positional order.
        
        Raises:
            ValueError: If provided values violate validation or domain constraints."""
        if len(voxel_spacing_zyx) != 3:
            raise ValueError("voxel_spacing_zyx must contain exactly 3 values.")
        spacing = tuple(float(v) for v in voxel_spacing_zyx)
        if any((not np.isfinite(v)) or v <= 0.0 for v in spacing):
            raise ValueError("voxel_spacing_zyx values must be positive finite numbers.")
        return spacing

    def update_batch_result(
        self,
        batch_result: BatchResult,
        voxel_spacing_zyx: tuple[float, float, float] | None = None,
    ):
        """Refresh batch result after upstream state changes.
        
        Args:
            batch_result (BatchResult): Value for batch result interpreted by this routine.
            voxel_spacing_zyx (tuple[float, float, float] | None): Value for voxel spacing zyx interpreted by this routine."""
        self._batch_result = batch_result
        if voxel_spacing_zyx is not None:
            try:
                self._voxel_spacing_zyx = self._parse_voxel_spacing(voxel_spacing_zyx)
            except Exception:
                self._voxel_spacing_zyx = (1.0, 1.0, 1.0)
        elif batch_result is not None:
            stack3d_meta = batch_result.stack3d_meta or {}
            spacing_meta = stack3d_meta.get("voxel_spacing_zyx")
            if spacing_meta is not None:
                try:
                    self._voxel_spacing_zyx = self._parse_voxel_spacing(spacing_meta)
                except Exception:
                    self._voxel_spacing_zyx = (1.0, 1.0, 1.0)
        self._update_view()

    def _collect_stack(self, selector) -> np.ndarray | None:
        """Provide an internal helper for collect stack.
        
        Args:
            selector (Any): Value for selector interpreted by this routine.
        
        Returns:
            np.ndarray | None: Array containing the collect stack output aligned with downstream expectations."""
        if self._batch_result is None:
            return None
        arrays = []
        for frame in self._batch_result.frame_results:
            arr = selector(frame)
            if arr is None:
                arrays.append(None)
                continue
            arr = np.asarray(arr)
            if arr.ndim == 3:
                arr = arr[..., 0]
            arrays.append(arr)
        available = [a for a in arrays if a is not None]
        if not available:
            return None
        min_h = min(a.shape[0] for a in available)
        min_w = min(a.shape[1] for a in available)
        stack = []
        for arr in arrays:
            if arr is None:
                stack.append(np.zeros((min_h, min_w), dtype=np.float32))
            else:
                stack.append(arr[:min_h, :min_w].astype(np.float32, copy=False))
        return np.stack(stack, axis=0)

    def _stack_for_method(
        self,
        method: str,
    ) -> np.ndarray | None:
        """Provide an internal helper for stack for method.
        
        Args:
            method (str): Value for method interpreted by this routine.
        
        Returns:
            np.ndarray | None: Array containing the stack for method output aligned with downstream expectations."""
        if method == "Original":
            return self._collect_stack(lambda fr: fr.orig_img)
        if method == "DT":
            return self._collect_stack(lambda fr: fr.dt)
        if method == "Semantic segmentation":
            if self._batch_result is not None and self._batch_result.semantic_volume is not None:
                volume = np.asarray(self._batch_result.semantic_volume)
                if volume.ndim == 3:
                    return volume.astype(np.int32, copy=False)
            stack = self._collect_stack(lambda fr: fr.labels_2d)
            if stack is None:
                return None
            return stack.astype(np.int32, copy=False)
        return None

    def _update_view(self):
        """Provide an internal helper for update view."""
        if self._batch_result is None:
            return
        method = self.cb_imaging.currentText()
        stack = self._stack_for_method(method)
        if stack is None:
            if method == "Semantic segmentation":
                self.lbl_render_info.setText(
                    "Semantic segmentation unavailable. Run segmentation or Analyze 3D first."
                )
            else:
                self.lbl_render_info.setText("")
            return

        if self._gl is None:
            return

        if self._volume_item is not None:
            self.gl_view.removeItem(self._volume_item)
            self._volume_item = None
        if self._scatter_item is not None:
            self.gl_view.removeItem(self._scatter_item)
            self._scatter_item = None
        if self._junction_scatter_item is not None:
            self.gl_view.removeItem(self._junction_scatter_item)
            self._junction_scatter_item = None
        if self._line_items:
            for line_item in self._line_items:
                self.gl_view.removeItem(line_item)
            self._line_items = []

        stack = np.asarray(stack)
        dataset_info: str | None = None
        if method == "Semantic segmentation":
            nonzero_voxels = int(np.count_nonzero(stack > 0)) if stack.size else 0
            num_labels = int(np.max(stack)) if stack.size else 0
            source = "semantic_volume"
            if self._batch_result is None or self._batch_result.semantic_volume is None:
                source = "frame_labels_2d (fallback)"
            dataset_info = f"source={source}, labels={num_labels}, voxels={nonzero_voxels}"

        stack = self._downsample_for_display(
            stack,
            preserve_nonzero=(method == "Semantic segmentation"),
        )
        self._update_render_info(method, dataset_info=dataset_info)

        if method == "Semantic segmentation":
            self._set_intensity_controls_enabled(False, projection_enabled=True)
            self._volume_source_u8 = None
            semantic = np.asarray(stack, dtype=np.int32)
            semantic_mask = semantic > 0
            if semantic_mask.size == 0 or not np.any(semantic_mask):
                return

            projection = self.cb_projection.currentText()
            overlay_on_top = projection == "Brightest Point"
            scale_x, scale_y, scale_z = self._display_scale_xyz()

            semantic_u8_xyz = np.transpose(
                semantic_mask.astype(np.ubyte, copy=False) * np.ubyte(255),
                (2, 1, 0),
            )
            rgba = self._make_rgba_for_projection(semantic_u8_xyz, projection)
            if rgba.size > 0:
                self._volume_item = self._gl.GLVolumeItem(rgba)
                self._apply_projection_mode(self._volume_item, projection)
                # Keep semantic volume in the same positive coordinate frame
                # as optional centerline points/segments (no centering translate).
                try:
                    self._volume_item.scale(scale_x, scale_y, scale_z)
                except Exception:
                    pass
                self.gl_view.addItem(self._volume_item)

            center_default = pg.Vector(
                (float(semantic.shape[2]) * scale_x) / 2.0,
                (float(semantic.shape[1]) * scale_y) / 2.0,
                (float(semantic.shape[0]) * scale_z) / 2.0,
            )
            max_dim_default = self._scaled_extent_max(
                (int(semantic.shape[2]), int(semantic.shape[1]), int(semantic.shape[0]))
            )

            show_centerlines = self.cb_semantic_centerlines.isChecked()
            show_junctions = self.cb_semantic_junctions.isChecked()
            if not show_centerlines and not show_junctions:
                self.gl_view.opts["center"] = center_default
                self.gl_view.setCameraPosition(distance=max_dim_default * 2.0)
                return

            skeleton_stack = self._batch_result.semantic_skeleton_volume
            if skeleton_stack is None:
                self.gl_view.opts["center"] = center_default
                self.gl_view.setCameraPosition(distance=max_dim_default * 2.0)
                return
            skeleton = np.asarray(skeleton_stack, dtype=np.int32)
            if skeleton.shape != semantic.shape:
                self.gl_view.opts["center"] = center_default
                self.gl_view.setCameraPosition(distance=max_dim_default * 2.0)
                return

            skel_mask = skeleton > 0
            if not np.any(skel_mask):
                self.gl_view.opts["center"] = center_default
                self.gl_view.setCameraPosition(distance=max_dim_default * 2.0)
                return

            vz, vy, vx = self._voxel_spacing_zyx
            scale_xyz = np.asarray((vx, vy, vz), dtype=np.float32)
            world_unit = float(max(1e-6, min(abs(vx), abs(vy), abs(vz))))
            overlay_positions: list[np.ndarray] = []

            if show_centerlines:
                line_pos, line_color = self._build_centerline_segments_xyz(
                    skeleton,
                    max_segments=350000,
                    connectivity=26,
                )
                if line_pos.size > 0:
                    line_pos = line_pos * scale_xyz
                    overlay_positions.append(line_pos)
                    line_color = np.tile(
                        np.array([[1.0, 0.0, 0.0, 1.0]], dtype=np.float32),
                        (line_pos.shape[0], 1),
                    )
                    try:
                        line_item = self._gl.GLLinePlotItem(
                            pos=line_pos,
                            color=line_color,
                            width=2.5 if overlay_on_top else 1.5,
                            mode="lines",
                            antialias=True,
                        )
                        if overlay_on_top:
                            try:
                                line_item.setGLOptions("additive")
                                from OpenGL import GL

                                line_item.updateGLOptions(
                                    {
                                        "glBlendEquation": (GL.GL_FUNC_ADD,),
                                        "glBlendFunc": (GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA),
                                    }
                                )
                            except Exception:
                                pass
                        self.gl_view.addItem(line_item)
                        self._line_items.append(line_item)
                    except Exception:
                        pass

            if show_junctions:
                kernel = np.ones((3, 3, 3), dtype=np.uint8)
                neighbor_count = convolve(
                    skel_mask.astype(np.uint8),
                    kernel,
                    mode="constant",
                    cval=0,
                ) - skel_mask.astype(np.uint8)
                junctions = np.column_stack(np.nonzero(skel_mask & (neighbor_count >= 3)))
                if junctions.size > 0:
                    max_junctions = 40000
                    if junctions.shape[0] > max_junctions:
                        j_step = int(np.ceil(junctions.shape[0] / max_junctions))
                        junctions = junctions[::j_step]
                    junction_pos = np.column_stack(
                        [junctions[:, 2], junctions[:, 1], junctions[:, 0]]
                    ).astype(np.float32)
                    junction_pos *= scale_xyz
                    overlay_positions.append(junction_pos)
                    junction_color = np.tile(
                        np.array([[0.0, 1.0, 0.0, 1.0]], dtype=np.float32),
                        (junction_pos.shape[0], 1),
                    )
                    self._junction_scatter_item = self._gl.GLScatterPlotItem(
                        pos=junction_pos,
                        color=junction_color,
                        size=(3.8 if overlay_on_top else 3.0) * world_unit,
                        pxMode=False,
                    )
                    try:
                        self._junction_scatter_item.setGLOptions("additive")
                        from OpenGL import GL

                        self._junction_scatter_item.updateGLOptions(
                            {
                                "glBlendEquation": (GL.GL_FUNC_ADD,),
                                "glBlendFunc": (GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA),
                            }
                        )
                    except Exception:
                        pass
                    self.gl_view.addItem(self._junction_scatter_item)

            if overlay_positions:
                all_pos = np.concatenate(overlay_positions, axis=0)
                center = np.mean(all_pos, axis=0)
                max_dim = float(np.max(np.ptp(all_pos, axis=0))) if all_pos.shape[0] > 1 else 1.0
                self.gl_view.opts["center"] = pg.Vector(center[0], center[1], center[2])
                self.gl_view.setCameraPosition(distance=max_dim * 2.0)
            else:
                self.gl_view.opts["center"] = center_default
                self.gl_view.setCameraPosition(distance=max_dim_default * 2.0)
            return

        self._set_intensity_controls_enabled(True)
        projection = self.cb_projection.currentText()

        vol = stack.astype(np.float32, copy=False)
        vmin = float(np.min(vol)) if vol.size else 0.0
        vmax = float(np.max(vol)) if vol.size else 1.0
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
            return
        vol_u8 = np.clip((vol - vmin) / (vmax - vmin), 0.0, 1.0)
        vol_u8 = (vol_u8 * 255.0).astype(np.ubyte)
        vol_u8 = np.transpose(vol_u8, (2, 1, 0))
        self._volume_source_u8 = vol_u8
        hist, _ = np.histogram(vol_u8, bins=256, range=(0, 256))
        self.range_slider.set_histogram(hist)
        self._update_level_labels()
        vol_u8 = self._apply_window_levels(vol_u8)
        rgba = self._make_rgba_for_projection(vol_u8, projection)

        self._volume_item = self._gl.GLVolumeItem(rgba)
        self._apply_projection_mode(self._volume_item, projection)
        self._center_and_scale_volume_item(self._volume_item, rgba.shape[:3])
        self.gl_view.addItem(self._volume_item)
        max_dim = self._scaled_extent_max(rgba.shape[:3]) if rgba.size else 1.0
        self.gl_view.opts["center"] = pg.Vector(0.0, 0.0, 0.0)
        self.gl_view.setCameraPosition(distance=max_dim * 2.0)

    def _set_intensity_controls_enabled(self, enabled: bool, projection_enabled: bool | None = None):
        """Provide an internal helper for set intensity controls enabled.
        
        Args:
            enabled (bool): Value for enabled interpreted by this routine.
            projection_enabled (bool | None): Value for projection enabled interpreted by this routine."""
        if projection_enabled is None:
            projection_enabled = enabled
        self.cb_projection.setEnabled(bool(projection_enabled))
        self.range_slider.setEnabled(enabled)
        self.lbl_low.setEnabled(enabled)
        self.lbl_high.setEnabled(enabled)

    def _apply_window_levels(self, vol_u8: np.ndarray) -> np.ndarray:
        """Provide an internal helper for apply window levels.
        
        Args:
            vol_u8 (np.ndarray): Value for vol u8 interpreted by this routine.
        
        Returns:
            np.ndarray: Array containing the apply window levels output aligned with downstream expectations."""
        low, high = self.range_slider.get_range()
        low = max(0.0, min(255.0, float(low)))
        high = max(low + 1.0, min(255.0, float(high)))
        data = vol_u8.astype(np.float32)
        data = (data - low) / (high - low)
        data = np.clip(data, 0.0, 1.0)
        return np.rint(data * 255.0).astype(np.ubyte)

    def _update_level_labels(self):
        """Provide an internal helper for update level labels."""
        low, high = self.range_slider.get_range()
        self.lbl_low.setText(f"{int(round(low))}")
        self.lbl_high.setText(f"{int(round(high))}")

    def _on_range_slider_changed(self, low: float, high: float):
        """Handle the range slider changed event and propagate UI state updates.
        
        Args:
            low (float): Value for low interpreted by this routine.
            high (float): Value for high interpreted by this routine."""
        self._update_level_labels()
        if self._volume_source_u8 is None or self._gl is None:
            return
        if self._volume_item is not None:
            self.gl_view.removeItem(self._volume_item)
            self._volume_item = None
        vol_u8 = self._apply_window_levels(self._volume_source_u8)
        projection = self.cb_projection.currentText()
        rgba = self._make_rgba_for_projection(vol_u8, projection)
        self._volume_item = self._gl.GLVolumeItem(rgba)
        self._apply_projection_mode(self._volume_item, projection)
        self._center_and_scale_volume_item(self._volume_item, rgba.shape[:3])
        self.gl_view.addItem(self._volume_item)

    def _make_rgba_for_projection(self, vol_u8: np.ndarray, projection: str) -> np.ndarray:
        """Provide an internal helper for make rgba for projection.
        
        Args:
            vol_u8 (np.ndarray): Value for vol u8 interpreted by this routine.
            projection (str): Value for projection interpreted by this routine.
        
        Returns:
            np.ndarray: Array containing the make rgba for projection output aligned with downstream expectations."""
        rgba = np.zeros(vol_u8.shape + (4,), dtype=np.ubyte)
        rgba[..., 0] = vol_u8
        rgba[..., 1] = vol_u8
        rgba[..., 2] = vol_u8

        if projection == "Nearest Point":
            rgba[..., 3] = np.where(vol_u8 > 0, 255, 0).astype(np.ubyte)
            return rgba

        if projection == "Mean Value":
            axis = self._dominant_view_axis()
            # Using full axis depth makes sparse volumes too dim; use occupied-ray
            # depth percentile so slider/levels remain effective in practice.
            counts = np.count_nonzero(vol_u8, axis=axis)
            nz_counts = counts[counts > 0]
            if nz_counts.size > 0:
                effective_depth = float(np.percentile(nz_counts, 90))
            else:
                effective_depth = float(vol_u8.shape[axis])
            effective_depth = max(1.0, min(effective_depth, float(vol_u8.shape[axis])))

            scaled_f = vol_u8.astype(np.float32) / effective_depth
            nonzero = vol_u8 > 0
            scaled_f[nonzero] = np.maximum(scaled_f[nonzero], 1.0)
            scaled = np.clip(np.rint(scaled_f), 0.0, 255.0).astype(np.ubyte)
            rgba[..., 0] = scaled
            rgba[..., 1] = scaled
            rgba[..., 2] = scaled
            rgba[..., 3] = scaled
            return rgba

        rgba[..., 3] = vol_u8
        return rgba

    @staticmethod
    def _downsample_with_steps(
        stack_zyx: np.ndarray,
        steps_zyx: tuple[int, int, int],
        preserve_nonzero: bool = False,
    ) -> np.ndarray:
        """Provide an internal helper for downsample with steps.
        
        Args:
            stack_zyx (np.ndarray): Array-like data carrying stack zyx values for analysis.
            steps_zyx (tuple[int, int, int]): Numeric control parameter for steps zyx during computation.
            preserve_nonzero (bool): Value for preserve nonzero interpreted by this routine.
        
        Returns:
            np.ndarray: Array containing the downsample with steps output aligned with downstream expectations."""
        arr = np.asarray(stack_zyx)
        if arr.ndim != 3:
            return arr
        sz = max(1, int(steps_zyx[0]))
        sy = max(1, int(steps_zyx[1]))
        sx = max(1, int(steps_zyx[2]))
        if (sz, sy, sx) == (1, 1, 1):
            return arr
        if not preserve_nonzero:
            return arr[::sz, ::sy, ::sx]

        z, y, x = arr.shape
        rz = int(np.ceil(z / sz))
        ry = int(np.ceil(y / sy))
        rx = int(np.ceil(x / sx))
        pad_z = int(rz * sz - z)
        pad_y = int(ry * sy - y)
        pad_x = int(rx * sx - x)
        arr_pad = np.pad(
            arr,
            ((0, pad_z), (0, pad_y), (0, pad_x)),
            mode="constant",
            constant_values=0,
        )
        arr_blocks = arr_pad.reshape(rz, sz, ry, sy, rx, sx)
        reduced = arr_blocks.max(axis=(1, 3, 5))
        return reduced.astype(arr.dtype, copy=False)

    def _downsample_for_display(
        self,
        stack_zyx: np.ndarray,
        preserve_nonzero: bool = False,
    ) -> np.ndarray:
        """Provide an internal helper for downsample for display.
        
        Args:
            stack_zyx (np.ndarray): Array-like data carrying stack zyx values for analysis.
            preserve_nonzero (bool): Value for preserve nonzero interpreted by this routine.
        
        Returns:
            np.ndarray: Array containing the downsample for display output aligned with downstream expectations."""
        arr = np.asarray(stack_zyx)
        if arr.ndim != 3:
            self._display_shape_zyx = None
            self._display_step_zyx = (1, 1, 1)
            return arr

        z, y, x = arr.shape
        steps = [
            max(1, int(np.ceil(z / float(self._max_display_edge)))),
            max(1, int(np.ceil(y / float(self._max_display_edge)))),
            max(1, int(np.ceil(x / float(self._max_display_edge)))),
        ]

        def reduced_dims() -> tuple[int, int, int]:
            """Orchestrate reduced dims within the current module workflow.
            
            Returns:
                tuple[int, int, int]: Tuple grouping the reduced dims outputs in a stable positional order."""
            return (
                int(np.ceil(z / steps[0])),
                int(np.ceil(y / steps[1])),
                int(np.ceil(x / steps[2])),
            )

        rz, ry, rx = reduced_dims()
        while (rz * ry * rx) > self._max_display_voxels:
            axis = int(np.argmax([rz, ry, rx]))
            steps[axis] += 1
            rz, ry, rx = reduced_dims()

        sz, sy, sx = int(steps[0]), int(steps[1]), int(steps[2])
        self._display_shape_zyx = (rz, ry, rx)
        self._display_step_zyx = (sz, sy, sx)
        if sz == 1 and sy == 1 and sx == 1:
            return arr

        if not preserve_nonzero:
            return arr[::sz, ::sy, ::sx]

        # For sparse skeleton/centerline data, block max keeps any occupied voxel
        # in each downsampled cell and avoids aliasing from strided sampling.
        pad_z = int(rz * sz - z)
        pad_y = int(ry * sy - y)
        pad_x = int(rx * sx - x)
        arr_pad = np.pad(
            arr,
            ((0, pad_z), (0, pad_y), (0, pad_x)),
            mode="constant",
            constant_values=0,
        )
        arr_blocks = arr_pad.reshape(rz, sz, ry, sy, rx, sx)
        reduced = arr_blocks.max(axis=(1, 3, 5))
        return reduced.astype(arr.dtype, copy=False)

    def _update_render_info(self, method: str, dataset_info: str | None = None):
        """Provide an internal helper for update render info.
        
        Args:
            method (str): Value for method interpreted by this routine.
            dataset_info (str | None): Value for dataset info interpreted by this routine."""
        if self._display_shape_zyx is None:
            text = ""
        else:
            rz, ry, rx = self._display_shape_zyx
            sz, sy, sx = self._display_step_zyx
            if (sz, sy, sx) == (1, 1, 1):
                text = f"Render grid (z,y,x): {rz} x {ry} x {rx}"
            else:
                text = (
                    f"Render grid (z,y,x): {rz} x {ry} x {rx} "
                    f"(downsample {method}, step {sz}/{sy}/{sx})"
                )
        if dataset_info:
            text = f"{text} | {dataset_info}" if text else dataset_info
        self.lbl_render_info.setText(text)

    def _display_scale_xyz(self) -> tuple[float, float, float]:
        """Provide an internal helper for display scale xyz.
        
        Returns:
            tuple[float, float, float]: Tuple grouping the display scale xyz outputs in a stable positional order."""
        sz, sy, sx = self._display_step_zyx
        vz, vy, vx = self._voxel_spacing_zyx
        return (
            float(max(1, sx)) * float(vx),
            float(max(1, sy)) * float(vy),
            float(max(1, sz)) * float(vz),
        )

    def _scaled_extent_max(self, shape_xyz: tuple[int, int, int]) -> float:
        """Provide an internal helper for scaled extent max.
        
        Args:
            shape_xyz (tuple[int, int, int]): Value for shape xyz interpreted by this routine.
        
        Returns:
            float: Result object produced by the scaled extent max routine."""
        scale_x, scale_y, scale_z = self._display_scale_xyz()
        return float(
            max(
                float(shape_xyz[0]) * scale_x,
                float(shape_xyz[1]) * scale_y,
                float(shape_xyz[2]) * scale_z,
            )
        )

    def _center_and_scale_volume_item(self, volume_item, shape_xyz: tuple[int, int, int]) -> None:
        """Provide an internal helper for center and scale volume item.
        
        Args:
            volume_item (Any): Array-like data carrying volume item values for analysis.
            shape_xyz (tuple[int, int, int]): Value for shape xyz interpreted by this routine.
        
        Returns:
            None: Result object produced by the center and scale volume item routine."""
        scale_x, scale_y, scale_z = self._display_scale_xyz()
        try:
            volume_item.scale(scale_x, scale_y, scale_z)
        except Exception:
            pass
        volume_item.translate(
            -(float(shape_xyz[0]) * scale_x) / 2.0,
            -(float(shape_xyz[1]) * scale_y) / 2.0,
            -(float(shape_xyz[2]) * scale_z) / 2.0,
        )

    @staticmethod
    def _labels_to_rgba(label_values: np.ndarray, alpha: float = 0.85) -> np.ndarray:
        """Provide an internal helper for labels to rgba.
        
        Args:
            label_values (np.ndarray): Value for label values interpreted by this routine.
            alpha (float): Numeric control parameter for alpha during computation.
        
        Returns:
            np.ndarray: Array containing the labels to rgba output aligned with downstream expectations."""
        labels = np.asarray(label_values, dtype=np.int32).reshape(-1)
        if labels.size == 0:
            return np.zeros((0, 4), dtype=np.float32)

        unique_labels, inverse = np.unique(labels, return_inverse=True)
        palette = np.zeros((unique_labels.size, 4), dtype=np.float32)
        for idx, label in enumerate(unique_labels):
            if int(label) <= 0:
                palette[idx] = np.array([1.0, 0.0, 0.0, float(alpha)], dtype=np.float32)
                continue
            r, g, b = label_to_rgb(int(label))
            palette[idx] = np.array(
                [float(r) / 255.0, float(g) / 255.0, float(b) / 255.0, float(alpha)],
                dtype=np.float32,
            )
        return palette[inverse]

    @staticmethod
    def _extract_polylines_from_edges(
        src_xyz: np.ndarray,
        dst_xyz: np.ndarray,
        labels: np.ndarray,
    ) -> list[tuple[np.ndarray, int]]:
        """Split an undirected voxel-edge graph into open chains between non-degree-2 nodes."""
        src_i = np.asarray(src_xyz, dtype=np.int32)
        dst_i = np.asarray(dst_xyz, dtype=np.int32)
        lab_i = np.asarray(labels, dtype=np.int32).reshape(-1)
        if src_i.size == 0 or dst_i.size == 0 or lab_i.size == 0:
            return []

        polylines: list[tuple[np.ndarray, int]] = []
        for label in np.unique(lab_i):
            mask = lab_i == int(label)
            if not np.any(mask):
                continue

            edges_src = src_i[mask]
            edges_dst = dst_i[mask]
            adjacency: dict[tuple[int, int, int], list[tuple[tuple[int, int, int], int]]] = {}
            edges: list[tuple[tuple[int, int, int], tuple[int, int, int]]] = []
            for idx in range(edges_src.shape[0]):
                u = tuple(int(v) for v in edges_src[idx])
                v = tuple(int(v) for v in edges_dst[idx])
                if u == v:
                    continue
                eid = len(edges)
                edges.append((u, v))
                adjacency.setdefault(u, []).append((v, eid))
                adjacency.setdefault(v, []).append((u, eid))

            if not edges:
                continue

            degree = {node: len(neighbors) for node, neighbors in adjacency.items()}
            visited_edges: set[int] = set()

            # Trace maximal open chains from endpoints and junctions.
            for node, deg in degree.items():
                if deg == 2:
                    continue
                for nbr, eid in adjacency[node]:
                    if eid in visited_edges:
                        continue
                    visited_edges.add(eid)
                    path = [node]
                    prev = node
                    curr = nbr
                    while True:
                        path.append(curr)
                        curr_deg = degree.get(curr, 0)
                        if curr_deg != 2:
                            break
                        n0, e0 = adjacency[curr][0]
                        n1, e1 = adjacency[curr][1]
                        nxt, next_eid = (n1, e1) if n0 == prev else (n0, e0)
                        if next_eid in visited_edges:
                            break
                        visited_edges.add(next_eid)
                        prev, curr = curr, nxt
                    polylines.append((np.asarray(path, dtype=np.float32), int(label)))

            # Any leftover edges belong to loops; keep them unchanged (edge-wise).
            for eid, (u, v) in enumerate(edges):
                if eid in visited_edges:
                    continue
                polylines.append((np.asarray([u, v], dtype=np.float32), int(label)))

        return polylines

    @staticmethod
    def _taubin_smooth_polyline(
        points_xyz: np.ndarray,
        fixed_mask: np.ndarray,
        iters: int = 12,
        lam: float = 0.4,
        mu: float = -0.42,
    ) -> np.ndarray:
        """Provide an internal helper for taubin smooth polyline.
        
        Args:
            points_xyz (np.ndarray): Value for points xyz interpreted by this routine.
            fixed_mask (np.ndarray): Array-like data carrying fixed mask values for analysis.
            iters (int): Numeric control parameter for iters during computation.
            lam (float): Value for lam interpreted by this routine.
            mu (float): Value for mu interpreted by this routine.
        
        Returns:
            np.ndarray: Array containing the taubin smooth polyline output aligned with downstream expectations.
        
        Raises:
            ValueError: If provided values violate validation or domain constraints."""
        pts = np.asarray(points_xyz, dtype=np.float32)
        fixed = np.asarray(fixed_mask, dtype=bool).reshape(-1)
        if pts.ndim != 2 or pts.shape[1] != 3 or pts.shape[0] < 3:
            return pts.copy()
        if fixed.shape[0] != pts.shape[0]:
            raise ValueError("fixed_mask length must match number of polyline points.")
        if not np.any(fixed):
            # Closed loops (no endpoints/junctions in chain representation): keep unchanged.
            return pts.copy()
        if iters <= 0:
            return pts.copy()

        base = pts.copy()
        out = pts.copy()
        for _ in range(int(iters)):
            for alpha in (float(lam), float(mu)):
                lap = np.zeros_like(out)
                lap[1:-1] = 0.5 * (out[:-2] + out[2:]) - out[1:-1]
                lap[fixed] = 0.0
                out += alpha * lap
                out[fixed] = base[fixed]
        return out

    @staticmethod
    def _build_centerline_segments_xyz(
        skeleton: np.ndarray,
        max_segments: int = 350000,
        connectivity: int = 6,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Provide an internal helper for build centerline segments xyz.
        
        Args:
            skeleton (np.ndarray): Value for skeleton interpreted by this routine.
            max_segments (int): Value for max segments interpreted by this routine.
            connectivity (int): Value for connectivity interpreted by this routine.
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: Array containing the build centerline segments xyz output aligned with downstream expectations.
        
        Raises:
            ValueError: If provided values violate validation or domain constraints."""
        skel = np.asarray(skeleton, dtype=np.int32)
        if skel.ndim != 3 or skel.size == 0:
            return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 4), dtype=np.float32)

        if connectivity not in (6, 18, 26):
            raise ValueError(f"Unsupported connectivity={connectivity}. Expected one of: 6, 18, 26.")

        # Forward half-neighborhood directions to avoid duplicate segments.
        if connectivity == 6:
            directions = (
                (1, 0, 0),
                (0, 1, 0),
                (0, 0, 1),
            )
        elif connectivity == 18:
            directions = (
                (1, 0, 0),
                (0, 1, 0),
                (0, 0, 1),
                (1, 1, 0),
                (1, -1, 0),
                (1, 0, 1),
                (1, 0, -1),
                (0, 1, 1),
                (0, 1, -1),
            )
        else:
            directions = (
                (1, 0, 0),
                (0, 1, 0),
                (0, 0, 1),
                (1, 1, 0),
                (1, -1, 0),
                (1, 0, 1),
                (1, 0, -1),
                (0, 1, 1),
                (0, 1, -1),
                (1, 1, 1),
                (1, 1, -1),
                (1, -1, 1),
                (1, -1, -1),
            )

        by_label = bool(np.any(skel > 1))
        src_chunks = []
        dst_chunks = []
        label_chunks = []
        z_max, y_max, x_max = skel.shape
        for dz, dy, dx in directions:
            z0 = max(0, -dz)
            y0 = max(0, -dy)
            x0 = max(0, -dx)
            z1 = z_max - max(0, dz)
            y1 = y_max - max(0, dy)
            x1 = x_max - max(0, dx)
            if z0 >= z1 or y0 >= y1 or x0 >= x1:
                continue

            src = skel[z0:z1, y0:y1, x0:x1]
            dst = skel[z0 + dz : z1 + dz, y0 + dy : y1 + dy, x0 + dx : x1 + dx]
            mask = (src > 0) & (dst > 0)
            if by_label:
                mask &= src == dst
            if not np.any(mask):
                continue

            local = np.column_stack(np.nonzero(mask))
            src_xyz = np.column_stack(
                (
                    local[:, 2] + x0,
                    local[:, 1] + y0,
                    local[:, 0] + z0,
                )
            ).astype(np.float32, copy=False)
            dst_xyz = src_xyz + np.array([dx, dy, dz], dtype=np.float32)
            seg_labels = src[mask].astype(np.int32, copy=False)

            src_chunks.append(src_xyz)
            dst_chunks.append(dst_xyz)
            label_chunks.append(seg_labels)

        if not src_chunks:
            return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 4), dtype=np.float32)

        src_xyz = np.concatenate(src_chunks, axis=0)
        dst_xyz = np.concatenate(dst_chunks, axis=0)
        labels = np.concatenate(label_chunks, axis=0)

        sm_src_chunks = []
        sm_dst_chunks = []
        sm_label_chunks = []
        z_smooth_passes = 10

        for points_xyz, seg_label in Stack3DWindow._extract_polylines_from_edges(src_xyz, dst_xyz, labels):
            n_points = points_xyz.shape[0]
            if n_points < 2:
                continue
            if n_points >= 5:
                fixed_mask = np.zeros(n_points, dtype=bool)
                fixed_mask[0] = True
                fixed_mask[-1] = True
                points_xyz = Stack3DWindow._taubin_smooth_polyline(
                    points_xyz,
                    fixed_mask=fixed_mask,
                    iters=10,
                    lam=0.4,
                    mu=-0.42,
                )
                if z_smooth_passes > 0 and points_xyz.shape[0] >= 3:
                    # Extra 1D smoothing on z only to reduce slice-to-slice jitter.
                    for _ in range(int(z_smooth_passes)):
                        z_prev = points_xyz[:, 2].copy()
                        z_next = z_prev.copy()
                        z_next[1:-1] = (
                            0.25 * z_prev[:-2]
                            + 0.50 * z_prev[1:-1]
                            + 0.25 * z_prev[2:]
                        )
                        z_next[0] = z_prev[0]
                        z_next[-1] = z_prev[-1]
                        points_xyz[:, 2] = z_next

            seg_src = points_xyz[:-1]
            seg_dst = points_xyz[1:]
            if seg_src.size == 0:
                continue
            keep = np.any(seg_src != seg_dst, axis=1)
            if not np.any(keep):
                continue
            seg_src = seg_src[keep]
            seg_dst = seg_dst[keep]

            sm_src_chunks.append(seg_src.astype(np.float32, copy=False))
            sm_dst_chunks.append(seg_dst.astype(np.float32, copy=False))
            sm_label_chunks.append(np.full(seg_src.shape[0], int(seg_label), dtype=np.int32))

        if sm_src_chunks:
            src_xyz = np.concatenate(sm_src_chunks, axis=0)
            dst_xyz = np.concatenate(sm_dst_chunks, axis=0)
            labels = np.concatenate(sm_label_chunks, axis=0)

        if src_xyz.shape[0] > max_segments:
            step = int(np.ceil(src_xyz.shape[0] / max_segments))
            src_xyz = src_xyz[::step]
            dst_xyz = dst_xyz[::step]
            labels = labels[::step]

        pos = np.empty((src_xyz.shape[0] * 2, 3), dtype=np.float32)
        pos[0::2] = src_xyz
        pos[1::2] = dst_xyz
        color = Stack3DWindow._labels_to_rgba(np.repeat(labels, 2), alpha=0.95)
        return pos, color

    def _dominant_view_axis(self) -> int:
        # GLVolumeItem selects slices from the dominant camera axis (x=0, y=1, z=2).
        """Provide an internal helper for dominant view axis.
        
        Returns:
            int: Result object produced by the dominant view axis routine."""
        if self._gl is None or not hasattr(self, "gl_view"):
            return 2
        try:
            cam = self.gl_view.cameraPosition()
            center = self.gl_view.opts.get("center", pg.Vector(0.0, 0.0, 0.0))
            cam_vec = np.array([float(cam.x()), float(cam.y()), float(cam.z())], dtype=np.float32)
            if hasattr(center, "x"):
                ctr_vec = np.array(
                    [float(center.x()), float(center.y()), float(center.z())],
                    dtype=np.float32,
                )
            else:
                ctr_vec = np.asarray(center, dtype=np.float32)
            delta = cam_vec - ctr_vec
            if not np.isfinite(delta).all():
                return 2
            return int(np.argmax(np.abs(delta)))
        except Exception:
            return 2

    def _apply_projection_mode(self, volume_item, projection: str):
        # Approximate ImageJ-like volume compositing with GL blending modes.
        """Provide an internal helper for apply projection mode.
        
        Args:
            volume_item (Any): Array-like data carrying volume item values for analysis.
            projection (str): Value for projection interpreted by this routine."""
        try:
            from OpenGL import GL
        except Exception:
            return

        if projection == "Nearest Point":
            volume_item.setGLOptions("translucent")
            volume_item.updateGLOptions(
                {
                    "glBlendEquation": (GL.GL_FUNC_ADD,),
                    "glBlendFuncSeparate": (
                        GL.GL_SRC_ALPHA,
                        GL.GL_ONE_MINUS_SRC_ALPHA,
                        GL.GL_ONE,
                        GL.GL_ONE_MINUS_SRC_ALPHA,
                    ),
                }
            )
        elif projection == "Brightest Point":
            volume_item.setGLOptions("additive")
            opts = {"glBlendFunc": (GL.GL_ONE, GL.GL_ONE)}
            if hasattr(GL, "GL_MAX"):
                opts["glBlendEquation"] = (GL.GL_MAX,)
            volume_item.updateGLOptions(opts)
        else:
            volume_item.setGLOptions("additive")
            volume_item.updateGLOptions(
                {
                    "glBlendEquation": (GL.GL_FUNC_ADD,),
                    "glBlendFunc": (GL.GL_ONE, GL.GL_ONE),
                }
            )
