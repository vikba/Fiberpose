
from typing import Dict
import numpy as np

from PyQt6 import QtGui, QtWidgets
from PyQt6.QtCore import QPointF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QPen, QPolygonF
from PyQt6.QtWidgets import QDialogButtonBox

import pyqtgraph as pg

from .datastructures import BatchResult
from .utils import label_to_rgb


class PlotProfileDialog(QtWidgets.QDialog):
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


class BatchSaveOptionsDialog(QtWidgets.QDialog):
    """Modal dialog that lets the user choose which batch artifacts to produce per sample."""

    def __init__(
        self,
        parent: QtWidgets.QWidget | None = None,
        initial_options: Dict[str, bool] | None = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Batch export options")
        self.setModal(True)

        layout = QtWidgets.QVBoxLayout(self)
        info_label = QtWidgets.QLabel(
            "Select the result files to save for each processed image."
        )
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        self._checkboxes: Dict[str, QtWidgets.QCheckBox] = {
            "dt": QtWidgets.QCheckBox("Distance transform (DT PNG)"),
            "labels": QtWidgets.QCheckBox("Segmentation labels (colored PNG)"),
            "overlay": QtWidgets.QCheckBox("Overlay visualization"),
            "debug": QtWidgets.QCheckBox("Debug visualization"),
            "samples": QtWidgets.QCheckBox("Samples table (CSV)"),
            "style_vector": QtWidgets.QCheckBox("Style vector (.npy)"),
        }

        defaults = initial_options or {}
        default_states = {
            "dt": True,
            "labels": True,
            "overlay": True,
            "debug": True,
            "samples": True,
            "style_vector": False,
        }
        for key, checkbox in self._checkboxes.items():
            checkbox.setChecked(defaults.get(key, default_states.get(key, True)))
            layout.addWidget(checkbox)

        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def options(self) -> Dict[str, bool]:
        return {key: checkbox.isChecked() for key, checkbox in self._checkboxes.items()}


class RangeSlider(QtWidgets.QWidget):
    """Histogram-backed min/max range slider for contrast/brightness windowing."""

    rangeChanged = pyqtSignal(float, float)

    def __init__(self, parent=None):
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
        if hist is None or len(hist) == 0:
            self._hist_data = None
            self.hist_polygon = QPolygonF()
            self.update()
            return
        self._hist_data = np.asarray(hist, dtype=float)
        self._rebuild_hist_polygon()
        self.update()

    def set_bounds(self, min_val: float, max_val: float):
        if max_val <= min_val:
            return
        self._min_val = float(min_val)
        self._max_val = float(max_val)
        self.set_range(self._low, self._high, emit_signal=False)

    def set_range(self, low: float, high: float, emit_signal: bool = False):
        self._low = float(low)
        self._high = float(high)
        self._enforce_constraints()
        self.update()
        if emit_signal:
            self.rangeChanged.emit(self._low, self._high)

    def get_range(self) -> tuple[float, float]:
        return self._low, self._high

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._rebuild_hist_polygon()

    def paintEvent(self, event):
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
        radius = self.handle_radius + (2 if is_active else 0)
        color = QColor(255, 255, 255) if is_active else QColor(210, 210, 210)
        painter.setBrush(color)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(QPointF(x, y), radius, radius)

    def mousePressEvent(self, event):
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
        self.active_handle = None
        self.update()

    def _enforce_constraints(self):
        self._low = max(self._min_val, min(self._low, self._max_val))
        self._high = max(self._min_val, min(self._high, self._max_val))
        if self._low > self._high:
            if self.active_handle == "low":
                self._high = self._low
            else:
                self._low = self._high

    def _val_to_x(self, val: float) -> float:
        denom = self._max_val - self._min_val
        if denom <= 0:
            return 0.0
        ratio = (val - self._min_val) / denom
        ratio = max(0.0, min(1.0, ratio))
        return ratio * max(self.width(), 1)

    def _x_to_val(self, x: float) -> float:
        width = max(self.width(), 1)
        ratio = max(0.0, min(1.0, x / width))
        return self._min_val + ratio * (self._max_val - self._min_val)

    def _rebuild_hist_polygon(self):
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
    def __init__(self, batch_result: BatchResult, parent: QtWidgets.QWidget | None = None):
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
        self._volume_source_u8: np.ndarray | None = None

        self._imaging_options = ["Original", "Labels", "Centerlines", "DT"]
        self._projection_options = ["Nearest Point", "Brightest Point", "Mean Value"]

        self._build_ui()
        self.update_batch_result(batch_result)

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)

        controls = QtWidgets.QHBoxLayout()
        controls.addWidget(QtWidgets.QLabel("Imaging:"))
        self.cb_imaging = QtWidgets.QComboBox()
        self.cb_imaging.addItems(self._imaging_options)
        controls.addWidget(self.cb_imaging)

        controls.addWidget(QtWidgets.QLabel("Projection Mode:"))
        self.cb_projection = QtWidgets.QComboBox()
        self.cb_projection.addItems(self._projection_options)
        controls.addWidget(self.cb_projection)
        controls.addStretch(1)
        layout.addLayout(controls)

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

        self.cb_imaging.currentIndexChanged.connect(self._update_view)
        self.cb_projection.currentIndexChanged.connect(self._update_view)
        self.range_slider.rangeChanged.connect(self._on_range_slider_changed)

    def update_batch_result(self, batch_result: BatchResult):
        self._batch_result = batch_result
        self._update_view()

    def _collect_stack(self, selector) -> np.ndarray | None:
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

    def _stack_for_method(self, method: str) -> np.ndarray | None:
        if method == "Original":
            stack = self._collect_stack(lambda fr: fr.orig_img)
            if stack is None:
                return None
            return stack
        if method == "Labels":
            if self._batch_result is not None and self._batch_result.merged_labels_volume is not None:
                labels_volume = np.asarray(self._batch_result.merged_labels_volume)
                if labels_volume.ndim == 3:
                    return labels_volume.astype(np.int32, copy=False)
            if self._batch_result is not None and self._batch_result.labels_volume is not None:
                labels_volume = np.asarray(self._batch_result.labels_volume)
                if labels_volume.ndim == 3:
                    return labels_volume.astype(np.int32, copy=False)
            stack = self._collect_stack(lambda fr: fr.labels_2d)
            if stack is None:
                return None
            return stack.astype(np.int32, copy=False)
        if method == "Centerlines":
            if self._batch_result is not None and self._batch_result.merged_skeleton_volume is not None:
                skeleton_volume = np.asarray(self._batch_result.merged_skeleton_volume)
                if skeleton_volume.ndim == 3:
                    return skeleton_volume.astype(np.int32, copy=False)
            stack = self._collect_stack(lambda fr: fr.centerlines)
            if stack is None:
                return None
            return (stack > 0).astype(np.int32)
        if method == "DT":
            return self._collect_stack(lambda fr: fr.dt)
        return None

    def _update_view(self):
        if self._batch_result is None:
            return
        method = self.cb_imaging.currentText()
        stack = self._stack_for_method(method)
        if stack is None:
            return

        if self._gl is None:
            return

        if self._volume_item is not None:
            self.gl_view.removeItem(self._volume_item)
            self._volume_item = None
        if self._scatter_item is not None:
            self.gl_view.removeItem(self._scatter_item)
            self._scatter_item = None

        if method == "Labels":
            self._set_intensity_controls_enabled(False)
            self._volume_source_u8 = None
            labels_volume = np.asarray(stack, dtype=np.int32)
            if labels_volume.size == 0 or not np.any(labels_volume > 0):
                return
            rgba = self._make_rgba_for_labels(labels_volume)
            if rgba.size == 0:
                return
            projection = self.cb_projection.currentText()
            self._volume_item = self._gl.GLVolumeItem(rgba)
            self._apply_projection_mode(self._volume_item, projection)
            self._volume_item.translate(-rgba.shape[0] / 2, -rgba.shape[1] / 2, -rgba.shape[2] / 2)
            self.gl_view.addItem(self._volume_item)
            max_dim = float(max(rgba.shape[:3])) if rgba.size else 1.0
            self.gl_view.opts["center"] = pg.Vector(0.0, 0.0, 0.0)
            self.gl_view.setCameraPosition(distance=max_dim * 2.0)
            return

        if method == "Centerlines":
            self._set_intensity_controls_enabled(False)
            self._volume_source_u8 = None
            skeleton = np.asarray(stack, dtype=np.int32)
            points = np.column_stack(np.nonzero(skeleton > 0))
            if points.size == 0:
                return
            max_points = 200000
            if points.shape[0] > max_points:
                idx = np.random.choice(points.shape[0], max_points, replace=False)
                points = points[idx]
            pos = np.column_stack([points[:, 2], points[:, 1], points[:, 0]]).astype(np.float32)
            point_labels = skeleton[points[:, 0], points[:, 1], points[:, 2]].astype(np.int32, copy=False)
            unique_labels, inverse = np.unique(point_labels, return_inverse=True)
            palette = np.zeros((unique_labels.size, 4), dtype=np.float32)
            for idx, label in enumerate(unique_labels):
                if int(label) <= 0:
                    palette[idx] = np.array([1.0, 0.0, 0.0, 0.8], dtype=np.float32)
                    continue
                r, g, b = label_to_rgb(int(label))
                palette[idx] = np.array(
                    [float(r) / 255.0, float(g) / 255.0, float(b) / 255.0, 0.85],
                    dtype=np.float32,
                )
            color = palette[inverse]
            self._scatter_item = self._gl.GLScatterPlotItem(pos=pos, color=color, size=1.5)
            try:
                from OpenGL import GL

                self._scatter_item.setGLOptions("translucent")
                self._scatter_item.updateGLOptions({"glBlendEquation": (GL.GL_FUNC_ADD,)})
            except Exception:
                pass
            self.gl_view.addItem(self._scatter_item)
            center = np.mean(pos, axis=0)
            max_dim = float(np.max(np.ptp(pos, axis=0))) if pos.shape[0] > 1 else 1.0
            self.gl_view.opts["center"] = pg.Vector(center[0], center[1], center[2])
            self.gl_view.setCameraPosition(distance=max_dim * 2.0)
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
        self._volume_item.translate(-rgba.shape[0] / 2, -rgba.shape[1] / 2, -rgba.shape[2] / 2)
        self.gl_view.addItem(self._volume_item)
        max_dim = float(max(rgba.shape[:3])) if rgba.size else 1.0
        self.gl_view.opts["center"] = pg.Vector(0.0, 0.0, 0.0)
        self.gl_view.setCameraPosition(distance=max_dim * 2.0)

    def _set_intensity_controls_enabled(self, enabled: bool):
        self.cb_projection.setEnabled(enabled)
        self.range_slider.setEnabled(enabled)
        self.lbl_low.setEnabled(enabled)
        self.lbl_high.setEnabled(enabled)

    def _apply_window_levels(self, vol_u8: np.ndarray) -> np.ndarray:
        low, high = self.range_slider.get_range()
        low = max(0.0, min(255.0, float(low)))
        high = max(low + 1.0, min(255.0, float(high)))
        data = vol_u8.astype(np.float32)
        data = (data - low) / (high - low)
        data = np.clip(data, 0.0, 1.0)
        return np.rint(data * 255.0).astype(np.ubyte)

    def _update_level_labels(self):
        low, high = self.range_slider.get_range()
        self.lbl_low.setText(f"{int(round(low))}")
        self.lbl_high.setText(f"{int(round(high))}")

    def _on_range_slider_changed(self, low: float, high: float):
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
        self._volume_item.translate(-rgba.shape[0] / 2, -rgba.shape[1] / 2, -rgba.shape[2] / 2)
        self.gl_view.addItem(self._volume_item)

    def _make_rgba_for_projection(self, vol_u8: np.ndarray, projection: str) -> np.ndarray:
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

    def _make_rgba_for_labels(self, labels_volume: np.ndarray) -> np.ndarray:
        labels = np.asarray(labels_volume, dtype=np.int32)
        if labels.ndim != 3:
            return np.zeros((0, 0, 0, 4), dtype=np.ubyte)

        labels_xyz = np.transpose(labels, (2, 1, 0))
        rgba = np.zeros(labels_xyz.shape + (4,), dtype=np.ubyte)
        positive = labels_xyz > 0
        if not positive.any():
            return rgba

        unique_labels = np.unique(labels_xyz[positive])
        for label in unique_labels:
            r, g, b = label_to_rgb(int(label))
            mask = labels_xyz == int(label)
            rgba[mask, 0] = np.ubyte(r)
            rgba[mask, 1] = np.ubyte(g)
            rgba[mask, 2] = np.ubyte(b)
            rgba[mask, 3] = np.ubyte(220)
        return rgba

    def _dominant_view_axis(self) -> int:
        # GLVolumeItem selects slices from the dominant camera axis (x=0, y=1, z=2).
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
