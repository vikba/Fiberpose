
from typing import Dict
import numpy as np

from PyQt6 import QtGui, QtWidgets
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QDialogButtonBox

import pyqtgraph as pg

from .datastructures import BatchResult


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

        self._imaging_options = ["Original", "Labels", "Centerlines", "DT"]
        self._projection_options = ["Max", "Mean", "Sum"]

        self._build_ui()
        self.update_batch_result(batch_result)

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)

        controls = QtWidgets.QHBoxLayout()
        controls.addWidget(QtWidgets.QLabel("Imaging:"))
        self.cb_imaging = QtWidgets.QComboBox()
        self.cb_imaging.addItems(self._imaging_options)
        controls.addWidget(self.cb_imaging)

        controls.addWidget(QtWidgets.QLabel("Projection:"))
        self.cb_projection = QtWidgets.QComboBox()
        self.cb_projection.addItems(self._projection_options)
        controls.addWidget(self.cb_projection)
        controls.addStretch(1)
        layout.addLayout(controls)

        content = QtWidgets.QHBoxLayout()

        if self._gl is None:
            label = QtWidgets.QLabel(
                "3D view unavailable. OpenGL backend could not be loaded."
            )
            label.setWordWrap(True)
            content.addWidget(label, 2)
        else:
            self.gl_view = self._gl.GLViewWidget()
            self.gl_view.setBackgroundColor("w")
            grid = self._gl.GLGridItem()
            grid.setSize(200, 200)
            grid.setSpacing(10, 10)
            self.gl_view.addItem(grid)
            content.addWidget(self.gl_view, 2)

        self.projection_plot = pg.PlotWidget()
        self.projection_plot.setBackground("w")
        self.projection_plot.getAxis("bottom").setPen("k")
        self.projection_plot.getAxis("left").setPen("k")
        self.projection_plot.invertY(True)
        self.projection_img = pg.ImageItem()
        self.projection_plot.addItem(self.projection_img)
        content.addWidget(self.projection_plot, 1)

        layout.addLayout(content)

        self.cb_imaging.currentIndexChanged.connect(self._update_view)
        self.cb_projection.currentIndexChanged.connect(self._update_view)

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
            stack = self._collect_stack(lambda fr: fr.labels_2d)
            if stack is None:
                return None
            return (stack > 0).astype(np.float32)
        if method == "Centerlines":
            stack = self._collect_stack(lambda fr: fr.centerlines)
            if stack is None:
                return None
            return (stack > 0).astype(np.float32)
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

        projection = self.cb_projection.currentText()
        if projection == "Mean":
            proj = np.mean(stack, axis=0)
        elif projection == "Sum":
            proj = np.sum(stack, axis=0)
        else:
            proj = np.max(stack, axis=0)
        if proj is not None:
            self.projection_img.setImage(proj.T, autoLevels=True)

        if self._gl is None:
            return

        if self._volume_item is not None:
            self.gl_view.removeItem(self._volume_item)
            self._volume_item = None
        if self._scatter_item is not None:
            self.gl_view.removeItem(self._scatter_item)
            self._scatter_item = None

        if method == "Centerlines":
            points = np.column_stack(np.nonzero(stack > 0))
            if points.size == 0:
                return
            max_points = 200000
            if points.shape[0] > max_points:
                idx = np.random.choice(points.shape[0], max_points, replace=False)
                points = points[idx]
            pos = np.column_stack([points[:, 2], points[:, 1], points[:, 0]]).astype(np.float32)
            color = np.tile(np.array([[1.0, 0.0, 0.0, 0.8]], dtype=np.float32), (pos.shape[0], 1))
            self._scatter_item = self._gl.GLScatterPlotItem(pos=pos, color=color, size=1.5)
            self.gl_view.addItem(self._scatter_item)
            center = np.mean(pos, axis=0)
            max_dim = float(np.max(np.ptp(pos, axis=0))) if pos.shape[0] > 1 else 1.0
            self.gl_view.opts["center"] = pg.Vector(center[0], center[1], center[2])
            self.gl_view.setCameraPosition(distance=max_dim * 2.0)
            return

        vol = stack
        vmax = float(np.max(vol)) if vol.size else 1.0
        if vmax <= 0:
            return
        vol_u8 = (vol / vmax * 255.0).astype(np.ubyte)
        vol_u8 = np.transpose(vol_u8, (2, 1, 0))
        rgba = np.zeros(vol_u8.shape + (4,), dtype=np.ubyte)
        rgba[..., 0] = vol_u8
        rgba[..., 1] = vol_u8
        rgba[..., 2] = vol_u8
        rgba[..., 3] = vol_u8
        self._volume_item = self._gl.GLVolumeItem(rgba)
        self._volume_item.translate(-rgba.shape[0] / 2, -rgba.shape[1] / 2, -rgba.shape[2] / 2)
        self.gl_view.addItem(self._volume_item)
        max_dim = float(max(rgba.shape[:3])) if rgba.size else 1.0
        self.gl_view.opts["center"] = pg.Vector(0.0, 0.0, 0.0)
        self.gl_view.setCameraPosition(distance=max_dim * 2.0)