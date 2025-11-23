# app/ui_main.py
import os
import math
from datetime import datetime
from typing import Any, Dict
import numpy as np
import cv2

from PyQt6 import QtGui, QtWidgets
from PyQt6.QtCore import Qt, QEvent, QTimer, QPoint, QLineF, QSize, pyqtSignal
from PyQt6.QtWidgets import (
    QMainWindow,
    QFileDialog,
    QGraphicsScene,
    QGraphicsPixmapItem,
    QGraphicsEllipseItem,
    QGraphicsItem,
    QToolButton,
    QDialogButtonBox,
    QMessageBox,
)
from PyQt6.uic import loadUi

import imageio.v3 as iio
import pyqtgraph as pg
import torch
from skimage.measure import profile_line

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

from .fiber_analyzer import FiberAnalyzer
from .utils import ensure_rgb_u8, label_to_rgb, to_uint8_for_view

# ------------------- Paths -------------------

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
DEFAULT_UI_PATH = os.path.join(PROJECT_ROOT, "resources", "fiber_tool.ui")
DEFAULT_CKPT_PATH = os.path.join(PROJECT_ROOT, "checkpoints", "fiber_2000_epoch.pt")
CHECKPOINTS_DIR = os.path.join(PROJECT_ROOT, "checkpoints")

# ------------------- Image helpers -------------------

def labels_to_color(labels: np.ndarray) -> np.ndarray:
    if labels is None:
        return None
    labels = labels.astype(np.int64, copy=False)
    h, w = labels.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    if labels.size == 0:
        return out
    for value in np.unique(labels):
        if value == 0:
            continue
        mask = labels == value
        if not mask.any():
            continue
        color = label_to_rgb(int(value))
        out[mask] = color
    return out

# ------------------- MainWindow -------------------


class ProfilePlotDialog(QtWidgets.QDialog):
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
        }

        defaults = initial_options or {}
        for key, checkbox in self._checkboxes.items():
            checkbox.setChecked(defaults.get(key, True))
            layout.addWidget(checkbox)

        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def options(self) -> Dict[str, bool]:
        return {key: checkbox.isChecked() for key, checkbox in self._checkboxes.items()}


class MainWindow(QMainWindow):
    """
    Single-file refactor notes:
    - One overlay renderer (removed duplicate).
    - One analysis pass driven by current thresholds/params.
    - Clear state when loading a new image.
    - Tightened UI updates/enabling.
    """
    def __init__(self, ui_path: str = DEFAULT_UI_PATH):
        super().__init__()
        loadUi(ui_path, self)
        self.setWindowTitle("Fiberpose — fiber segmentation and analysis tool")
        
        # Scene and view setup (unchanged)...
        self._scene = QGraphicsScene(self.graphicsView)
        self.graphicsView.setScene(self._scene)
        self._pixitem = QGraphicsPixmapItem()
        self._scene.addItem(self._pixitem)
        self._install_drag_drop()

        self.graphicsView.setRenderHints(
            QtGui.QPainter.RenderHint.Antialiasing |
            QtGui.QPainter.RenderHint.SmoothPixmapTransform
        )
        self.graphicsView.setViewportUpdateMode(QtWidgets.QGraphicsView.ViewportUpdateMode.BoundingRectViewportUpdate)
        self.graphicsView.setTransformationAnchor(QtWidgets.QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.graphicsView.setResizeAnchor(QtWidgets.QGraphicsView.ViewportAnchor.AnchorViewCenter)

        # Working dir & state setup (unchanged)...
        self.working_dir = os.path.dirname(ui_path)
        self.orig_img = None
        self.orig_img_scaled = None
        self.dt = None
        self.gray_u8 = None          # CPNet DT (0..255)
        self.gray_img = None         # normalized float [0..1]
        self.lapl_img = None
        self.gx = None
        self.gy = None
        self.analyzer: FiberAnalyzer | None = None
        self.scale_factor: float = 1.0

        self.labels_img = None       # watershed labels (uint16)
        self.overlay_img = None
        self.sampling_overlay = None
        self.measurements_overlay = None

        self.samples_table = None
        self.fiber_summary = None
        self._viz_points = None
        self._angles_deg = None
        self._thickness_units = None
        self._porosity = None
        self._pore_diams_units = None
        self._labels_rgb_cache: tuple[int | None, np.ndarray | None] = (None, None)
        self.current_image_name: str | None = None
        self._scalebar_pending: bool = False
        self._scalebar_points: list[tuple[float, float]] = []
        self._scalebar_length_value: float = 100.0
        self._scalebar_unit: str = "um"
        self._pixmap_source_shape: tuple[int, int] | None = None

        # View/zoom state
        self._zoom_mode_active: bool = False
        self._base_scale: float = 1.0
        self._current_zoom: float = 1.0
        self._min_zoom_factor: float = 0.2
        self._max_zoom_factor: float = 5.0

        self.btn_reset_view = QToolButton(self.graphicsView.viewport())
        icon = self.style().standardIcon(QtWidgets.QStyle.StandardPixmap.SP_TitleBarNormalButton)
        self.btn_reset_view.setIcon(icon)
        self.btn_reset_view.setIconSize(QSize(16, 16))
        self.btn_reset_view.setCursor(Qt.CursorShape.ArrowCursor)
        self.btn_reset_view.setToolTip("Reset zoom to fit the image")
        self.btn_reset_view.setAutoRaise(True)
        self.btn_reset_view.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btn_reset_view.clicked.connect(self.reset_zoom)
        self.btn_reset_view.hide()

        # Batch mode state
        self._batch_mode_active: bool = False
        self._batch_save_options: Dict[str, bool] | None = None

        # Profile plotting tool state
        self._profile_mode_active: bool = False
        self._profile_points_scene: list[tuple[float, float]] = []
        self._profile_points_img: list[tuple[float, float]] = []
        self._profile_line_item: QGraphicsItem | None = None
        self._profile_point_items: list[QGraphicsItem] = []
        self._profile_dialog: ProfilePlotDialog | None = None

        self.segmentation_mode = "watershed"
        self.rb_watershed.blockSignals(True)
        self.rb_watershed.setChecked(True)
        self.rb_watershed.blockSignals(False)
        self.rb_ridge.blockSignals(True)
        self.rb_ridge.setChecked(False)
        self.rb_ridge.blockSignals(False)
        self.chb_strict.blockSignals(True)
        self.chb_strict.setChecked(False)
        self.chb_strict.setEnabled(False)
        self.chb_strict.blockSignals(False)
        self.cluster_enabled = False
        self.chb_cluster.setChecked(False)


        self.cb_scale.blockSignals(True)
        self.cb_scale.clear()
        for value in ("0.5x", "1x", "2x", "4x"):
            self.cb_scale.addItem(value)
        idx = self.cb_scale.findText("1x")
        if idx >= 0:
            self.cb_scale.setCurrentIndex(idx)
        self.cb_scale.blockSignals(False)

        # Menu & actions: updated UI element names (rb_mask1_cent -> rb_centerlines; rb_mask2_segm -> rb_segmentation;
        # sp_threshold1 -> sp_max_cent_angle; sp_threshold2 -> sp_segm_threshold)
        self.action_load_image.triggered.connect(self.load_original)
        self.action_save_masks_predictions.triggered.connect(self.save_predictions)
        self.action_save_measurements.triggered.connect(self.save_measurements)

        self.btn_segment.clicked.connect(self.segment)
        self.btn_analyze_fibers.clicked.connect(self.analyze_fibers)
        self.btn_save_csv.clicked.connect(self.save_measurements)
        self.btn_set_scalebar.clicked.connect(self._on_set_scalebar_clicked)
        self.btn_plot_profile.clicked.connect(self._on_plot_profile_clicked)
        self.action_batch_processing.triggered.connect(self._on_batch_processing)

        # Remove connection of radiobutton signals and instead hook up the combobox change
        self.cb_view.currentIndexChanged.connect(self.refresh_view)

        # GUI defaults
        self.le_minarea.setText("50")
        self.le_mineccentricity.setText("0.9")
        self.le_minskeletonlen.setText("20")
        self.le_samspacing.setText("5")
        self.le_pixelsize.setText("")

        self._init_checkpoint_selector()

        # Disable saves initially
        self.action_save_masks_predictions.setEnabled(False)
        self.action_save_measurements.setEnabled(False)

        self.resize(1100, 800)
        self.refresh_view()
        
        # Delay loading default image until the UI is fully shown.
        QTimer.singleShot(0, self.load_default_image)

        self.ckpt_path = self._current_checkpoint_path() or DEFAULT_CKPT_PATH

        # ---- device / CUDA handling ----
        self.device = torch.device('cpu')   # default
        self._init_device_from_checkbox()   # sets checkbox + self.device accordingly

        # Analyzer instantiated once and reused
        self.analyzer = FiberAnalyzer(
            ckpt_path=self.ckpt_path,
            device=self.device,
            strict=self._strict_centerline_enabled(),
        )
        self.analyzer.set_strict_centerline_filter(self._strict_centerline_enabled())

        self.pipeline_config: Dict[str, Dict[str, Any]] = {}
        self._init_pipeline_config()

        # react to checkbox changes at runtime
        self.chb_usegpu.toggled.connect(self._on_usegpu_toggled)
        self.chb_cluster.toggled.connect(self._on_cluster_toggled)
        self._on_cluster_toggled(self.chb_cluster.isChecked())
        self.rb_watershed.toggled.connect(self._on_segmentation_mode_changed)
        self.rb_ridge.toggled.connect(self._on_segmentation_mode_changed)
        self.chb_strict.toggled.connect(self._on_strict_checkbox_toggled)
        self._update_strict_checkbox_state()

        # Connect toggled signals of rb_show_diam and rb_show_direct to update overlay
        self.rb_show_diam.toggled.connect(self._render_vector_overlay)
        self.rb_show_direct.toggled.connect(self._render_vector_overlay)
        self.rb_show_pores.toggled.connect(self._render_vector_overlay)

    def _init_checkpoint_selector(self) -> None:
        checkpoints = self._discover_checkpoint_files()
        rel_default = os.path.relpath(DEFAULT_CKPT_PATH, PROJECT_ROOT)
        self.cb_model.blockSignals(True)
        self.cb_model.clear()
        if not checkpoints:
            self.cb_model.addItem(rel_default)
        else:
            for rel_path in checkpoints:
                self.cb_model.addItem(rel_path)
            default_index = self.cb_model.findText(rel_default)
            if default_index >= 0:
                self.cb_model.setCurrentIndex(default_index)
            else:
                self.cb_model.setCurrentIndex(0)
        self.cb_model.blockSignals(False)
        if not getattr(self, "_ckpt_selector_connected", False):
            self.cb_model.currentIndexChanged.connect(self._on_checkpoint_selection_changed)
            self._ckpt_selector_connected = True

    def _on_checkpoint_selection_changed(self, index: int):
        new_path = self._current_checkpoint_path()
        if not new_path:
            return
        if not os.path.isfile(new_path):
            QMessageBox.warning(
                self,
                "Checkpoint missing",
                f"Selected checkpoint was not found:\n{new_path}",
            )
            return
        prev_path = getattr(self, "ckpt_path", None)
        if new_path == prev_path:
            return
        self.ckpt_path = new_path
        if self.analyzer is not None:
            self.analyzer.set_checkpoint_path(new_path)
        else:
            logger.info("Checkpoint selection stored for later init: %s", new_path)

    def _discover_checkpoint_files(self) -> list[str]:
        if not os.path.isdir(CHECKPOINTS_DIR):
            return []
        candidates = []
        for name in sorted(os.listdir(CHECKPOINTS_DIR)):
            if not name.lower().endswith(".pt"):
                continue
            abs_path = os.path.join(CHECKPOINTS_DIR, name)
            rel_path = os.path.relpath(abs_path, PROJECT_ROOT)
            candidates.append(rel_path)
        return candidates

    def _current_checkpoint_path(self) -> str | None:
        entry = self.cb_model.currentText().strip()
        if not entry:
            return None
        if os.path.isabs(entry):
            return entry
        return os.path.join(PROJECT_ROOT, entry)


    # ------------------- Display helpers -------------------

    def _set_pixmap_from_array(self, img: np.ndarray | None):
        prev_shape = self._pixmap_source_shape
        if img is None:
            self._pixitem.setPixmap(QtGui.QPixmap())
            self._pixmap_source_shape = None
            self._zoom_mode_active = False
            self._current_zoom = 1.0
            self._base_scale = 1.0
            self.btn_reset_view.hide()
            self._remove_profile_items()
            return
        self._pixmap_source_shape = img.shape[:2]
        qimg = self._numpy_to_qimage(img)
        pix = QtGui.QPixmap.fromImage(qimg)
        # Scale pixmap if it is smaller than the viewport
        view_size = self.graphicsView.viewport().size()
        if pix.width() < view_size.width() or pix.height() < view_size.height():
            pix = pix.scaled(view_size, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self._pixitem.setPixmap(pix)
        self._scene.setSceneRect(self._pixitem.boundingRect())
        force_fit = (not self._zoom_mode_active) or (prev_shape != self._pixmap_source_shape)
        self._fit_view_to_pixmap(force=force_fit)
        self._update_reset_button_position()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_reset_button_position()
        if not self._zoom_mode_active:
            self._fit_view_to_pixmap(force=True)

    @staticmethod
    def _numpy_to_qimage(img: np.ndarray) -> QtGui.QImage:
        if img.ndim == 2:
            u8 = np.ascontiguousarray(img.astype(np.uint8, copy=False))
            h, w = u8.shape
            qimg = QtGui.QImage(u8.data, w, h, w, QtGui.QImage.Format.Format_Grayscale8)
            qimg.ndarray_ref = u8
            return qimg
        elif img.ndim == 3 and img.shape[2] == 3:
            u8 = np.ascontiguousarray(img.astype(np.uint8, copy=False))
            h, w, c = u8.shape
            qimg = QtGui.QImage(u8.data, w, h, c*w, QtGui.QImage.Format.Format_RGB888)
            qimg.ndarray_ref = u8
            return qimg
        else:
            u8 = to_uint8_for_view(img)
            return MainWindow._numpy_to_qimage(u8)
        
    def _fit_view_to_pixmap(self, force: bool = False):
        pm = self._pixitem.pixmap()
        if pm is not None and not pm.isNull():
            if self._zoom_mode_active and not force:
                return
            self.graphicsView.resetTransform()
            self.graphicsView.fitInView(self._pixitem, Qt.AspectRatioMode.KeepAspectRatio)
            self.graphicsView.centerOn(self._pixitem)
            scale = self.graphicsView.transform().m11()
            self._base_scale = scale if scale not in (0.0, None) else 1.0
            self._current_zoom = 1.0
            self._zoom_mode_active = False
            self._update_reset_button_visibility()

    def _create_overlay(
        self,
        base_img: np.ndarray | None,
        labels_img: np.ndarray | None,
        alpha: float = 0.5,
        labels_rgb: np.ndarray | None = None,
    ) -> np.ndarray | None:
        if base_img is None or labels_img is None:
            return None
        base_rgb = ensure_rgb_u8(base_img)
        if labels_rgb is None:
            labels_rgb = labels_to_color(labels_img)
        if base_rgb is None or labels_rgb is None:
            return None
        if base_rgb.shape[:2] != labels_rgb.shape[:2]:
            return None
        out = base_rgb.astype(np.float32, copy=True)
        over = labels_rgb.astype(np.float32, copy=False)
        mask = labels_img > 0
        if mask.any():
            out[mask] = out[mask] * (1.0 - alpha) + over[mask] * alpha
        return out.clip(0, 255).astype(np.uint8)

    def _labels_rgb(self) -> np.ndarray | None:
        if self.labels_img is None:
            return None
        cache_id, cache_img = self._labels_rgb_cache
        current_id = id(self.labels_img)
        if cache_id != current_id:
            cache_img = labels_to_color(self.labels_img)
            self._labels_rgb_cache = (current_id, cache_img)
        return cache_img

    def _ensure_dt_ready(self) -> bool:
        if self.orig_img is None:
            self._show_info("Load original", "Load an image before using the profile tool.")
            return False
        if self.analyzer is None:
            self._show_error("Analyzer unavailable", "Fiber analyzer has not been initialized.")
            return False

        self._sync_pipeline_config_from_ui()
        ridge_cfg = self.pipeline_config.get("ridge_detection", {})
        max_angle = float(ridge_cfg.get("max_angle_deg", self._current_max_angle()))
        centerline_kwargs = dict(ridge_cfg.get("centerline_kwargs", {}))

        if not self.analyzer.dt_predicted():
            try:
                self.analyzer.predict_dt(
                    image=self.orig_img,
                    max_angle_deg=max_angle,
                    centerline_kwargs=centerline_kwargs or None,
                )
            except Exception as exc:
                self._show_error("Prediction failed", str(exc))
                logger.error("Profile DT ensure failed: %s", exc)
                return False

        dt = self.analyzer.get_dt()
        if dt is None:
            self._show_error("DT unavailable", "Distance transform could not be computed.")
            return False

        self.dt = np.asarray(dt, dtype=np.float32, copy=False)

        dt_u8 = self.analyzer.get_dt_u8()
        if dt_u8 is not None:
            self.gray_u8 = np.asarray(dt_u8, dtype=np.uint8, copy=False)

        dt_norm = self.analyzer.get_dt_norm()
        if dt_norm is not None:
            self.gray_img = np.asarray(dt_norm, dtype=np.float32, copy=False)

        lap = self.analyzer.get_laplacian()
        if lap is not None:
            self.lapl_img = np.asarray(lap, dtype=np.float32, copy=False)

        gx, gy = self.analyzer.get_gradients()
        if gx is not None:
            self.gx = np.asarray(gx, dtype=np.float32, copy=False)
        if gy is not None:
            self.gy = np.asarray(gy, dtype=np.float32, copy=False)

        return True

    def _show_info(self, title: str, message: str):
        if self._batch_mode_active:
            logger.info("[Batch] %s: %s", title, message)
        else:
            QtWidgets.QMessageBox.information(self, title, message)

    def _show_warning(self, title: str, message: str):
        if self._batch_mode_active:
            logger.warning("[Batch] %s: %s", title, message)
        else:
            QtWidgets.QMessageBox.warning(self, title, message)

    def _show_error(self, title: str, message: str):
        if self._batch_mode_active:
            logger.error("[Batch] %s: %s", title, message)
        else:
            QtWidgets.QMessageBox.critical(self, title, message)


    # ------------------- Drag & drop -------------------

    def _install_drag_drop(self):
        v = self.graphicsView
        v.setAcceptDrops(True)
        v.viewport().setAcceptDrops(True)
        v.viewport().installEventFilter(self)

    def eventFilter(self, obj, event):
        if obj is self.graphicsView.viewport():
            et = event.type()
            if et == QEvent.Type.Resize:
                self._update_reset_button_position()
            elif et == QEvent.Type.Wheel:
                if isinstance(event, QtGui.QWheelEvent):
                    if self._handle_wheel_zoom(event):
                        return True
            if getattr(self, "_scalebar_pending", False) and et == QEvent.Type.MouseButtonPress:
                if isinstance(event, QtGui.QMouseEvent):
                    self._handle_scalebar_mouse(event)
                    return True
            if getattr(self, "_profile_mode_active", False) and et == QEvent.Type.MouseButtonPress:
                if isinstance(event, QtGui.QMouseEvent):
                    self._handle_profile_mouse(event)
                    return True
            if et in (QEvent.Type.DragEnter, QEvent.Type.DragMove):
                md = event.mimeData()
                if md and md.hasUrls() and self._urls_are_images(md.urls()):
                    event.acceptProposedAction()
                    return True
            elif et == QEvent.Type.Drop:
                md = event.mimeData()
                if md and md.hasUrls():
                    for u in md.urls():
                        if u.isLocalFile():
                            p = u.toLocalFile()
                            if self._is_image_path(p):
                                self.load_original_from_path(p)
                                event.acceptProposedAction()
                                return True
        return super().eventFilter(obj, event)

    def _is_image_path(self, path: str) -> bool:
        ext = os.path.splitext(path)[1].lower()
        return ext in {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp", ".webp"}

    def _urls_are_images(self, urls) -> bool:
        return any(u.isLocalFile() and self._is_image_path(u.toLocalFile()) for u in urls)

    def _compute_orientation_stats(
        self, angles_deg: np.ndarray
    ) -> tuple[float, float, float]:
        """
        Return orientation index, circular mean angle (deg), and circular std (deg)
        for canonical angles in degrees.
        """
        if angles_deg is None or getattr(angles_deg, "size", 0) == 0:
            return float("nan"), float("nan"), float("nan")

        angles_rad = np.deg2rad(angles_deg)
        mean_cos = float(np.mean(np.cos(angles_rad)))
        mean_sin = float(np.mean(np.sin(angles_rad)))

        orientation_index = float(np.clip(np.hypot(mean_cos, mean_sin), 0.0, 1.0))
        mean_angle_rad = float(np.arctan2(mean_sin, mean_cos))
        mean_angle_deg = float(np.rad2deg(mean_angle_rad))
        if mean_angle_deg > 90.0:
            mean_angle_deg -= 180.0
        elif mean_angle_deg <= -90.0:
            mean_angle_deg += 180.0

        if orientation_index > 0 and np.isfinite(orientation_index):
            r = float(np.clip(orientation_index, 1e-8, 1.0))
            std_angle_rad = float(np.sqrt(max(0.0, -2.0 * np.log(r))))
            std_angle_deg = float(np.rad2deg(std_angle_rad))
        else:
            std_angle_deg = float("nan")

        return orientation_index, mean_angle_deg, std_angle_deg

    def _measurement_export_summary(self) -> tuple[float, float, float, float, float, float, int]:
        """
        Return mean diameter, std, orientation index, mean angle, std angle, porosity, and sample count for CSV export.
        Falls back to recomputing from cached arrays if the summary is missing.
        """
        if self.fiber_summary:
            return (
                float(self.fiber_summary.get("mean_diameter", float("nan"))),
                float(self.fiber_summary.get("std_diameter", float("nan"))),
                float(self.fiber_summary.get("orientation_index", float("nan"))),
                float(self.fiber_summary.get("mean_angle_deg", float("nan"))),
                float(self.fiber_summary.get("std_angle_deg", float("nan"))),
                float(self.fiber_summary.get("porosity", float("nan"))),
                int(self.fiber_summary.get("sample_count", 0)),
            )

        diameters = self._thickness_units if self._thickness_units is not None else np.array([])
        angles = self._angles_deg if self._angles_deg is not None else np.array([])
        mean_val = float(np.mean(diameters)) if diameters.size else float("nan")
        std_val = float(np.std(diameters)) if diameters.size else float("nan")
        sample_count = int(self.samples_table.shape[0]) if self.samples_table is not None else int(diameters.size)
        orientation_index, mean_angle_deg, std_angle_deg = self._compute_orientation_stats(angles)
        porosity = float(self._porosity) if self._porosity is not None else float("nan")
        return mean_val, std_val, orientation_index, mean_angle_deg, std_angle_deg, porosity, sample_count

    # ------------------- I/O -------------------

    def load_original(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select Original Image",
            self.working_dir,
            "Images (*.tif *.tiff *.png *.jpg *.jpeg *.bmp *.webp);;All Files (*)"
        )
        if not path:
            return
        self.load_original_from_path(path)

    def load_original_from_path(self, path: str):
        self._reset_internal_state()
        try:
            img = iio.imread(path)
            self.working_dir = os.path.dirname(path)
        except Exception as e:
            self._show_error("Error", f"Failed to read image:\n{e}")
            logger.error(f"Failed to read image: {e}")
            return
        
        logger.info(f"Image loaded successfully: {path}")

        if img.ndim not in (2, 3):
            self._show_warning("Unsupported", "Original must be 2D or 3D (H×W×C).")
            return
        self.orig_img = img
        self.orig_img_scaled = None
        self.scale_factor = 1.0
        self.current_image_name = os.path.splitext(os.path.basename(path))[0]
        self.refresh_view()

    def save_predictions(self):
        if self.gray_u8 is None and self.labels_img is None and self.overlay_img is None:
            self._show_info("Nothing to save", "Run segmentation first.")
            return
        default_base = (self.current_image_name or "fiberpose") + "_results.png"
        base, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save predictions (choose base name)",
            os.path.join(self.working_dir, default_base),
            "PNG (*.png);;TIFF (*.tif *.tiff)",
        )
        if not base:
            return
        dest_dir = os.path.dirname(base)
        root_name = self.current_image_name or os.path.splitext(os.path.basename(base))[0]
        root = os.path.join(dest_dir, root_name)
        logger.info(f"Saving predictions with base name: {root}")

        try:
            if self.gray_u8 is not None:
                iio.imwrite(root + "_dt.png", np.asarray(self.gray_u8, dtype=np.uint8))
            if self.labels_img is not None and self.labels_img.max() > 0:
                iio.imwrite(root + "_labels.png", labels_to_color(self.labels_img))
            if self.overlay_img is not None:
                iio.imwrite(root + "_overlay.png", self.overlay_img.astype(np.uint8))
            if self.measurements_overlay is not None:
                iio.imwrite(root + "_debug.png", self.measurements_overlay.astype(np.uint8))
            self.working_dir = dest_dir
            self._show_info("Saved", f"Saved to:\n{dest_dir}")
        except Exception as e:
            self._show_error("Error", f"Failed to save:\n{e}")

    def save_measurements(self):
        if self.samples_table is None or len(self.samples_table) == 0:
            self._show_info("Nothing to save", "Run Analyze fibers first.")
            return
        default_csv = (self.current_image_name or "fiberpose") + "_measurements.csv"
        base, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save measurements (CSV)",
            os.path.join(self.working_dir, default_csv),
            "CSV (*.csv)",
        )
        if not base:
            return
        dest_dir = os.path.dirname(base)
        root_name = self.current_image_name or os.path.splitext(os.path.basename(base))[0]
        root = os.path.join(dest_dir, root_name)
        logger.info(f"Saving measurements to base: {root}")

        try:
            with open(root + "_samples.csv", "w", newline="") as f:
                mean_val, std_val, orientation_index, mean_angle_deg, std_angle_deg, porosity, sample_count = self._measurement_export_summary()
                f.write("metric,value\n")
                f.write(f"mean,{mean_val:.6f}\n")
                f.write(f"std,{std_val:.6f}\n")
                f.write(f"orientation_index,{orientation_index:.6f}\n")
                f.write(f"mean_angle_deg,{mean_angle_deg:.6f}\n")
                f.write(f"std_angle_deg,{std_angle_deg:.6f}\n")
                f.write(f"porosity,{porosity:.6f}\n")
                f.write(f"sample_number,{sample_count}\n")
                f.write("\n\n")
                f.write("x,y,diameter,angle_deg,fiber_id\n")

                for idx, row in enumerate(self.samples_table, start=1):
                    x, y, t_px = row[:3]
                    angle_val = float(row[3]) if len(row) > 3 else float("nan")
                    if self._thickness_units is not None and len(self._thickness_units) >= idx:
                        diameter_val = float(self._thickness_units[idx - 1])
                    else:
                        diameter_val = float(t_px)
                    f.write(f"{int(x)},{int(y)},{diameter_val:.6f},{angle_val:.6f},0\n")
        except Exception as e:
            self._show_error("Error", f"Failed to save CSVs:\n{e}")
            return
        self.working_dir = dest_dir
        self._show_info("Saved", f"Saved CSV(s) to:\n{dest_dir}")


    # ------------------- Scalebar -------------------

    def _on_set_scalebar_clicked(self):
        if self.orig_img is None:
            self._show_info("No image", "Load an image before setting the scalebar.")
            return

        self._cancel_scalebar_selection()

        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Set Scalebar Length")
        form = QtWidgets.QFormLayout(dialog)

        spin = QtWidgets.QDoubleSpinBox(dialog)
        spin.setDecimals(2)
        spin.setMinimum(1e-6)
        spin.setMaximum(1e9)
        spin.setValue(float(self._scalebar_length_value or 100.0))
        form.addRow("Length", spin)

        combo = QtWidgets.QComboBox(dialog)
        units = ["nm", "um", "mm", "m"]
        combo.addItems(units)
        if self._scalebar_unit in units:
            combo.setCurrentIndex(units.index(self._scalebar_unit))
        else:
            combo.setCurrentText(self._scalebar_unit)
        form.addRow("Units", combo)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel,
            parent=dialog,
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        form.addRow(buttons)

        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return

        length_value = float(spin.value())
        if length_value <= 0:
            self._show_warning("Invalid length", "Scalebar length must be positive.")
            return

        self._scalebar_length_value = length_value
        self._scalebar_unit = combo.currentText()

        self._scalebar_points = []
        self._scalebar_pending = True
        self.graphicsView.viewport().setCursor(Qt.CursorShape.CrossCursor)
        self._show_info(
            "Select Scalebar",
            "Click the two endpoints of the scalebar on the image. Right-click to cancel.",
        )

    def _handle_scalebar_mouse(self, event: QtGui.QMouseEvent):
        if event.button() == Qt.MouseButton.RightButton:
            self._cancel_scalebar_selection()
            return
        if event.button() != Qt.MouseButton.LeftButton:
            return
        if self._pixitem.pixmap().isNull():
            return

        pos = event.position() if hasattr(event, "position") else event.localPos()
        scene_pos = self.graphicsView.mapToScene(QPoint(int(round(pos.x())), int(round(pos.y()))))
        x = float(scene_pos.x())
        y = float(scene_pos.y())

        pix = self._pixitem.pixmap()
        width = pix.width()
        height = pix.height()
        if x < 0 or y < 0 or x >= width or y >= height:
            return

        self._scalebar_points.append((x, y))
        if len(self._scalebar_points) == 2:
            self._finalize_scalebar_selection()

    def _finalize_scalebar_selection(self):
        if len(self._scalebar_points) < 2:
            return
        (x0, y0), (x1, y1) = self._scalebar_points[:2]
        pixel_length = math.hypot(x1 - x0, y1 - y0)
        if pixel_length <= 0:
            self._show_warning("Invalid selection", "Select two distinct points for the scalebar.")
            self._cancel_scalebar_selection()
            return

        pixel_size = self._scalebar_length_value / pixel_length
        self.le_pixelsize.setText(f"{pixel_size:.6g}")
        index = self.cb_units.findText(self._scalebar_unit)
        if index is not None and index >= 0:
            self.cb_units.setCurrentIndex(index)
        else:
            self.cb_units.setCurrentText(self._scalebar_unit)

        self._show_info(
            "Scalebar Set",
            f"Pixel size set to {pixel_size:.6g} {self._scalebar_unit} per pixel.",
        )
        self._cancel_scalebar_selection()

    def _cancel_scalebar_selection(self, reset_cursor: bool = True, clear_points: bool = True):
        self._scalebar_pending = False
        if clear_points:
            self._scalebar_points = []
        if reset_cursor:
            self.graphicsView.viewport().unsetCursor()

    # ------------------- Zoom helpers -------------------

    def reset_zoom(self):
        self._fit_view_to_pixmap(force=True)

    def _update_reset_button_position(self):
        viewport = self.graphicsView.viewport()
        if viewport is None:
            return
        margin = 8
        hint = self.btn_reset_view.sizeHint()
        x = viewport.width() - hint.width() - margin
        y = margin
        self.btn_reset_view.move(max(margin, x), y)

    def _update_reset_button_visibility(self):
        self.btn_reset_view.setVisible(self._zoom_mode_active)

    def _handle_wheel_zoom(self, event: QtGui.QWheelEvent) -> bool:
        if self._pixitem.pixmap().isNull():
            return False
        angle = event.angleDelta().y()
        if angle == 0:
            return False
        zoom_factor = 1.25 if angle > 0 else 0.8
        current_scale = self.graphicsView.transform().m11()
        if current_scale == 0:
            current_scale = 1.0
        target_scale = current_scale * zoom_factor
        base = self._base_scale if self._base_scale not in (0.0, None) else 1.0
        relative = target_scale / base
        if relative < self._min_zoom_factor or relative > self._max_zoom_factor:
            return True
        was_anchor = self.graphicsView.transformationAnchor()
        self.graphicsView.setTransformationAnchor(QtWidgets.QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.graphicsView.scale(zoom_factor, zoom_factor)
        self.graphicsView.setTransformationAnchor(was_anchor)
        self._zoom_mode_active = True
        self._current_zoom = relative
        self._update_reset_button_visibility()
        return True

    # ------------------- Batch processing -------------------

    @staticmethod
    def _supported_image_extensions() -> set[str]:
        return {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp", ".webp"}

    def _collect_images_in_directory(self, directory: str) -> list[str]:
        files: list[str] = []
        try:
            entries = sorted(os.listdir(directory))
        except Exception as exc:
            logger.error("Failed to list directory %s: %s", directory, exc)
            return files
        exts = self._supported_image_extensions()
        for name in entries:
            path = os.path.join(directory, name)
            if not os.path.isfile(path):
                continue
            if os.path.splitext(name)[1].lower() in exts:
                files.append(path)
        return files

    def _save_batch_outputs(
        self,
        dest_dir: str,
        base_name: str,
        save_options: Dict[str, bool] | None = None,
    ):
        """Persist batch artifacts according to the save options selected by the user."""
        options = save_options or {}

        def wants(key: str) -> bool:
            return bool(options.get(key, True))

        if wants("dt") and self.gray_u8 is not None:
            dt_path = os.path.join(dest_dir, f"{base_name}_dt.png")
            iio.imwrite(dt_path, np.asarray(self.gray_u8, dtype=np.uint8))
        if wants("labels") and self.labels_img is not None and np.max(self.labels_img) > 0:
            labels_path = os.path.join(dest_dir, f"{base_name}_labels.png")
            iio.imwrite(labels_path, labels_to_color(self.labels_img))
        if wants("overlay") and self.overlay_img is not None:
            overlay_path = os.path.join(dest_dir, f"{base_name}_overlay.png")
            iio.imwrite(overlay_path, np.asarray(self.overlay_img, dtype=np.uint8))
        if wants("debug") and self.measurements_overlay is not None:
            debug_path = os.path.join(dest_dir, f"{base_name}_debug.png")
            iio.imwrite(debug_path, np.asarray(self.measurements_overlay, dtype=np.uint8))

        if wants("samples") and self.samples_table is not None and len(self.samples_table) > 0:
            samples_path = os.path.join(dest_dir, f"{base_name}_samples.csv")
            with open(samples_path, "w", newline="") as f:
                mean_val, std_val, orientation_index, mean_angle_deg, std_angle_deg, porosity, sample_count = self._measurement_export_summary()
                f.write("metric,value\n")
                f.write(f"mean,{mean_val:.6f}\n")
                f.write(f"std,{std_val:.6f}\n")
                f.write(f"orientation_index,{orientation_index:.6f}\n")
                f.write(f"mean_angle_deg,{mean_angle_deg:.6f}\n")
                f.write(f"std_angle_deg,{std_angle_deg:.6f}\n")
                f.write(f"porosity,{porosity:.6f}\n")
                f.write(f"sample_number,{sample_count}\n")
                f.write("\n\n")
                f.write("x,y,diameter,angle_deg,fiber_id\n")

                for idx, row in enumerate(self.samples_table, start=1):
                    x, y, t_px = row[:3]
                    angle_val = float(row[3]) if len(row) > 3 else float("nan")
                    if self._thickness_units is not None and len(self._thickness_units) >= idx:
                        diameter_val = float(self._thickness_units[idx - 1])
                    else:
                        diameter_val = float(t_px)
                    f.write(f"{int(x)},{int(y)},{diameter_val:.6f},{angle_val:.6f},0\n")

    def _prompt_batch_save_options(self) -> Dict[str, bool] | None:
        dialog = BatchSaveOptionsDialog(self, self._batch_save_options)
        result = dialog.exec()
        if result == QtWidgets.QDialog.DialogCode.Accepted:
            options = dialog.options()
            self._batch_save_options = options
            return options
        return None

    def _on_batch_processing(self):
        directory = QFileDialog.getExistingDirectory(self, "Select folder with images", self.working_dir)
        if not directory:
            return

        image_paths = self._collect_images_in_directory(directory)
        if not image_paths:
            self._show_info("Batch processing", "No supported images were found in the selected folder.")
            return

        save_options = self._prompt_batch_save_options()
        if not save_options:
            return

        prev_strict_mode = self._strict_centerline_enabled()
        prev_seg_mode = self._current_segmentation_mode()
        batch_strict_mode = bool(save_options.get("strict", prev_strict_mode))
        self._set_strict_mode(batch_strict_mode)
        if batch_strict_mode and prev_seg_mode != "ridge":
            if getattr(self, "rb_ridge", None):
                self.rb_ridge.setChecked(True)
            else:
                self.segmentation_mode = "ridge"
                self._sync_pipeline_config_from_ui()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_root = os.path.join(directory, f"fiberpose_batch_{timestamp}")
        os.makedirs(output_root, exist_ok=True)

        previous_batch_state = self._batch_mode_active
        self._batch_mode_active = True
        QtWidgets.QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)

        failures: list[tuple[str, str]] = []
        processed = 0

        try:
            for idx, path in enumerate(image_paths, start=1):
                base_name = os.path.splitext(os.path.basename(path))[0]
                self.statusbar.showMessage(f"Processing {idx}/{len(image_paths)}: {os.path.basename(path)}")
                try:
                    self.load_original_from_path(path)
                    QtWidgets.QApplication.processEvents()

                    self.segment()
                    QtWidgets.QApplication.processEvents()
                    if self.gray_u8 is None or self.labels_img is None:
                        failures.append((path, "Segmentation did not produce results"))
                        continue

                    self.analyze_fibers()
                    QtWidgets.QApplication.processEvents()

                    self._save_batch_outputs(output_root, base_name, save_options)
                    processed += 1
                except Exception as exc:
                    logger.exception("Batch processing failed for %s", path)
                    failures.append((path, str(exc)))
        finally:
            self._set_strict_mode(prev_strict_mode)
            if prev_seg_mode != self._current_segmentation_mode():
                if prev_seg_mode == "ridge" and getattr(self, "rb_ridge", None):
                    self.rb_ridge.setChecked(True)
                elif prev_seg_mode == "watershed" and getattr(self, "rb_watershed", None):
                    self.rb_watershed.setChecked(True)
                else:
                    self.segmentation_mode = prev_seg_mode
                    self._sync_pipeline_config_from_ui()
            QtWidgets.QApplication.restoreOverrideCursor()
            self._batch_mode_active = previous_batch_state
            self.statusbar.clearMessage()

        summary_lines = [
            f"Processed {processed} of {len(image_paths)} images.",
            f"Results saved in: {output_root}",
        ]
        if failures:
            summary_lines.append("Failures:")
            for path, reason in failures[:10]:
                summary_lines.append(f"  - {os.path.basename(path)}: {reason}")
            if len(failures) > 10:
                summary_lines.append(f"  ... ({len(failures) - 10} more)")

        summary = "\n".join(summary_lines)
        if failures:
            self._show_warning("Batch processing complete with issues", summary)
        else:
            self._show_info("Batch processing complete", summary)

    # ------------------- Profile plot tool -------------------

    def _ensure_profile_dialog(self) -> ProfilePlotDialog:
        if self._profile_dialog is None:
            self._profile_dialog = ProfilePlotDialog(self)
            self._profile_dialog.clearRequested.connect(self._on_profile_clear_requested)
            self._profile_dialog.closed.connect(self._on_profile_dialog_closed)
        return self._profile_dialog

    def _on_plot_profile_clicked(self):
        if not self._ensure_dt_ready():
            return
        dialog = self._ensure_profile_dialog()
        if not dialog.isVisible():
            dialog.reset_plot()
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()
        self._profile_mode_active = True
        self._reset_profile_selection(keep_dialog=True)

    def _on_profile_clear_requested(self):
        self._reset_profile_selection(keep_dialog=True)

    def _on_profile_dialog_closed(self):
        self._profile_mode_active = False
        self._profile_dialog = None
        self._reset_profile_selection()

    def _reset_profile_selection(self, keep_dialog: bool = False):
        self._profile_points_scene = []
        self._profile_points_img = []
        self._remove_profile_items()
        dialog = self._profile_dialog
        if dialog is not None and (keep_dialog or self._profile_mode_active):
            dialog.reset_plot()
        if self._profile_mode_active:
            self.graphicsView.viewport().setCursor(Qt.CursorShape.CrossCursor)
        else:
            self.graphicsView.viewport().unsetCursor()

    def _remove_profile_line(self):
        if self._profile_line_item is not None and self._profile_line_item.scene() is not None:
            self._scene.removeItem(self._profile_line_item)
        self._profile_line_item = None

    def _remove_profile_items(self):
        self._remove_profile_line()
        if not self._profile_point_items:
            self._profile_point_items = []
            return
        for item in self._profile_point_items:
            if item is not None and item.scene() is not None:
                self._scene.removeItem(item)
        self._profile_point_items = []

    def _add_profile_marker(self, x: float, y: float):
        if self._scene is None:
            return
        radius = 4.0
        marker = QGraphicsEllipseItem(x - radius, y - radius, 2 * radius, 2 * radius)
        pen = QtGui.QPen(QtGui.QColor("#FFD966"))
        pen.setWidth(2)
        marker.setPen(pen)
        marker.setBrush(QtGui.QBrush(QtGui.QColor(255, 217, 102, 120)))
        marker.setZValue(1000)
        self._scene.addItem(marker)
        self._profile_point_items.append(marker)

    def _draw_profile_line(self):
        if len(self._profile_points_scene) < 2 or self._scene is None:
            return
        self._remove_profile_line()
        (x0, y0), (x1, y1) = self._profile_points_scene[:2]
        pen = QtGui.QPen(QtGui.QColor("#FAD000"))
        pen.setWidth(2)
        pen.setCosmetic(True)
        line_item = self._scene.addLine(QLineF(x0, y0, x1, y1), pen)
        line_item.setZValue(999)
        self._profile_line_item = line_item

    def _scene_to_image_coords(self, scene_x: float, scene_y: float) -> tuple[float, float] | None:
        if self._pixmap_source_shape is None:
            return None
        pix = self._pixitem.pixmap()
        if pix.isNull():
            return None
        display_w = float(pix.width())
        display_h = float(pix.height())
        if display_w <= 0 or display_h <= 0:
            return None
        src_h, src_w = self._pixmap_source_shape
        scale_x = float(src_w) / display_w
        scale_y = float(src_h) / display_h
        return scene_x * scale_x, scene_y * scale_y

    def _handle_profile_mouse(self, event: QtGui.QMouseEvent):
        if event.button() == Qt.MouseButton.RightButton:
            self._reset_profile_selection(keep_dialog=True)
            return
        if event.button() != Qt.MouseButton.LeftButton:
            return
        if self._pixitem.pixmap().isNull():
            return

        pos = event.position() if hasattr(event, "position") else event.localPos()
        scene_pos = self.graphicsView.mapToScene(QPoint(int(round(pos.x())), int(round(pos.y()))))
        sx = float(scene_pos.x())
        sy = float(scene_pos.y())

        pix = self._pixitem.pixmap()
        if sx < 0 or sy < 0 or sx >= pix.width() or sy >= pix.height():
            return

        img_coords = self._scene_to_image_coords(sx, sy)
        if img_coords is None:
            return

        ix, iy = img_coords
        if self._pixmap_source_shape is None:
            return
        src_h, src_w = self._pixmap_source_shape
        if ix < 0 or iy < 0 or ix >= src_w or iy >= src_h:
            return

        if len(self._profile_points_scene) >= 2:
            self._reset_profile_selection(keep_dialog=True)

        self._profile_points_scene.append((sx, sy))
        self._profile_points_img.append((ix, iy))
        self._add_profile_marker(sx, sy)

        if len(self._profile_points_scene) == 2:
            self._finalize_profile_selection()

    def _finalize_profile_selection(self):
        if len(self._profile_points_scene) < 2 or len(self._profile_points_img) < 2:
            return
        self._draw_profile_line()
        self._update_profile_plot()

    def _update_profile_plot(self):
        if self.dt is None or len(self._profile_points_img) < 2:
            return
        (x0, y0), (x1, y1) = self._profile_points_img[:2]
        length_px = float(np.hypot(x1 - x0, y1 - y0))
        if length_px <= 0:
            return
        profile_vals = profile_line(self.dt, (y0, x0), (y1, x1), mode="reflect", order=1)
        if profile_vals is None or profile_vals.size == 0:
            return

        profile_vals = np.asarray(profile_vals, dtype=np.float32, copy=False)
        scale = getattr(self, "scale_factor", 1.0)
        if scale is None or not np.isfinite(scale) or scale <= 0:
            scale = 1.0
        true_length_px = length_px / scale
        distances = np.linspace(0.0, true_length_px, profile_vals.size, dtype=np.float32)
        delta = float(profile_vals[-1] - profile_vals[0])
        summary = (
            f"Length: {true_length_px:.2f} px · ΔDT: {delta:+.2f} · "
            f"Min/Max: {float(profile_vals.min()):.2f} / {float(profile_vals.max()):.2f}"
        )
        dialog = self._ensure_profile_dialog()
        dialog.update_profile(distances, profile_vals, summary)

    # ------------------- Segmentation -------------------

    def segment(self):
        if self.orig_img is None:
            self._show_info("Load original", "Please load an original image first.")
            logger.info("Segment: No original image loaded.")
            return
        if self.analyzer is None:
            self._show_error("Analyzer unavailable", "Fiber analyzer has not been initialized.")
            logger.error("Segmentation failed: analyzer not initialized.")
            return

        previous_scale = getattr(self, "scale_factor", 1.0)
        target_scale = self._current_scale_factor()
        scale_changed = not math.isclose(previous_scale, target_scale, rel_tol=1e-6, abs_tol=1e-6)
        processing_img = self.orig_img
        if math.isclose(target_scale, 1.0, rel_tol=1e-6, abs_tol=1e-6):
            self.scale_factor = 1.0
            self.orig_img_scaled = None
        else:
            if self.orig_img_scaled is None or scale_changed:
                interp = cv2.INTER_AREA if target_scale < 1.0 else cv2.INTER_CUBIC
                try:
                    self.orig_img_scaled = cv2.resize(
                        self.orig_img,
                        None,
                        fx=target_scale,
                        fy=target_scale,
                        interpolation=interp,
                    )
                except Exception as e:
                    self._show_error("Scaling failed", f"Unable to scale image:\n{e}")
                    logger.error("Scaling failed: %s", e)
                    return
            processing_img = self.orig_img_scaled
            self.scale_factor = float(target_scale)
        self.analyzer.scale_factor = self.scale_factor
        if scale_changed:
            self.analyzer.reset_prediction_state()

        self._sync_pipeline_config_from_ui()
        config = self.pipeline_config
        ridge_cfg = config.get("ridge_detection", {})
        seg_cfg = config.get("segmentation", {})
        watershed_cfg = seg_cfg.get("watershed", {}) if isinstance(seg_cfg, dict) else {}
        high_conf_cfg = seg_cfg.get("high_confidence", {}) if isinstance(seg_cfg, dict) else {}

        max_angle = float(ridge_cfg.get("max_angle_deg", self._current_max_angle()))
        centerline_params = dict(
            ridge_cfg.get("centerline_kwargs", FiberAnalyzer._default_centerline_kwargs())
        )
        if not centerline_params:
            params = getattr(self.analyzer, "_centerline_kwargs", None)
            if params:
                centerline_params = dict(params)
            else:
                centerline_params = FiberAnalyzer._default_centerline_kwargs()

        self.progress_bar.setValue(0)

        if not self.analyzer.dt_predicted():
            logger.info("Starting Distance transform prediction.")
            try:
                self.analyzer.predict_dt(
                    image=processing_img,
                    max_angle_deg=max_angle,
                    centerline_kwargs=centerline_params,
                )
            except Exception as e:
                self._show_error("Prediction failed", str(e))
                logger.error("Segmentation failed: %s", e)
                return

        self.progress_bar.setValue(40)

        try:
            self.analyzer.compute_centerlines(
                max_angle_deg=max_angle,
                **centerline_params,
            )
        except Exception as e:
            self._show_error("Centerline computation failed", str(e))
            logger.error("Centerline computation failed: %s", e)
            return

        self.progress_bar.setValue(50)

        dt = self.analyzer.get_dt()
        gx, gy = self.analyzer.get_gradients()
        if dt is None or gx is None or gy is None:
            self._show_error("Segmentation failed", "Distance transform or gradients unavailable.")
            logger.error("Segmentation failed: DT or gradients unavailable.")
            return

        self.dt = np.asarray(dt, dtype=np.float32, copy=False)
        self.gx = np.asarray(gx, dtype=np.float32, copy=False)
        self.gy = np.asarray(gy, dtype=np.float32, copy=False)
        self.gray_u8 = self.analyzer.get_dt_u8()
        self.gray_img = self.analyzer.get_dt_norm()
        self.lapl_img = self.analyzer.get_laplacian()

        thr_u8 = int(watershed_cfg.get("threshold_u8", self._current_segmentation_threshold()))
        mode = self._current_segmentation_mode()
        ridge_mode = mode == "ridge"
        radius_thresh_px = None
        ridge_options = None
        if ridge_mode:
            current_scale = self.scale_factor if self.scale_factor and np.isfinite(self.scale_factor) else 1.0
            radius_thresh_px_val = high_conf_cfg.get("radius_thresh_px")
            if radius_thresh_px_val is None:
                radius_thresh_px_val = self._current_radius_threshold()
            radius_thresh_px = float(radius_thresh_px_val) if radius_thresh_px_val is not None else None
            if radius_thresh_px is not None:
                radius_thresh_px *= current_scale
            ridge_options = {
                "dilate_extra_px": float(high_conf_cfg.get("dilate_extra_px", 0.0)) * current_scale,
                "min_run_points": int(high_conf_cfg.get("min_run_points", 1)),
            }
        try:
            if ridge_mode:
                if radius_thresh_px is None:
                    raise ValueError("Ridge segmentation requires a radius threshold.")
                self.labels_img = self.analyzer.segment_strict_ridge(
                    radius_thresh_px=radius_thresh_px,
                    dilate_extra_px=float(ridge_options.get("dilate_extra_px", 0.0)) if ridge_options else 0.0,
                    min_run_points=int(ridge_options.get("min_run_points", 1)) if ridge_options else 1,
                )
            else:
                self.labels_img = self.analyzer.segment_watershed(thr_u8=thr_u8)
        except Exception as e:
            self._show_error("Segmentation failed", str(e))
            logger.error("Segmentation failed: %s", e)
            return

        if self.labels_img is not None:
            labels_rgb = labels_to_color(self.labels_img)
            self._labels_rgb_cache = (id(self.labels_img), labels_rgb)
        else:
            labels_rgb = None
            self._labels_rgb_cache = (None, None)

        base_img = self.orig_img if self.scale_factor == 1.0 or self.orig_img_scaled is None else self.orig_img_scaled
        self.overlay_img = self._create_overlay(base_img, self.labels_img, alpha=0.5, labels_rgb=labels_rgb)
        self._centerlines_dt = self.analyzer.get_centerlines_image()

        if self.cb_view.currentText() not in ["Overlay", "Segmentation", "Measurements overlay", "Sampling"]:
            self.cb_view.setCurrentText("Overlay")

        self.action_save_masks_predictions.setEnabled(True)

        self.refresh_view()
        self.progress_bar.setValue(70)
        logger.info("Segmentation completed successfully.")

    def analyze_fibers(self):
        """
        Computes thickness samples via analyzer and renders overlays/plots.
        """
        self._sync_pipeline_config_from_ui()
        seg_cfg = self.pipeline_config.get("segmentation", {})
        target_scale = self._current_scale_factor()
        current_scale = getattr(self, "scale_factor", 1.0)
        scale_mismatch = (
            self.analyzer is not None
            and self.labels_img is not None
            and self.analyzer.dt_predicted()
            and not math.isclose(current_scale, target_scale, rel_tol=1e-6, abs_tol=1e-6)
        )
        if scale_mismatch or self.gray_u8 is None or self.labels_img is None:
            if scale_mismatch:
                logger.info(
                    "Scale factor changed from %.3f to %.3f; rerunning segmentation for analysis.",
                    current_scale,
                    target_scale,
                )
            self.segment()
            if self.gray_u8 is None or self.labels_img is None:
                self._show_info("No segmentation", "Segmentation could not be performed.")
                return
        logger.info("Starting fiber analysis.")
        self._viz_points = []

        # Params
        def _getf(line, default):
            txt = line.text().strip()
            try:
                return float(txt) if txt else default
            except Exception:
                return default
        spacing_px_input = max(1.0, float(_getf(self.le_samspacing, 5)))
        current_scale = self.scale_factor if self.scale_factor and np.isfinite(self.scale_factor) else 1.0
        spacing_px_scaled = max(1.0, spacing_px_input * current_scale)
        pixel_size  = float(_getf(self.le_pixelsize, 1.0))
        units_label = self.cb_units.currentText() if self.cb_units.currentText().strip() and self.le_pixelsize.text().strip() else "px"

        # compute via analyzer
        if self.analyzer is None:
            self._show_info("No analyzer", "Segmentation data unavailable.")
            return

        if not self.analyzer.get_centerline_segments():
            self._show_info("No centerlines", "No centerlines available. Run segmentation first.")
            return

        self.analyzer.scale_factor = self.scale_factor

        hist_centers, viz_points, angles_deg = self.analyzer.analyze(
            spacing_px=spacing_px_scaled
        )
        self._viz_points = viz_points

        self.progress_bar.setValue(80)

        # Build sample table and stats (keep in GUI)
        if hist_centers:
            radii_px        = np.array([c[2] for c in hist_centers], dtype=np.float32)
            thickness_px    = 2.0 * radii_px
            thickness_units = thickness_px * pixel_size
            xs = np.array([c[1] for c in hist_centers], dtype=np.int32)
            ys = np.array([c[0] for c in hist_centers], dtype=np.int32)
            self.samples_table = np.column_stack([xs, ys, thickness_px, np.array(angles_deg, dtype=np.float32)]).astype(object)
        else:
            thickness_units = np.array([], dtype=np.float32)
            self.samples_table = None

        self._thickness_units = thickness_units
        self._angles_deg = np.array([a for a in angles_deg if np.isfinite(a)], dtype=np.float32)


        # --- pore splitting + sampling ------------------------------------------
        watershed_cfg = seg_cfg.get("watershed", {}) if isinstance(seg_cfg, dict) else {}
        thr_u8 = int(watershed_cfg.get("threshold_u8", self._current_segmentation_threshold()))

        # Build void mask from your chosen source (DT or grayscale-based threshold).
        # If you already have self._void_mask, reuse it.
        void_mask_u8 = getattr(self, "_void_mask", None)

        if void_mask_u8 is None:
            # If your pipeline stores grayscale in self.gray_u8, pass that DT/threshold you use
            self._void_mask, self._pore_dt = self.analyzer.void_mask_and_pore_dt(thr_u8)

        self._porosity = float(self._void_mask.mean())

        # --- NEW: sample pores as inscribed circles from DT peaks ---------------------
        min_radius_px = 1.5 * current_scale   # tweakable; ignores tiny specks
        peak_footprint     = 5     # local-max window; 3..7 works; larger -> fewer peaks
        max_pores     = None  # or set an upper cap for speed/UI clarity
        min_peak_px = 3.0 * current_scale

        self._pores = self.analyzer.sample_inscribed_circles(
            pore_dt=self._pore_dt,
            void_mask=self._void_mask,
            min_radius_px=min_radius_px,
            min_peak_px=min_peak_px,          # <- NEW param
            peak_footprint=peak_footprint,
            max_pores=max_pores,
        )

        # convenience arrays for hist/units
        pixel_size  = float(_getf(self.le_pixelsize, 1.0))
        pore_scale = self.scale_factor if self.scale_factor and np.isfinite(self.scale_factor) else 1.0
        if pore_scale == 0:
            pore_scale = 1.0
        self._pore_diams_units = np.array(
            [2.0 * p.get('r_px_orig', p['r_px'] / pore_scale) * pixel_size for p in self._pores],
            dtype=np.float32,
        )




        # Summary text
        if self._thickness_units.size:
            mean_thickness = np.mean(self._thickness_units)
            std_thickness  = np.std(self._thickness_units)
        else:
            mean_thickness = float('nan')
            std_thickness  = float('nan')
        orientation_index, mean_angle_deg, std_angle_deg = self._compute_orientation_stats(self._angles_deg)

        info_text = (
            f"Mean diameter: {mean_thickness:.2f} {units_label}\n"
            f"Std diameter: {std_thickness:.2f} {units_label}\n"
            f"Orientation index: {orientation_index:.2f}\n"
            f"Mean angle: {mean_angle_deg:.2f} deg\n"
            f"Std angle: {std_angle_deg:.2f} deg\n"
            f"Porosity (area fraction): {self._porosity*100:.1f} %"
        )
        self.fiber_summary = {
            "mean_diameter": float(mean_thickness),
            "std_diameter": float(std_thickness),
            "orientation_index": float(orientation_index),
            "mean_angle_deg": float(mean_angle_deg),
            "std_angle_deg": float(std_angle_deg),
            "porosity": float(self._porosity),
            "sample_count": len(hist_centers) if hist_centers else 0,
            "units": units_label,
        }
        self.text_edit2.setPlainText(info_text)

        # Overlay + plots
        self._render_vector_overlay()
        self._update_distributions_pyqtgraph(units_label)
        self.progress_bar.setValue(100)

        self._show_info(
            "Analysis complete",
            f"Histogram samples: {len(hist_centers)} (spacing={int(spacing_px_input)} px)\n"
            f"Stored viz points: {len(self._viz_points)}\n"
            f"Mean diameter: {mean_thickness:.2f} {units_label}\n"
            f"Std diameter: {std_thickness:.2f} {units_label}\n"
            f"Orientation index: {orientation_index:.2f}\n"
            f"Mean angle: {mean_angle_deg:.2f} deg\n"
            f"Std angle: {std_angle_deg:.2f} deg"
        )

        self.action_save_measurements.setEnabled(True)
        logger.info("Fiber analysis completed successfully.")

    # ---- single overlay renderer (kept once) ----
    def _render_vector_overlay(self):
        if not getattr(self, "_viz_points", None):
            return
        show_diam   = getattr(self, "rb_show_diam", None) and self.rb_show_diam.isChecked()
        show_direct = getattr(self, "rb_show_direct", None) and self.rb_show_direct.isChecked()
        show_pores  = getattr(self, "rb_show_pores", None) and self.rb_show_pores.isChecked()

        if not show_diam and not show_direct and not show_pores:
            show_diam = True

        if show_pores:
            base = self._pore_dt_to_rgb(self._pore_dt, self._void_mask)
            if getattr(self, "_pores", None):
                out = base.copy()
                for p in self._pores:
                    x, y = int(p['x']), int(p['y'])
                    r = int(round(p['r_px']))
                    if r <= 0:
                        continue
                    self._draw_circle(out, (x, y), r, color=(255, 255, 255), thickness=1)  
                    # optional tiny center mark for readability (kept 1 px too)
                    # draw_circle(out, (x, y), max(1, r // 12), color=(0, 255, 255), thickness=-1)
                self._set_pixmap_from_array(out)
                return

        sampling_base = ensure_rgb_u8(self.gray_u8 if self.gray_u8 is not None else self.gray_img)
        if sampling_base is None and self.overlay_img is not None:
            sampling_base = ensure_rgb_u8(self.overlay_img)
        if sampling_base is None:
            return

        base_img = self.orig_img if self.scale_factor == 1.0 or self.orig_img_scaled is None else self.orig_img_scaled
        measurement_base = self.overlay_img.copy() if self.overlay_img is not None else ensure_rgb_u8(base_img)
        sampling_canvas = sampling_base.copy()
        measurement_canvas = measurement_base.copy() if measurement_base is not None else None
        if measurement_canvas is None:
            measurement_canvas = sampling_canvas.copy()

        H, W = sampling_canvas.shape[:2]
        MAX_SEGMENTS = 15000
        pts = self._viz_points if len(self._viz_points) <= MAX_SEGMENTS else self._viz_points[::int(np.ceil(len(self._viz_points)/MAX_SEGMENTS))]

        line_color = (255, 255, 0)
        point_color = (255, 0, 0)
        for (y, x, r, ny, nx, ty, tx) in pts:
            if not (0 <= y < H and 0 <= x < W) or r <= 0:
                continue
            if show_diam:
                p0 = (y - ny * r, x - nx * r)
                p1 = (y + ny * r, x + nx * r)
                self._draw_line(sampling_canvas, p0, p1, color=line_color)
                self._draw_line(measurement_canvas, p0, p1, color=line_color)
            else:
                self._draw_arrow_centered(
                    sampling_canvas,
                    float(y),
                    float(x),
                    float(tx),
                    float(ty),
                    length_px=2.0 * r,
                    color=line_color,
                )
                self._draw_arrow_centered(
                    measurement_canvas,
                    float(y),
                    float(x),
                    float(tx),
                    float(ty),
                    length_px=2.0 * r,
                    color=line_color,
                )
            self._draw_circle(
                sampling_canvas,
                (int(round(x)), int(round(y))),
                radius_px=1,
                color=point_color,
                thickness=-1,
            )

        self.sampling_overlay = sampling_canvas
        self.measurements_overlay = measurement_canvas
        # Default to the new overlay view if user is on an unrelated tab
        if self.cb_view.currentText() not in ("Measurements overlay", "Sampling"):
            self.cb_view.setCurrentText("Measurements overlay")
        self.refresh_view()

    def _clip_point(self, y, x, H, W):
        """Clip float (y,x) to valid integer image coordinates."""
        y = int(round(float(y))); x = int(round(float(x)))
        if y < 0: y = 0
        if x < 0: x = 0
        if y >= H: y = H - 1
        if x >= W: x = W - 1
        return y, x

    def _draw_line(self, img, p0, p1, color=(255, 255, 0), thickness=1):
        """Draw a line on img using points p0,p1 in (y,x) coordinates."""
        if img is None:
            return
        H, W = img.shape[:2]
        y0, x0 = p0
        y1, x1 = p1
        y0, x0 = self._clip_point(y0, x0, H, W)
        y1, x1 = self._clip_point(y1, x1, H, W)
        cv2.line(img, (x0, y0), (x1, y1), color, int(thickness), cv2.LINE_AA)

    def _draw_arrow_centered(self, img, y, x, tx, ty, length_px=10.0, color=(255, 255, 0), thickness=1, tip=0.25):
        """
        Draw an arrow centered at (y,x) along tangent (tx,ty) with length length_px.
        tx,ty can be non-normalized; function normalizes internally.
        """
        if img is None or length_px <= 0:
            return
        vlen = float(np.hypot(tx, ty))
        if not np.isfinite(vlen) or vlen <= 1e-6:
            return
        tx /= vlen; ty /= vlen
        half = 0.5 * float(length_px)
        p0 = (y - ty * half, x - tx * half)
        p1 = (y + ty * half, x + tx * half)
        H, W = img.shape[:2]
        y0, x0 = self._clip_point(*p0, H, W)
        y1, x1 = self._clip_point(*p1, H, W)
        cv2.arrowedLine(img, (x0, y0), (x1, y1), color, int(thickness), tipLength=float(tip))


    def _draw_circle(self, img: np.ndarray, center_xy: tuple[int, int], radius_px: int,
                    color: tuple[int, int, int] = (255, 255, 0),  # yellow 
                    thickness: int = 1,
                    lineType: int = cv2.LINE_AA) -> None:
        """
        Draw a circle in-place on `img`, consistent with draw_line/draw_arrow helpers.
        center_xy: (x, y), radius_px: int pixels.
        """
        cv2.circle(img, center_xy, int(radius_px), color, thickness=thickness, lineType=lineType)

    def _pore_dt_to_rgb(self, pore_dt: np.ndarray, void_mask_u8: np.ndarray) -> np.ndarray:
        """
        Returns an RGB uint8 image where the void region shows the pore DT
        as a TURBO colormap (normalized robustly), and fibers are black.
        """
        vm = (void_mask_u8.astype(np.uint8) > 0)
        norm = np.zeros_like(pore_dt, dtype=np.float32)
        if np.any(vm):
            # robust upper bound to avoid blown highlights
            vmax = float(np.percentile(pore_dt[vm], 99.0))
            if vmax <= 0:
                vmax = float(np.max(pore_dt))
            if vmax > 0:
                norm[vm] = (pore_dt[vm] / vmax).clip(0.0, 1.0)
        base_gray = (norm * 255.0).astype(np.uint8)
        base_rgb = cv2.applyColorMap(base_gray, cv2.COLORMAP_TURBO)
        # zero out fibers for clarity
        base_rgb[~vm] = (0, 0, 0)
        return base_rgb


    # ------------------- Plotting -------------------

    def _style_plotwidget(self, pw: pg.PlotWidget, title: str, xlab: str, ylab: str):
        pw.setBackground("w")
        pw.clear()
        pw.setTitle(title, color="k", size="12pt")
        pw.showGrid(x=True, y=True, alpha=0.25)
        ax = pw.getAxis("bottom"); ax.setPen("k"); ax.setTextPen("k"); ax.setLabel(xlab, color="k")
        ay = pw.getAxis("left");   ay.setPen("k"); ay.setTextPen("k"); ax.setLabel(ylab, color="k")
        pw.setAntialiasing(True)

    def _update_distributions_pyqtgraph(self, units_label: str):
        # Thickness
        self._style_plotwidget(self.graph_diameters, "Fiber thickness", "", f"Thickness ({units_label})")
        tu = getattr(self, "_thickness_units", np.array([], dtype=np.float32))
        if tu.size:
            hist, edges = np.histogram(tu, bins=40)
            centers = 0.5 * (edges[:-1] + edges[1:])
            widths  = (edges[1:] - edges[:-1])
            # Thickness (blue)
            bg = pg.BarGraphItem(
                x=centers, height=hist, width=widths,
                brush=pg.mkBrush("#4472C4"),      # blue fill
                pen=pg.mkPen(color="#2F5597", width=1)  # darker blue edge
            )
            self.graph_diameters.addItem(bg)

        show_pores = getattr(self, "rb_show_pores", None) and self.rb_show_pores.isChecked()

        if not show_pores:
            # Angles (-90..90)
            self._style_plotwidget(self.graph_angles, "Fiber orientation", "", "Angle (deg, canonical)")
            ang = getattr(self, "_angles_deg", np.array([], dtype=np.float32))
            if ang.size:
                hist, edges = np.histogram(ang, bins=36, range=(-90, 90))
                centers = 0.5 * (edges[:-1] + edges[1:])
                widths  = (edges[1:] - edges[:-1])
                # Angles (green)
                bg = pg.BarGraphItem(
                    x=centers, height=hist, width=widths,
                    brush=pg.mkBrush("#76A64E"),      # green fill
                    pen=pg.mkPen(color="#3C6E2C", width=1)  # darker green edge
                )
                self.graph_angles.addItem(bg)
        else:
            self._style_plotwidget(self.graph_angles, "Pore size distribution", f"Diameter ({units_label})", "Count")
            ds = getattr(self, "_pore_diams_units", np.array([], dtype=np.float32))
            if ds.size:
                hist, edges = np.histogram(ds, bins=40)
                centers = 0.5 * (edges[:-1] + edges[1:])
                widths  = (edges[1:] - edges[:-1])
                bg = pg.BarGraphItem(x=centers, height=hist, width=widths,
                                    brush=pg.mkBrush("#70AD47"),
                                    pen=pg.mkPen(color="#507E2D", width=1))
                self.graph_angles.addItem(bg)

    # ------------------- View & reset -------------------

    def refresh_view(self):
        img = None
        # Use the combo box text to select appropriate view.
        view = self.cb_view.currentText()
        if view == "Original":
            img = ensure_rgb_u8(self.orig_img)
        elif view == "DT":
            img = to_uint8_for_view(self.gray_img)
        elif view == "Laplacian":
            img = to_uint8_for_view(self.lapl_img)
        elif view == "Centerlines":
            img = getattr(self, "_centerlines_dt", None)
        elif view == "Segmentation":
            img = self._labels_rgb()
        elif view == "Overlay":
            img = self.overlay_img
        elif view == "Sampling":
            img = self.sampling_overlay if self.sampling_overlay is not None else ensure_rgb_u8(self.gray_u8)
        elif view in ("Measurements overlay", "Measurements"):
            fallback = self.overlay_img if self.overlay_img is not None else self._labels_rgb()
            img = self.measurements_overlay if self.measurements_overlay is not None else fallback
        elif view == "gx": 
            img = to_uint8_for_view(self.gx)
        elif view == "gy":
            img = to_uint8_for_view(self.gy)

        self._set_pixmap_from_array(img)

    def _reset_internal_state(self, preserve_original: bool = False):
        """Clear all derived data, images, analysis artifacts, and charts."""
        
        self.analyzer.reset_prediction_state()
        if self._profile_dialog is not None:
            self._profile_dialog.close()
        # ---- raw / processing buffers ----
        if not preserve_original:
            self.orig_img = None
            self.current_image_name = None
        self.orig_img_scaled = None
        self.scale_factor = 1.0
        self._scalebar_pending = False
        self._scalebar_points = []
        self.graphicsView.viewport().unsetCursor()
        self.dt = None
        self.gray_img = None
        self.gray_u8 = None
        self.lapl_img = None
        self.gx = None
        self.gy = None

        # ---- segmentation outputs ----
        self.labels_img = None
        self.overlay_img = None
        self.sampling_overlay = None
        self._centerlines_dt = None

        # ---- analysis / debug ----
        self.samples_table = None
        self.fiber_summary = None
        self.measurements_overlay = None
        self._viz_points = None
        self._angles_deg = None
        self._thickness_units = None
        self._void_mask = None
        self._pore_dt = None
        self._pores = None
        self._porosity = None
        self._pore_diams_units = None
        self._labels_rgb_cache = (None, None)
        self._profile_mode_active = False
        self._profile_points_scene = []
        self._profile_points_img = []
        self._remove_profile_items()
        self.btn_reset_view.hide()

        # ---- charts (pyqtgraph) ----
        self.graph_diameters.clear()
        self.graph_diameters.setTitle("")
        self.graph_diameters.showGrid(x=False, y=False)
        self.graph_angles.clear()
        self.graph_angles.setTitle("")
        self.graph_angles.showGrid(x=False, y=False)

        # ---- UI actions / state ----
        self._pixitem.setPixmap(QtGui.QPixmap())

        self.action_save_masks_predictions.setEnabled(False)
        self.action_save_measurements.setEnabled(False)

        self.cb_view.setCurrentText("Original")

    def _init_device_from_checkbox(self):
        """On startup: if CUDA is available, check the box and use cuda; else uncheck and use cpu."""
        cuda_ok = False
        try:
            cuda_ok = torch.cuda.is_available()
        except Exception:
            cuda_ok = False

        if cuda_ok:
            # user prefers GPU by default when available
            self.chb_usegpu.setChecked(True)
            self.device = torch.device('cuda')
        else:
            self.chb_usegpu.setChecked(False)
            self.device = torch.device('cpu')
            # notify once at init if the UI is asking for GPU (edge case: pre-checked in .ui)
            if getattr(self.chb_usegpu, "isChecked", lambda: False)():
                self._show_info(
                    "CUDA not available",
                    "No CUDA-capable GPU detected. Falling back to CPU."
                )

    def _on_usegpu_toggled(self, checked: bool):
        """When user toggles the GPU checkbox, validate and set device; notify if not available."""
        cuda_ok = torch.cuda.is_available()
        if checked and not cuda_ok:
            self._show_warning(
                "CUDA not available",
                "CUDA GPU not detected on this system. Using CPU instead."
            )
            # force UI to reflect reality
            self.chb_usegpu.setChecked(False)
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda' if checked else 'cpu')

        if self.analyzer is not None:
            self.analyzer.set_device(self.device)

    def _on_segmentation_mode_changed(self, checked: bool | None = None):
        new_mode = "watershed"
        if self.rb_ridge.isChecked():
            new_mode = "ridge"
        elif self.rb_watershed.isChecked():
            new_mode = "watershed"

        prev_mode = getattr(self, "segmentation_mode", "watershed")
        if new_mode == prev_mode:
            return

        self.segmentation_mode = new_mode
        self._update_strict_checkbox_state()
        if self.analyzer is not None:
            self.analyzer.set_strict_centerline_filter(self._strict_centerline_enabled())
        self._sync_pipeline_config_from_ui()

    def _on_strict_checkbox_toggled(self, checked: bool):
        self._set_strict_mode(bool(checked))
        self._sync_pipeline_config_from_ui()

    def _on_cluster_toggled(self, checked: bool):
        self.cluster_enabled = bool(checked)
        self._sync_pipeline_config_from_ui()
        cluster_cfg = self.pipeline_config.get("clustering", {})
        cluster_params = dict(cluster_cfg.get("params", {}))
        if self.analyzer is not None:
            self.analyzer.enable_fiber_clustering(self.cluster_enabled, **cluster_params)

    # ------------------- New default image loading -------------------

    def load_default_image(self):
        default_img_path = os.path.join(PROJECT_ROOT, "resources", "test.png")
        if os.path.exists(default_img_path):
            self.load_original_from_path(default_img_path)
        else:
            logger.info("Default image not found: resources/test.png")

    # ------------------- Processing configuration -------------------

    def _init_pipeline_config(self):
        ridge_defaults = FiberAnalyzer._default_centerline_kwargs()
        cluster_defaults: Dict[str, Any] = {}
        if self.analyzer is not None:
            try:
                cluster_defaults = self.analyzer.get_cluster_config()
            except Exception:
                cluster_defaults = {}
        self.pipeline_config = {
            "ridge_detection": {
                "max_angle_deg": self._current_max_angle(),
                "centerline_kwargs": dict(ridge_defaults),
                "strict_centerlines": self._strict_centerline_enabled(),
            },
            "segmentation": {
                "mode": self._current_segmentation_mode(),
                "watershed": {
                    "threshold_u8": self._current_segmentation_threshold(),
                    "radius_thresh_px": None,
                },
                "high_confidence": {
                    "radius_thresh_px": self._current_radius_threshold(),
                    "dilate_extra_px": 0.0,
                    "min_run_points": 1,
                },
            },
            "clustering": {
                "enabled": bool(self.cluster_enabled),
                "params": dict(cluster_defaults),
            },
        }

    def _sync_pipeline_config_from_ui(self):
        if not getattr(self, "pipeline_config", None):
            self._init_pipeline_config()
            return

        ridge_cfg = self.pipeline_config.setdefault("ridge_detection", {})
        ridge_cfg["max_angle_deg"] = self._current_max_angle()
        ridge_cfg.setdefault("centerline_kwargs", dict(FiberAnalyzer._default_centerline_kwargs()))
        ridge_cfg["strict_centerlines"] = self._strict_centerline_enabled()

        seg_cfg = self.pipeline_config.setdefault("segmentation", {})
        seg_cfg["mode"] = self._current_segmentation_mode()
        watershed_cfg = seg_cfg.setdefault("watershed", {})
        watershed_cfg["threshold_u8"] = self._current_segmentation_threshold()
        watershed_cfg.setdefault("radius_thresh_px", None)

        high_conf_cfg = seg_cfg.setdefault("high_confidence", {})
        high_conf_cfg["radius_thresh_px"] = self._current_radius_threshold()
        high_conf_cfg.setdefault("dilate_extra_px", 0.0)
        high_conf_cfg.setdefault("min_run_points", 1)

        cluster_cfg = self.pipeline_config.setdefault("clustering", {})
        cluster_cfg["enabled"] = bool(self.cluster_enabled)
        if "params" not in cluster_cfg or not cluster_cfg["params"]:
            if self.analyzer is not None:
                try:
                    cluster_cfg["params"] = self.analyzer.get_cluster_config()
                except Exception:
                    cluster_cfg["params"] = {}
            else:
                cluster_cfg["params"] = {}

    def _current_scale_factor(self) -> float:
        scale_text = self.cb_scale.currentText().strip().lower()
        scale_text = scale_text.replace("×", "x")
        if scale_text.endswith("x"):
            scale_text = scale_text[:-1]
        try:
            target_scale = float(scale_text)
        except ValueError:
            target_scale = 1.0
        if not np.isfinite(target_scale) or target_scale <= 0:
            target_scale = 1.0
        return float(target_scale)

    def _current_max_angle(self) -> float:
        return float(self.sp_max_cent_angle.value())

    def _strict_centerline_enabled(self) -> bool:
        return bool(getattr(self, "chb_strict", None) and self.chb_strict.isChecked())

    def _set_strict_mode(self, enabled: bool) -> None:
        """Set strict centerline mode across UI, analyzer, and config."""
        if getattr(self, "chb_strict", None):
            self.chb_strict.blockSignals(True)
            self.chb_strict.setChecked(bool(enabled))
            self.chb_strict.blockSignals(False)
        if self.analyzer is not None:
            self.analyzer.set_strict_centerline_filter(bool(enabled))
        if getattr(self, "pipeline_config", None) is not None:
            ridge_cfg = self.pipeline_config.setdefault("ridge_detection", {})
            ridge_cfg["strict_centerlines"] = bool(enabled)

    def _update_strict_checkbox_state(self) -> None:
        enable = self._uses_ridge_segmentation()
        if enable:
            if not self.chb_strict.isEnabled():
                self.chb_strict.setEnabled(True)
        else:
            if self.chb_strict.isChecked():
                self.chb_strict.blockSignals(True)
                self.chb_strict.setChecked(False)
                self.chb_strict.blockSignals(False)
            if self.chb_strict.isEnabled():
                self.chb_strict.setEnabled(False)

    def _is_strict_segmentation(self) -> bool:
        return self._current_segmentation_mode() == "ridge"

    def _uses_ridge_segmentation(self) -> bool:
        return self._current_segmentation_mode() == "ridge"

    def _current_segmentation_mode(self) -> str:
        mode = getattr(self, "segmentation_mode", "watershed")
        if mode not in ("watershed", "ridge"):
            mode = "watershed"
        return mode

    def _current_segmentation_threshold(self) -> int:
        return int(self.sp_segm_threshold.value())

    def _current_radius_threshold(self) -> float | None:
        if not self._uses_ridge_segmentation():
            return None
        return float(self._current_segmentation_threshold())
