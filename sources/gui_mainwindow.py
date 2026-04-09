import os
import math
import json
from datetime import datetime
from typing import Any, Dict
import numpy as np
import cv2
import imageio.v3 as iio

from PyQt6 import QtGui, QtWidgets
from PyQt6.QtCore import Qt, QEvent, QTimer, QPoint, QLineF, QSize
from PyQt6.QtWidgets import (
    QMainWindow,
    QFileDialog,
    QGraphicsScene,
    QGraphicsPixmapItem,
    QGraphicsEllipseItem,
    QGraphicsItem,
    QToolButton,
    QMessageBox,
)
from PyQt6.uic import loadUi

import pyqtgraph as pg
import torch
from scipy.spatial import cKDTree
from skimage.measure import profile_line

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

from .gui_dialogs import (
    PlotProfileDialog,
    BatchSaveOptionsDialog,
    Stack3DWindow,
    ConfigEditorDialog,
)
from .fiber_analyzer_2D import FiberAnalyzer_2D
from .datastructures import BatchResult, Frame, ImageBatch, AppConfig, SampleProperties
from .batch_pipeline import BatchAnalyzer, StackAnalyzer_3D
from .utils import ensure_rgb_u8, label_to_rgb, to_uint8_for_view

# ------------------- Paths -------------------

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
DEFAULT_UI_PATH = os.path.join(PROJECT_ROOT, "resources", "fiber_tool.ui")
DEFAULT_CONFIG_PATH = os.path.join(PROJECT_ROOT, "resources", "config.json")
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


class MainWindow(QMainWindow):
    """
    Single-file refactor notes:
    - One overlay renderer (removed duplicate).
    - One analysis pass driven by current thresholds/params.
    - Clear state when loading a new image.
    - Tightened UI updates/enabling.
    """
    def __init__(self, ui_path: str = DEFAULT_UI_PATH, config_path: str = DEFAULT_CONFIG_PATH):
        super().__init__()
        loadUi(ui_path, self)
        self.setWindowTitle("Fiberpose — fiber segmentation and analysis tool")
        self.config_filepath = os.path.abspath(config_path)

        # Load config and sync to UI
        self.config = AppConfig.load_from_json(self.config_filepath)
        self.sample_props = SampleProperties.defaults()
        self.sync_config_to_ui()
        self.sync_sample_properties_to_ui()

        # Enable real-time syncing!
        self._connect_config_signals()
        
        # Scene and view setup 
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

        # Working dir & runtime state setup
        self.working_dir = os.path.dirname(ui_path)
        self.current_frame: Frame = Frame.empty()
        self.analyzer: FiberAnalyzer_2D | None = None
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
        self._suppress_batch_dialogs: bool = False
        self._batch_save_options: Dict[str, bool] | None = None
        self._is_batch_processing: bool = False
        self._image_batch: ImageBatch | None = None
        self._batch_result: BatchResult | None = None
        self._batch_results_have_measurements: bool = False
        self._last_batch_measurements_signature: str | None = None
        self._stack_diameter_histogram_data: np.ndarray | None = None
        self._stack_angle_histogram_data: np.ndarray | None = None
        self._stack_diameter_histogram_enabled: bool = False
        self._batch_total_samples: int = 0
        self._batch_processed_samples: int = 0
        self._batch_collects_measurements: bool = False
        self._current_z_index: int = 0
        self._stack3d_window: Stack3DWindow | None = None
        self._slice_label = QtWidgets.QLabel("")
        self.statusbar.addPermanentWidget(self._slice_label)
        self.z_scroll.setEnabled(False)
        self.z_scroll.valueChanged.connect(self._on_z_scroll_changed)
        self._update_slice_label()

        # Profile plotting tool state
        self._profile_mode_active: bool = False
        self._profile_points_scene: list[tuple[float, float]] = []
        self._profile_points_img: list[tuple[float, float]] = []
        self._profile_line_item: QGraphicsItem | None = None
        self._profile_point_items: list[QGraphicsItem] = []
        self._profile_dialog: PlotProfileDialog | None = None

        self.rb_watershed.blockSignals(True)
        self.rb_watershed.setChecked(True)
        self.rb_watershed.blockSignals(False)
        self.rb_ridge.blockSignals(True)
        self.rb_ridge.setChecked(False)
        self.rb_ridge.blockSignals(False)
        if hasattr(self, "rb_fastest_ascend"):
            self.rb_fastest_ascend.blockSignals(True)
            self.rb_fastest_ascend.setChecked(False)
            self.rb_fastest_ascend.blockSignals(False)
        self.chb_strict.blockSignals(True)
        self.chb_strict.setChecked(False)
        self.chb_strict.setEnabled(False)
        self.chb_strict.blockSignals(False)
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
        if hasattr(self, "action_edit"):
            self.action_edit.setText("Edit config...")
            self.action_edit.triggered.connect(self._on_edit_config)
        if hasattr(self, "menuEdit"):
            self.action_load_config = QtGui.QAction("Load config...", self)
            self.action_load_config.triggered.connect(self._on_load_config)
            if hasattr(self, "action_edit"):
                self.menuEdit.insertAction(self.action_edit, self.action_load_config)
            else:
                self.menuEdit.addAction(self.action_load_config)

        self.btn_segment.clicked.connect(self.segment)
        self.btn_analyze_2D.clicked.connect(self.analyze_fibers)
        self.btn_save_csv.clicked.connect(self.save_measurements)
        self.btn_set_scalebar.clicked.connect(self._on_set_scalebar_clicked)
        self.btn_plot_profile.clicked.connect(self._on_plot_profile_clicked)
        self.btn_analyze_3D.clicked.connect(self._on_analyze_3d_clicked)
        self.action_batch_processing.triggered.connect(self._on_batch_processing)
        self.btn_open_3d.clicked.connect(self._on_open_3d_view)

        # Remove connection of radiobutton signals and instead hook up the combobox change
        self.cb_view.currentIndexChanged.connect(self.refresh_view)

        # Set default checkpoint
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
        self.analyzer = FiberAnalyzer_2D(
            ckpt_path=self.ckpt_path,
            device=self.device,
            strict=self.config.strict_centerlines, 
        )
        self.analyzer.set_strict_centerline_filter(self.config.strict_centerlines)

        # react to checkbox changes at runtime
        self.chb_usegpu.toggled.connect(self._on_usegpu_toggled)
        self.chb_cluster.toggled.connect(self._on_cluster_toggled)
        self._on_cluster_toggled(self.chb_cluster.isChecked())
        self.rb_watershed.toggled.connect(self._on_segmentation_mode_changed)
        self.rb_ridge.toggled.connect(self._on_segmentation_mode_changed)
        if hasattr(self, "rb_fastest_ascend"):
            self.rb_fastest_ascend.toggled.connect(self._on_segmentation_mode_changed)
        self.chb_strict.toggled.connect(self._on_strict_checkbox_toggled)
        self._update_strict_checkbox_state()

        # Connect toggled signals of rb_show_diam and rb_show_direct to update overlay
        self.rb_show_diam.toggled.connect(self._render_vector_overlay)
        self.rb_show_direct.toggled.connect(self._render_vector_overlay)
        self.rb_show_pores.toggled.connect(self._render_vector_overlay)

    def closeEvent(self, event: QtGui.QCloseEvent):
        """Triggered when the user closes the main window."""
        # Grab the latest state from the UI
        self.sync_ui_to_config()
        self.sync_ui_to_sample_properties()
        # Save it to disk
        cfg_dir = os.path.dirname(self.config_filepath)
        if cfg_dir:
            os.makedirs(cfg_dir, exist_ok=True)
        self.config.save_to_json(self.config_filepath)
        super().closeEvent(event)

    def _connect_config_signals(self):
        """Connects UI widget signals to update the config immediately on change."""
        
        # Combo boxes (Dropdowns)
        self.cb_model.currentIndexChanged.connect(self.sync_ui_to_config)
        self.cb_scale.currentIndexChanged.connect(self.sync_ui_to_config)
        self.cb_units.currentTextChanged.connect(self.sync_ui_to_sample_properties)

        # Checkboxes and Radio Buttons
        self.chb_usegpu.toggled.connect(self.sync_ui_to_config)
        self.chb_strict.toggled.connect(self.sync_ui_to_config)
        self.chb_cluster.toggled.connect(self.sync_ui_to_config)
        
        self.rb_watershed.toggled.connect(self.sync_ui_to_config)
        self.rb_ridge.toggled.connect(self.sync_ui_to_config)
        if hasattr(self, "rb_fastest_ascend"):
            self.rb_fastest_ascend.toggled.connect(self.sync_ui_to_config)

        # Spinboxes
        # `sp_max_cent_angle` is config-only (not read from main panel).
        self.sp_segm_threshold.valueChanged.connect(self.sync_ui_to_config)

        # Line Edits (Text boxes)
        # Note: We use `editingFinished` instead of `textChanged`.
        # `le_samspacing` is config-only (not read from main panel).
        self.le_pixelsize.editingFinished.connect(self.sync_ui_to_sample_properties)
        if hasattr(self, "le_voxeldepth"):
            self.le_voxeldepth.editingFinished.connect(self.sync_ui_to_sample_properties)

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

    def _centerlines_with_connections(self) -> np.ndarray | None:
        """
        Return the cached centerline view with clustering links overlaid for visibility.
        """
        img = self.current_frame.centerlines_overlay_img
        base = ensure_rgb_u8(img)
        if base is None:
            return None
        if not self.config.cluster_enabled or self.analyzer is None:
            return base
        try:
            connections = self.analyzer.get_cluster_connections()
        except Exception:
            connections = []
        if not connections:
            return base
        out = base.copy()
        color = (0, 255, 0)
        for p_xy, q_xy in connections:
            px = int(round(float(p_xy[0])))
            py = int(round(float(p_xy[1])))
            qx = int(round(float(q_xy[0])))
            qy = int(round(float(q_xy[1])))
            cv2.line(out, (px, py), (qx, qy), color, 1, cv2.LINE_AA)
        return out

    def _labels_rgb(self) -> np.ndarray | None:
        if self.current_frame.labels_2d is None:
            return None
        cache_id, cache_img = self.current_frame.labels_rgb_cache
        current_id = id(self.current_frame.labels_2d)
        if cache_id != current_id:
            cache_img = labels_to_color(self.current_frame.labels_2d)
            self.current_frame.labels_rgb_cache = (current_id, cache_img)
        return cache_img

    def _ensure_dt_ready(self) -> bool:
        if self.current_frame.orig_img is None:
            self._show_info("Load original", "Load an image before using the profile tool.")
            return False
            
        # 1. ADD THIS CHECK: If current_frame already has its DT, we are ready.
        if self.current_frame.dt is not None:
            return True

        if self.analyzer is None:
            self._show_error("Analyzer unavailable", "Fiber analyzer has not been initialized.")
            return False

        self.sync_ui_to_config()
        max_angle = self.config.max_angle_deg
        centerline_kwargs = getattr(self.analyzer, "_centerline_kwargs", FiberAnalyzer_2D._default_centerline_kwargs())

    
        # If we reach here, current_frame.dt is None, so we MUST predict 
        # specifically for this frame's original image.
        try:
            self.analyzer.predict_dt(
                image=self.current_frame.orig_img,
                max_angle_deg=max_angle,
                centerline_kwargs=centerline_kwargs or None,
            )
        except Exception as exc:
            self.current_frame.fail(str(exc))
            self._show_error("Prediction failed", str(exc))
            logger.error("Profile DT ensure failed: %s", exc)
            return False

        dt = self.analyzer.get_dt()
        if dt is None:
            self.current_frame.fail("Distance transform could not be computed.")
            self._show_error("DT unavailable", "Distance transform could not be computed.")
            return False

        gx, gy = self.analyzer.get_gradients()
        dt_u8 = self.analyzer.get_dt_u8()
        dt_norm = self.analyzer.get_dt_norm()
        lapl = self.analyzer.get_laplacian()
        style_vec = self.analyzer.get_style_vector()
        
        self.current_frame.set_prediction_outputs(
            dt=np.asarray(dt, dtype=np.float32, copy=False),
            gx=np.asarray(gx, dtype=np.float32, copy=False) if gx is not None else None,
            gy=np.asarray(gy, dtype=np.float32, copy=False) if gy is not None else None,
            dt_u8=np.asarray(dt_u8, dtype=np.uint8, copy=False) if dt_u8 is not None else None,
            dt_normalized=np.asarray(dt_norm, dtype=np.float32, copy=False) if dt_norm is not None else None,
            lapl_img=np.asarray(lapl, dtype=np.float32, copy=False) if lapl is not None else None,
            style_vec=np.asarray(style_vec, dtype=np.float32, copy=False) if style_vec is not None else None,
            orig_img_scaled=self.current_frame.orig_img_scaled,
            scale_factor=self.current_frame.scale_factor,
            invalidate_downstream=False,
        )
        return True

    def _show_info(self, title: str, message: str):
        if self._suppress_batch_dialogs:
            logger.info("[Batch] %s: %s", title, message)
        else:
            QtWidgets.QMessageBox.information(self, title, message)

    def _show_warning(self, title: str, message: str):
        if self._suppress_batch_dialogs:
            logger.warning("[Batch] %s: %s", title, message)
        else:
            QtWidgets.QMessageBox.warning(self, title, message)

    def _show_error(self, title: str, message: str):
        if self._suppress_batch_dialogs:
            logger.error("[Batch] %s: %s", title, message)
        else:
            QtWidgets.QMessageBox.critical(self, title, message)

    def _apply_runtime_after_config_change(self) -> None:
        checkpoint_entry = str(getattr(self.config, "checkpoint_path", "") or "").strip()
        if checkpoint_entry and self.cb_model.findText(checkpoint_entry) < 0:
            self.cb_model.addItem(checkpoint_entry)

        self.sync_config_to_ui()
        self.sync_ui_to_config()

        if self.analyzer is not None:
            self._on_checkpoint_selection_changed(self.cb_model.currentIndex())
            self._on_usegpu_toggled(self.chb_usegpu.isChecked())
            self._on_cluster_toggled(self.chb_cluster.isChecked())
            self.analyzer.set_strict_centerline_filter(self.config.strict_centerlines)

    def _load_config_from_path(self, path: str) -> bool:
        selected_path = os.path.abspath(path)
        try:
            with open(selected_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            if not isinstance(raw, dict):
                raise ValueError("Config JSON root must be an object.")
            valid_keys = set(AppConfig.__dataclass_fields__.keys())
            filtered = {k: v for k, v in raw.items() if k in valid_keys}
            self.config = AppConfig(**filtered)
        except Exception as exc:
            self._show_error("Load config failed", str(exc))
            return False

        self.config_filepath = selected_path
        self._apply_runtime_after_config_change()
        return True

    def _on_load_config(self):
        start_dir = os.path.dirname(self.config_filepath) if self.config_filepath else PROJECT_ROOT
        selected_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select config file",
            start_dir,
            "JSON Files (*.json);;All Files (*)",
        )
        if not selected_path:
            return
        if self._load_config_from_path(selected_path):
            self.statusbar.showMessage(f"Config loaded: {self.config_filepath}", 5000)

    def _on_edit_config(self):
        dialog = ConfigEditorDialog(self.config, self.config_filepath, self)
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return

        self.config = dialog.result_config()
        self.config_filepath = dialog.result_config_path()
        self._apply_runtime_after_config_change()
        self.statusbar.showMessage(f"Config updated: {self.config_filepath}", 5000)


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
                if md and md.hasUrls() and self._urls_are_loadable(md.urls()):
                    event.acceptProposedAction()
                    return True
            elif et == QEvent.Type.Drop:
                md = event.mimeData()
                if md and md.hasUrls():
                    for u in md.urls():
                        if u.isLocalFile():
                            p = u.toLocalFile()
                            if self._is_loadable_path(p):
                                if os.path.isdir(p):
                                    self._run_batch_processing_for_folder(p)
                                else:
                                    self.open_source(path=p, source_kind="auto")
                                event.acceptProposedAction()
                                return True
        return super().eventFilter(obj, event)

    def _is_image_path(self, path: str) -> bool:
        ext = os.path.splitext(path)[1].lower()
        return ext in self._supported_image_extensions()

    def _is_loadable_path(self, path: str) -> bool:
        return os.path.isdir(path) or self._is_image_path(path)

    def _urls_are_loadable(self, urls) -> bool:
        return any(u.isLocalFile() and self._is_loadable_path(u.toLocalFile()) for u in urls)

    def _compute_orientation_stats(
        self, angles_samples_deg: np.ndarray
    ) -> tuple[float, float, float]:
        """
        Return orientation index, circular mean angle (deg), and circular std (deg)
        for canonical angles in degrees.
        """
        if angles_samples_deg is None or getattr(angles_samples_deg, "size", 0) == 0:
            return float("nan"), float("nan"), float("nan")

        angles_rad = np.deg2rad(angles_samples_deg)
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
        if self.current_frame.fiber_summary:
            return (
                float(self.current_frame.fiber_summary.get("mean_diameter", float("nan"))),
                float(self.current_frame.fiber_summary.get("std_diameter", float("nan"))),
                float(self.current_frame.fiber_summary.get("orientation_index", float("nan"))),
                float(self.current_frame.fiber_summary.get("mean_angle_deg", float("nan"))),
                float(self.current_frame.fiber_summary.get("std_angle_deg", float("nan"))),
                float(self.current_frame.fiber_summary.get("porosity", float("nan"))),
                int(self.current_frame.fiber_summary.get("sample_count", 0)),
            )

        diameters = self.current_frame.thickness_units if self.current_frame.thickness_units is not None else np.array([])
        angles = self.current_frame.angles_samples_deg if self.current_frame.angles_samples_deg is not None else np.array([])
        mean_val = float(np.mean(diameters)) if diameters.size else float("nan")
        std_val = float(np.std(diameters)) if diameters.size else float("nan")
        sample_count = int(self.current_frame.samples_table.shape[0]) if self.current_frame.samples_table is not None else int(diameters.size)
        orientation_index, mean_angle_deg, std_angle_deg = self._compute_orientation_stats(angles)
        porosity = float(self.current_frame.porosity) if self.current_frame.porosity is not None else float("nan")
        return mean_val, std_val, orientation_index, mean_angle_deg, std_angle_deg, porosity, sample_count

    def _current_batch_measurement_signature(self) -> str | None:
        if self._image_batch is None:
            return None

        metadata = self._image_batch.metadata or {}
        filenames = [str(v) for v in metadata.get("filenames", [])]
        z_indices = [int(v) for v in metadata.get("z_indices", [])]
        spacing_zyx = tuple(float(v) for v in self.sample_props.spacing_zyx())
        cluster_params = getattr(self.config, "cluster_params", {}) or {}

        payload = {
            "batch": {
                "source_type": str(metadata.get("source_type", "")),
                "source_path": str(metadata.get("source_path", "")),
                "frame_count": int(len(self._image_batch)),
                "filenames_head": filenames[:5],
                "filenames_tail": filenames[-5:] if len(filenames) > 5 else [],
                "z_head": z_indices[:5],
                "z_tail": z_indices[-5:] if len(z_indices) > 5 else [],
            },
            "config": {
                "checkpoint_path": str(self.config.checkpoint_path),
                "scale_label": str(self.config.scale_label),
                "max_angle_deg": float(self.config.max_angle_deg),
                "strict_centerlines": bool(self.config.strict_centerlines),
                "segmentation_mode": str(self.config.segmentation_mode),
                "seg_threshold": int(self.config.seg_threshold),
                "sample_spacing": float(self.config.sample_spacing),
                "cluster_enabled": bool(self.config.cluster_enabled),
                "cluster_params": cluster_params,
            },
            "sample_properties": {
                "spacing_zyx": spacing_zyx,
                "unit": str(self.sample_props.unit),
            },
        }
        return json.dumps(payload, sort_keys=True, default=str)

    def _batch_measurements_match_current_settings(self) -> bool:
        if self._batch_result is None or self._image_batch is None:
            return False
        if not self._batch_results_have_measurements:
            return False
        if len(self._batch_result.frame_results) != len(self._image_batch):
            return False
        if self._last_batch_measurements_signature != self._current_batch_measurement_signature():
            return False
        return True

    @staticmethod
    def _collect_stack_diameter_samples(batch_result: BatchResult | None) -> np.ndarray:
        if batch_result is None:
            return np.array([], dtype=np.float32)

        chunks: list[np.ndarray] = []
        for frame in batch_result.frame_results:
            values = frame.thickness_units
            if values is None:
                continue
            arr = np.asarray(values, dtype=np.float32).reshape(-1)
            if arr.size == 0:
                continue
            arr = arr[np.isfinite(arr)]
            arr = arr[arr > 0]
            if arr.size:
                chunks.append(arr)

        if not chunks:
            return np.array([], dtype=np.float32)
        return np.concatenate(chunks).astype(np.float32, copy=False)

    @staticmethod
    def _collect_stack_angle_samples(batch_result: BatchResult | None) -> np.ndarray:
        if batch_result is None:
            return np.array([], dtype=np.float32)

        chunks: list[np.ndarray] = []
        for frame in batch_result.frame_results:
            values = frame.angles_samples_deg
            if values is None:
                continue
            arr = np.asarray(values, dtype=np.float32).reshape(-1)
            if arr.size == 0:
                continue
            arr = arr[np.isfinite(arr)]
            if arr.size:
                chunks.append(arr)

        if not chunks:
            return np.array([], dtype=np.float32)
        return np.concatenate(chunks).astype(np.float32, copy=False)

    # ------------------- I/O -------------------

    def _set_image_batch(self, image_batch: ImageBatch):
        self._image_batch = image_batch
        self._batch_result = None
        self._batch_results_have_measurements = False
        self._last_batch_measurements_signature = None
        self._stack_diameter_histogram_data = None
        self._stack_angle_histogram_data = None
        self._stack_diameter_histogram_enabled = False
        self._current_z_index = 0
        self._configure_z_scroll(len(image_batch))
        self._set_current_slice(0, update_view=True)

    def _apply_sample_properties_from_batch(self, image_batch: ImageBatch) -> None:
        source_path = str(image_batch.metadata.get("source_path", ""))
        self.sample_props = SampleProperties.from_image_metadata(
            image_batch.metadata,
            source_path=source_path,
        )
        self.sync_sample_properties_to_ui()

        z, y, x = self.sample_props.spacing_zyx()
        logger.info(
            "Sample metadata applied: pixel_width=%s pixel_height=%s voxel_depth=%s unit=%s source=%s",
            image_batch.metadata.get("pixel_width"),
            image_batch.metadata.get("pixel_height"),
            image_batch.metadata.get("voxel_depth"),
            image_batch.metadata.get("unit"),
            self.sample_props.source,
        )
        if self.sample_props.source == "metadata":
            self.statusbar.showMessage(
                f"Loaded sample spacing from metadata: z={z:.6g}, y={y:.6g}, x={x:.6g} {self.sample_props.unit}/px",
                6000,
            )
        else:
            self.statusbar.showMessage(
                "No spacing metadata found. Using default sample spacing: z=1, y=1, x=1 px.",
                6000,
            )

    def open_source(self, path: str | None = None, source_kind: str = "auto") -> ImageBatch | None:
        selected_path = path

        if selected_path is None:
            if source_kind == "folder":
                selected_path = QFileDialog.getExistingDirectory(
                    self, "Select folder with images", self.working_dir
                )
            else:
                selected_path, _ = QtWidgets.QFileDialog.getOpenFileName(
                    self,
                    "Select Original Image",
                    self.working_dir,
                    "Images (*.tif *.tiff *.png *.jpg *.jpeg *.bmp *.webp);;All Files (*)",
                )
        if not selected_path:
            return None

        if source_kind not in {"auto", "file", "folder"}:
            self._show_error("Open source", f"Unsupported source kind: {source_kind}")
            return None

        try:
            if source_kind == "folder":
                image_batch = ImageBatch.from_folder(
                    selected_path,
                    extensions=self._supported_image_extensions(),
                )
            elif source_kind == "file":
                image_batch = ImageBatch.from_path(selected_path)
            else:
                image_batch = ImageBatch.from_source(
                    selected_path,
                    extensions=self._supported_image_extensions(),
                )
        except Exception as e:
            self._show_error("Error", f"Failed to read source:\n{e}")
            logger.error("Failed to read source %s: %s", selected_path, e)
            return None

        if len(image_batch) == 0:
            self._show_warning("Unsupported", "No readable frames were found in the selected source.")
            return None

        self._reset_internal_state()
        if os.path.isdir(selected_path):
            self.working_dir = selected_path
        else:
            self.working_dir = os.path.dirname(selected_path)
        self._set_image_batch(image_batch)
        self._apply_sample_properties_from_batch(image_batch)
        logger.info("Loaded %d frame(s) from %s", len(image_batch), selected_path)
        return image_batch

    def load_original(self):
        self.open_source(source_kind="file")

    def load_original_from_path(self, path: str):
        self.open_source(path=path, source_kind="auto")

    def _is_single_tiff_stack_source(self) -> bool:
        if self._image_batch is None:
            return False
        metadata = self._image_batch.metadata or {}
        source_type = str(metadata.get("source_type", "")).strip().lower()
        if source_type != "stack" or len(self._image_batch) <= 1:
            return False
        source_path = str(metadata.get("source_path", "")).strip()
        if not source_path or not os.path.isfile(source_path):
            return False
        return os.path.splitext(source_path)[1].lower() in {".tif", ".tiff"}

    def save_predictions(self):
        batch_frame_count = (
            len(self._batch_result.frame_results)
            if self._batch_result is not None
            else 0
        )
        if batch_frame_count > 1:
            has_predictions = any(
                fr.has_prediction or fr.has_segmentation
                for fr in self._batch_result.frame_results
            )
            if not has_predictions:
                self._show_info("Nothing to save", "Run segmentation first.")
                return

            dest_dir = QFileDialog.getExistingDirectory(
                self,
                "Save batch predictions to folder",
                self.working_dir,
            )
            if not dest_dir:
                return
            if self._is_single_tiff_stack_source():
                source_path = str(self._image_batch.metadata.get("source_path", ""))
                base_name = os.path.splitext(os.path.basename(source_path))[0] or "stack"
                frames = list(self._batch_result.frame_results)
                written_count = 0
                errors: list[dict[str, Any]] = []

                def _stack_path(suffix: str) -> str:
                    base_path = os.path.join(dest_dir, f"{base_name}{suffix}.tiff")
                    if not os.path.exists(base_path):
                        return base_path
                    idx = 2
                    while True:
                        cand = os.path.join(dest_dir, f"{base_name}{suffix}_{idx}.tiff")
                        if not os.path.exists(cand):
                            return cand
                        idx += 1

                if all(fr.dt_u8 is not None for fr in frames):
                    try:
                        dt_stack = np.stack(
                            [np.asarray(fr.dt_u8, dtype=np.uint8) for fr in frames],
                            axis=0,
                        )
                        iio.imwrite(_stack_path("_dt"), dt_stack)
                        written_count += 1
                    except Exception as exc:
                        errors.append({"artifact": "dt", "error": str(exc)})

                if all(fr.labels_2d is not None for fr in frames):
                    try:
                        labels_slices: list[np.ndarray] = []
                        has_foreground = False
                        for fr in frames:
                            labels_2d = np.asarray(fr.labels_2d)
                            if labels_2d.ndim == 3:
                                labels_2d = labels_2d[..., 0]
                            if labels_2d.ndim != 2:
                                raise ValueError("Labels are not 2D.")
                            if labels_2d.size and int(np.max(labels_2d)) > 0:
                                has_foreground = True
                            labels_slices.append(np.asarray(labels_2d, dtype=np.uint16))
                        if has_foreground:
                            labels_stack = np.stack(labels_slices, axis=0)
                            iio.imwrite(_stack_path("_labels"), labels_stack)
                            written_count += 1
                    except Exception as exc:
                        errors.append({"artifact": "labels", "error": str(exc)})

                if all(fr.overlay_img is not None for fr in frames):
                    try:
                        overlay_stack = np.stack(
                            [np.asarray(fr.overlay_img, dtype=np.uint8) for fr in frames],
                            axis=0,
                        )
                        iio.imwrite(_stack_path("_overlay"), overlay_stack)
                        written_count += 1
                    except Exception as exc:
                        errors.append({"artifact": "overlay", "error": str(exc)})

                if all(fr.measurements_overlay is not None for fr in frames):
                    try:
                        debug_stack = np.stack(
                            [np.asarray(fr.measurements_overlay, dtype=np.uint8) for fr in frames],
                            axis=0,
                        )
                        iio.imwrite(_stack_path("_debug"), debug_stack)
                        written_count += 1
                    except Exception as exc:
                        errors.append({"artifact": "debug", "error": str(exc)})

                export_result = {"written_count": written_count, "errors": errors}
            else:
                export_config = {
                    "preset": "predictions",
                    "image_format": "png",
                    "naming": "frame_id_z",
                    "collision": "suffix",
                    "write_manifest": True,
                }
                try:
                    export_result = self._batch_result.save_all(dest_dir, export_config)
                except Exception as e:
                    self._show_error("Error", f"Failed to save batch predictions:\n{e}")
                    return

            self.working_dir = dest_dir
            error_count = len(export_result.get("errors", []))
            if error_count:
                self._show_warning(
                    "Saved with issues",
                    f"Saved {export_result.get('written_count', 0)} file(s) to:\n{dest_dir}\n"
                    f"Encountered {error_count} export error(s).",
                )
                return
            self._show_info(
                "Saved",
                f"Saved {export_result.get('written_count', 0)} file(s) to:\n{dest_dir}",
            )
            return

        if not (self.current_frame.has_prediction or self.current_frame.has_segmentation):
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
        dest_dir = os.path.dirname(base) or self.working_dir
        base_name = os.path.splitext(os.path.basename(base))[0] or (self.current_image_name or "fiberpose")
        selected_ext = os.path.splitext(base)[1].lower()
        image_format = "tiff" if selected_ext in {".tif", ".tiff"} else "png"
        export_config = {
            "preset": "predictions",
            "image_format": image_format,
            "naming": "frame_id",
            "collision": "overwrite",
            "base_name": base_name,
            "write_manifest": False,
        }

        try:
            export_result = self._export_current_frame(dest_dir, export_config)
        except Exception as e:
            self._show_error("Error", f"Failed to save:\n{e}")
            return

        self.working_dir = dest_dir
        error_count = len(export_result.get("errors", []))
        if error_count:
            self._show_warning(
                "Saved with issues",
                f"Saved {export_result.get('written_count', 0)} file(s) to:\n{dest_dir}\n"
                f"Encountered {error_count} export error(s).",
            )
            return
        self._show_info(
            "Saved",
            f"Saved {export_result.get('written_count', 0)} file(s) to:\n{dest_dir}",
        )

    def save_measurements(self):
        batch_frame_count = (
            len(self._batch_result.frame_results)
            if self._batch_result is not None
            else 0
        )
        if batch_frame_count > 1:
            has_measurements = any(
                fr.has_measurements and fr.samples_table is not None and len(fr.samples_table) > 0
                for fr in self._batch_result.frame_results
            )
            if not has_measurements:
                self._show_info("Nothing to save", "Run Analyze fibers first.")
                return

            dest_dir = QFileDialog.getExistingDirectory(
                self,
                "Save batch measurements to folder",
                self.working_dir,
            )
            if not dest_dir:
                return
            if self._is_single_tiff_stack_source():
                source_path = str(self._image_batch.metadata.get("source_path", ""))
                base_name = os.path.splitext(os.path.basename(source_path))[0] or "stack"
                csv_path = os.path.join(dest_dir, f"{base_name}_samples.csv")
                if os.path.exists(csv_path):
                    idx = 2
                    while True:
                        cand = os.path.join(dest_dir, f"{base_name}_samples_{idx}.csv")
                        if not os.path.exists(cand):
                            csv_path = cand
                            break
                        idx += 1
                try:
                    with open(csv_path, "w", newline="") as handle:
                        handle.write("z_index,frame_id,x,y,diameter,angle_deg\n")
                        for fr in self._batch_result.frame_results:
                            if fr.samples_table is None or len(fr.samples_table) == 0:
                                continue
                            z_index = int(getattr(fr, "z_index", 0))
                            frame_id = str(getattr(fr, "frame_id", ""))
                            thickness_units = getattr(fr, "thickness_units", None)
                            for sample_index, row in enumerate(fr.samples_table, start=1):
                                x, y, t_px = row[:3]
                                angle_val = float(row[3]) if len(row) > 3 else float("nan")
                                if thickness_units is not None and len(thickness_units) >= sample_index:
                                    diameter_val = float(thickness_units[sample_index - 1])
                                else:
                                    diameter_val = float(t_px)
                                handle.write(
                                    f"{z_index},{frame_id},{int(x)},{int(y)},{diameter_val:.6f},{angle_val:.6f}\n"
                                )
                    export_result = {"written_count": 1, "errors": []}
                except Exception as e:
                    self._show_error("Error", f"Failed to save stack measurements:\n{e}")
                    return
            else:
                export_config = {
                    "preset": "measurements",
                    "naming": "frame_id_z",
                    "collision": "suffix",
                    "write_manifest": True,
                }
                try:
                    export_result = self._batch_result.save_all(dest_dir, export_config)
                except Exception as e:
                    self._show_error("Error", f"Failed to save batch measurements:\n{e}")
                    return

            self.working_dir = dest_dir
            error_count = len(export_result.get("errors", []))
            if error_count:
                self._show_warning(
                    "Saved with issues",
                    f"Saved {export_result.get('written_count', 0)} file(s) to:\n{dest_dir}\n"
                    f"Encountered {error_count} export error(s).",
                )
                return
            self._show_info(
                "Saved",
                f"Saved {export_result.get('written_count', 0)} file(s) to:\n{dest_dir}",
            )
            return

        if not self.current_frame.has_measurements or self.current_frame.samples_table is None or len(self.current_frame.samples_table) == 0:
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
        dest_dir = os.path.dirname(base) or self.working_dir
        base_name = os.path.splitext(os.path.basename(base))[0] or (self.current_image_name or "fiberpose")
        export_config = {
            "preset": "measurements",
            "naming": "frame_id",
            "collision": "overwrite",
            "base_name": base_name,
            "write_manifest": False,
        }

        try:
            export_result = self._export_current_frame(dest_dir, export_config)
        except Exception as e:
            self._show_error("Error", f"Failed to save CSVs:\n{e}")
            return

        error_count = len(export_result.get("errors", []))
        if error_count:
            self._show_warning(
                "Saved with issues",
                f"Saved {export_result.get('written_count', 0)} file(s) to:\n{dest_dir}\n"
                f"Encountered {error_count} export error(s).",
            )
            return
        self.working_dir = dest_dir
        self._show_info(
            "Saved",
            f"Saved {export_result.get('written_count', 0)} CSV file(s) to:\n{dest_dir}",
        )

    def _export_current_frame(self, output_dir: str, export_config: Dict[str, Any]) -> Dict[str, Any]:
        frame_result = self.current_frame.snapshot(
            include_measurements=True,
            frame_id=str(self.current_frame.frame_id),
            z_index=int(self.current_frame.z_index),
        )
        batch_result = BatchResult(frame_results=[frame_result])
        return batch_result.save_all(output_dir, export_config)


    # ------------------- Scalebar -------------------

    def _on_set_scalebar_clicked(self):
        if self.current_frame.orig_img is None:
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
        self.sample_props.pixel_size_x = float(pixel_size)
        self.sample_props.pixel_size_y = float(pixel_size)
        self.sample_props.unit = str(self._scalebar_unit or "px")
        self.sample_props.source = "manual"
        self.sync_sample_properties_to_ui()

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

    def _set_progress_bar(self, sample_progress_percent: float) -> None:
        progress = float(max(0.0, min(100.0, sample_progress_percent)))
        if self._is_batch_processing and self._batch_total_samples > 0:
            current_sample_progress = progress / 100.0
            processed = min(max(int(self._batch_processed_samples), 0), self._batch_total_samples)
            total = float(self._batch_total_samples)
            # Global batch progress: (processed samples + current sample progress) / total samples.
            overall = ((processed + current_sample_progress) * 100.0) / total
            overall = max(0.0, min(100.0, overall))
            self.progress_bar.setValue(int(round(overall)))
            return
        self.progress_bar.setValue(int(round(progress)))

    def _run_batch_analysis(
        self,
        compute_measurements: bool,
        image_batch: ImageBatch | None = None,
        error_title: str | None = None,
    ) -> BatchResult | None:
        # Resolve the active batch source. Most callers use the currently loaded batch.
        active_batch = image_batch if image_batch is not None else self._image_batch
        if active_batch is None or len(active_batch) == 0:
            self._show_info("Load original", "Please load an image or folder first.")
            return None

        # Build user-facing error text per processing mode unless explicitly provided.
        title = error_title
        if not title:
            title = "Batch analysis failed" if compute_measurements else "Batch segmentation failed"

        QtWidgets.QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            total_images = len(active_batch)
            self.statusbar.showMessage(f"Processing 0/{total_images} images")
            self._batch_total_samples = total_images
            self._batch_processed_samples = 0
            self._batch_collects_measurements = bool(compute_measurements)
            self._set_progress_bar(0.0)
            if self.analyzer is None:
                raise RuntimeError("Fiber analyzer has not been initialized.")

            # Enter internal batch execution mode so frame callbacks can call segment/analyze safely.
            batch_analyzer = BatchAnalyzer(self.analyzer)
            previous_batch_state = self._suppress_batch_dialogs
            previous_processing_state = self._is_batch_processing
            self._is_batch_processing = True
            if len(active_batch) > 1:
                self._suppress_batch_dialogs = True

            processed_images = 0

            def _process_frame(frame: np.ndarray, frame_id: str, z_index: int) -> Frame:
                nonlocal processed_images
                self.statusbar.showMessage(f"Processing {processed_images + 1}/{total_images} images")
                self._set_progress_bar(0.0)
                QtWidgets.QApplication.processEvents()
                # Per-frame routine: hydrate original image, run processing, capture outputs.
                self._reset_internal_state(preserve_original=False, reset_view=False)
                self.current_frame.begin_frame(frame, str(frame_id), int(z_index))
                self.current_image_name = self._frame_base_name(frame_id, z_index)
                if compute_measurements:
                    self.analyze_fibers()
                else:
                    self.segment()
                result = self.current_frame.snapshot(
                    include_measurements=compute_measurements,
                    frame_id=str(frame_id),
                    z_index=int(z_index),
                )
                processed_images += 1
                self._batch_processed_samples = processed_images
                self._set_progress_bar(0.0)
                return result

            try:
                batch_result = batch_analyzer.analyze(active_batch, _process_frame)
                batch_result.stack3d_meta = {
                    "info": "3D post-processing not run yet. Click 'Analyze 3D' to merge labels across slices."
                }
            finally:
                self._suppress_batch_dialogs = previous_batch_state
                self._is_batch_processing = previous_processing_state
        except Exception as exc:
            self._show_error(title, str(exc))
            logger.error("%s: %s", title, exc)
            return None
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
            self.statusbar.clearMessage()
            self._batch_total_samples = 0
            self._batch_processed_samples = 0
            self._batch_collects_measurements = False

        # Publish batch outputs back into the GUI and refresh the active slice.
        self._batch_result = batch_result
        self._batch_results_have_measurements = bool(compute_measurements)
        if compute_measurements:
            self._last_batch_measurements_signature = self._current_batch_measurement_signature()
        else:
            self._last_batch_measurements_signature = None
            self._stack_diameter_histogram_data = None
            self._stack_angle_histogram_data = None
            self._stack_diameter_histogram_enabled = False
        self._set_current_slice(self._current_z_index, update_view=True)
        self.action_save_masks_predictions.setEnabled(True)
        if compute_measurements:
            self.action_save_measurements.setEnabled(True)
        logger.info(
            "Batch %s completed successfully for %d frame(s).",
            "analysis" if compute_measurements else "segmentation",
            len(active_batch),
        )
        return batch_result

    @staticmethod
    def _batch_export_config_from_options(options: Dict[str, bool]) -> Dict[str, Any]:
        default_flags = {
            "dt": True,
            "labels": True,
            "overlay": True,
            "debug": True,
            "samples": True,
            "style_vector": False,
        }
        artifacts = tuple(default_flags.keys())
        excluded = [
            name
            for name in artifacts
            if not bool(options.get(name, default_flags[name]))
        ]
        return {
            "preset": "full",
            "image_format": "png",
            "naming": "frame_id_z",
            "collision": "suffix",
            "exclude": excluded,
            "write_manifest": True,
        }

    def _prompt_batch_save_options(self) -> Dict[str, bool] | None:
        dialog = BatchSaveOptionsDialog(self, self._batch_save_options)
        result = dialog.exec()
        if result == QtWidgets.QDialog.DialogCode.Accepted:
            options = dialog.options()
            self._batch_save_options = options
            return options
        return None

    def _on_batch_processing(self):
        self._run_batch_processing_for_folder(None)

    def _run_batch_processing_for_folder(self, folder_path: str | None):
        image_batch = self.open_source(path=folder_path, source_kind="folder")
        if image_batch is None:
            return

        save_options = self._prompt_batch_save_options()
        if not save_options:
            return

        # Use self.config instead of deleted helpers
        prev_strict_mode = self.config.strict_centerlines
        prev_seg_mode = self.config.segmentation_mode


        batch_strict_mode = bool(save_options.get("strict", prev_strict_mode))
        self._set_strict_mode(batch_strict_mode)
        if batch_strict_mode and prev_seg_mode != "ridge":
            self.rb_ridge.setChecked(True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        directory = self.working_dir
        output_root = os.path.join(directory, f"fiberpose_batch_{timestamp}")
        os.makedirs(output_root, exist_ok=True)

        batch_result: BatchResult | None = None
        try:
            batch_result = self._run_batch_analysis(
                compute_measurements=True,
                image_batch=image_batch,
                error_title="Batch processing failed",
            )
        finally:
            # Restore previous state using config and UI setters
            self._set_strict_mode(prev_strict_mode)
            if prev_seg_mode != self.config.segmentation_mode:
                self._set_segmentation_mode(prev_seg_mode)
                
        if batch_result is None:
            return

        export_config = self._batch_export_config_from_options(save_options)
        export_result = batch_result.save_all(output_root, export_config)

        failures: list[tuple[str, str]] = []
        processed = 0
        for result in batch_result.frame_results:
            if not result.has_segmentation:
                failures.append((result.frame_id, "Segmentation did not produce results"))
            else:
                processed += 1
        export_errors = export_result.get("errors", [])
        if export_errors:
            failures.extend(
                (
                    err.get("frame_id", f"frame_{err.get('frame_index', '?')}"),
                    f"Export {err.get('artifact', 'artifact')} failed: {err.get('error', 'unknown error')}",
                )
                for err in export_errors
            )

        summary_lines = [
            f"Processed {processed} of {len(image_batch)} frames.",
            f"Saved {export_result.get('written_count', 0)} file(s) in: {output_root}",
        ]
        if failures:
            summary_lines.append("Failures:")
            for path, reason in failures[:10]:
                summary_lines.append(f"  - {os.path.basename(str(path))}: {reason}")
            if len(failures) > 10:
                summary_lines.append(f"  ... ({len(failures) - 10} more)")

        summary = "\n".join(summary_lines)
        if failures:
            self._show_warning("Batch processing complete with issues", summary)
        else:
            self._show_info("Batch processing complete", summary)

    def _on_open_3d_view(self):
        if self._batch_result is None:
            self._show_info("3D view", "Run segmentation or analysis on a stack first.")
            return
        if len(self._batch_result.frame_results) < 2:
            self._show_info("3D view", "3D view requires more than one slice.")
            return
        self.sync_ui_to_sample_properties()
        voxel_spacing_zyx = self.sample_props.spacing_zyx()
        max_display_voxels = max(1, int(self.config.render3d_max_display_voxels))
        max_display_edge = max(1, int(self.config.render3d_max_display_edge))
        if self._stack3d_window is None:
            self._stack3d_window = Stack3DWindow(
                self._batch_result,
                parent=self,
                voxel_spacing_zyx=voxel_spacing_zyx,
                max_display_voxels=max_display_voxels,
                max_display_edge=max_display_edge,
            )
        else:
            self._stack3d_window.set_display_limits(
                max_display_voxels=max_display_voxels,
                max_display_edge=max_display_edge,
            )
            self._stack3d_window.update_batch_result(
                self._batch_result,
                voxel_spacing_zyx=voxel_spacing_zyx,
            )
        self._stack3d_window.show()
        self._stack3d_window.raise_()
        self._stack3d_window.activateWindow()

    def _on_analyze_3d_clicked(self):
        if self._image_batch is None or len(self._image_batch) == 0:
            self._show_info("Analyze 3D", "Load a stack first.")
            return
        if len(self._image_batch) < 2:
            self._show_info("Analyze 3D", "3D analysis requires at least two slices.")
            return

        self.sync_ui_to_config()
        self.sync_ui_to_sample_properties()
        voxel_spacing_zyx = self.sample_props.spacing_zyx()

        if not self._batch_measurements_match_current_settings():
            self._stack_diameter_histogram_data = None
            self._stack_angle_histogram_data = None
            self._stack_diameter_histogram_enabled = False
            batch_result = self._run_batch_analysis(
                compute_measurements=True,
                error_title="Analyze 3D failed during 2D batch analysis",
            )
            if batch_result is None:
                return
        elif self._batch_result is None:
            self._show_warning("Analyze 3D", "No batch result available for 3D processing.")
            return

        pooled_diameters = self._collect_stack_diameter_samples(self._batch_result)
        pooled_angles = self._collect_stack_angle_samples(self._batch_result)
        if pooled_diameters.size:
            self._stack_diameter_histogram_data = pooled_diameters
            self._stack_angle_histogram_data = pooled_angles
            self._stack_diameter_histogram_enabled = True
        else:
            self._stack_diameter_histogram_data = None
            self._stack_angle_histogram_data = None
            self._stack_diameter_histogram_enabled = False

        # Local 3D mask cleanup knobs (kept out of global app config on purpose).
        stack3d_open_iters = 1
        stack3d_close_iters = 1
        stack3d_min_component_voxels = 50
        stack3d_connectivity = 1

        QtWidgets.QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            stack_analyzer_3d = StackAnalyzer_3D(
                voxel_spacing_zyx=voxel_spacing_zyx,
                open_iters=stack3d_open_iters,
                close_iters=stack3d_close_iters,
                min_component_voxels=stack3d_min_component_voxels,
                connectivity=stack3d_connectivity,
            )
            self._batch_result = stack_analyzer_3d.process(
                self._batch_result,
                force=True,
            )
        except Exception as exc:
            logger.warning("3D post-processing failed: %s", exc)
            self._show_error("Analyze 3D failed", str(exc))
            return
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

        stack3d_meta = self._batch_result.stack3d_meta or {}
        if stack3d_meta.get("error"):
            self._show_warning("Analyze 3D", f"3D processing failed: {stack3d_meta.get('error')}")
            return

        semantic_volume = self._batch_result.semantic_volume
        semantic_skeleton_volume = self._batch_result.semantic_skeleton_volume
        if semantic_volume is None or semantic_skeleton_volume is None:
            self._show_warning(
                "Analyze 3D",
                "3D processing finished, but semantic volume or semantic skeleton is missing.",
            )
            return
        semantic_arr = np.asarray(semantic_volume)
        skeleton_arr = np.asarray(semantic_skeleton_volume)
        if semantic_arr.ndim != 3 or skeleton_arr.ndim != 3:
            self._show_warning(
                "Analyze 3D",
                "3D processing produced invalid dataset dimensionality; expected 3D volumes.",
            )
            return
        if semantic_arr.shape != skeleton_arr.shape:
            self._show_warning(
                "Analyze 3D",
                "Semantic volume and semantic skeleton volume shapes do not match.",
            )
            return

        self._set_current_slice(self._current_z_index, update_view=True)
        if self._stack3d_window is not None:
            self._stack3d_window.update_batch_result(
                self._batch_result,
                voxel_spacing_zyx=voxel_spacing_zyx,
            )
        self.action_save_masks_predictions.setEnabled(True)

        method = stack3d_meta.get("skeleton_method")
        semantic_labels = int(stack3d_meta.get("semantic_num_labels", int(np.max(semantic_arr)) if semantic_arr.size else 0))
        skeleton_voxels = int(stack3d_meta.get("num_skeleton_voxels", int(np.count_nonzero(skeleton_arr)) if skeleton_arr.size else 0))
        diameter_samples = int(pooled_diameters.size)
        if method:
            self.statusbar.showMessage(
                (
                    f"3D analysis complete ({method}). "
                    f"semantic labels={semantic_labels}, skeleton voxels={skeleton_voxels}, "
                    f"stack diameter samples={diameter_samples}."
                ),
                5000,
            )
        else:
            self.statusbar.showMessage(
                (
                    f"3D analysis complete. semantic labels={semantic_labels}, "
                    f"skeleton voxels={skeleton_voxels}, stack diameter samples={diameter_samples}."
                ),
                5000,
            )

    # ------------------- Profile plot tool -------------------

    def _ensure_profile_dialog(self) -> PlotProfileDialog:
        if self._profile_dialog is None:
            self._profile_dialog = PlotProfileDialog(self)
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
        if not self.current_frame.has_prediction or len(self._profile_points_img) < 2:
            return
        (x0, y0), (x1, y1) = self._profile_points_img[:2]
        length_px = float(np.hypot(x1 - x0, y1 - y0))
        if length_px <= 0:
            return
        profile_vals = profile_line(self.current_frame.dt, (y0, x0), (y1, x1), mode="reflect", order=1)
        if profile_vals is None or profile_vals.size == 0:
            return

        profile_vals = np.asarray(profile_vals, dtype=np.float32, copy=False)
        scale = float(self.current_frame.scale_factor)
        if not np.isfinite(scale) or scale <= 0:
            scale = 1.0
        true_length_px = length_px / scale
        # DT magnitudes are in processing-pixel units; map them back to original scale.
        profile_vals_true = profile_vals / scale
        distances = np.linspace(0.0, true_length_px, profile_vals.size, dtype=np.float32)
        delta = float(profile_vals_true[-1] - profile_vals_true[0])
        summary = (
            f"Length: {true_length_px:.2f} px · ΔDT: {delta:+.2f} · "
            f"Min/Max: {float(profile_vals_true.min()):.2f} / {float(profile_vals_true.max()):.2f}"
        )
        dialog = self._ensure_profile_dialog()
        dialog.update_profile(distances, profile_vals_true, summary)

    # ------------------- Segmentation -------------------

    def segment(self):
        fr = self.current_frame

        if self._image_batch is not None and not self._is_batch_processing:
            self._run_batch_analysis(compute_measurements=False)
            return
        if fr.orig_img is None:
            self._show_info("Load original", "Please load an original image first.")
            logger.info("Segment: No original image loaded.")
            return
        if self.analyzer is None:
            self._show_error("Analyzer unavailable", "Fiber analyzer has not been initialized.")
            logger.error("Segmentation failed: analyzer not initialized.")
            return

        fr.error_message = None
        fr.clear_segmentation()
        fr.clear_measurements()

        # Ensure config is up to date first
        self.sync_ui_to_config()
        
        previous_scale = float(fr.scale_factor)
        # Parse the string (e.g., "1x" -> 1.0)
        try:
            target_scale = float(self.config.scale_label.lower().replace("x", "").replace("×", ""))
        except ValueError:
            target_scale = 1.0
            
        scale_changed = not (previous_scale == target_scale)

        if scale_changed:
            self.analyzer.reset_prediction_state()
            fr.clear_prediction()

        
        processing_img = fr.orig_img
        if 1 == target_scale:
            fr.scale_factor = 1.0
            fr.orig_img_scaled = None
        else:
            if fr.orig_img_scaled is None or scale_changed:
                interp = cv2.INTER_AREA if target_scale < 1.0 else cv2.INTER_CUBIC
                try:
                    fr.orig_img_scaled = cv2.resize(
                        fr.orig_img,
                        None,
                        fx=target_scale,
                        fy=target_scale,
                        interpolation=interp,
                    )
                except Exception as e:
                    self._show_error("Scaling failed", f"Unable to scale image:\n{e}")
                    logger.error("Scaling failed: %s", e)
                    return
            processing_img = fr.orig_img_scaled
        fr.scale_factor = float(target_scale)
        self.analyzer.scale_factor = fr.scale_factor

        # We already called sync_ui_to_config() above, so self.config is ready.
        max_angle = self.config.max_angle_deg
        
        # Centerline kwargs aren't exposed in the basic UI, so we just use the analyzer's defaults
        centerline_params = getattr(self.analyzer, "_centerline_kwargs", None)
        if not centerline_params:
            centerline_params = FiberAnalyzer_2D._default_centerline_kwargs()

        self._set_progress_bar(0.0)

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
                fr.fail(str(e))
                logger.error("Segmentation failed: %s", e)
                return

        self._set_progress_bar(40.0)

        try:
            self.analyzer.process_centerlines(
                max_angle_deg=max_angle,
                **centerline_params,
            )
        except Exception as e:
            self._show_error("Centerline computation failed", str(e))
            fr.fail(str(e))
            logger.error("Centerline computation failed: %s", e)
            return

        self._set_progress_bar(50.0)

        dt = self.analyzer.get_dt()
        gx, gy = self.analyzer.get_gradients()
        if dt is None or gx is None or gy is None:
            self._show_error("Segmentation failed", "Distance transform or gradients unavailable.")
            fr.fail("Distance transform or gradients unavailable.")
            logger.error("Segmentation failed: DT or gradients unavailable.")
            return

        dt_norm = self.analyzer.get_dt_norm()
        lapl = self.analyzer.get_laplacian()
        style_vec = self.analyzer.get_style_vector()
        fr.set_prediction_outputs(
            dt=np.asarray(dt, dtype=np.float32, copy=False),
            gx=np.asarray(gx, dtype=np.float32, copy=False),
            gy=np.asarray(gy, dtype=np.float32, copy=False),
            dt_u8=self.analyzer.get_dt_u8(),
            dt_normalized=dt_norm,
            lapl_img=lapl,
            style_vec=style_vec,
            orig_img_scaled=fr.orig_img_scaled,
            scale_factor=fr.scale_factor,
            invalidate_downstream=True,
        )

        mode = self.config.segmentation_mode
        thr_u8 = self.config.seg_threshold

        if mode == "ridge":
            current_scale = fr.scale_factor if fr.scale_factor and np.isfinite(fr.scale_factor) else 1.0
            # If the user is in Ridge mode, they likely expect a small radius (e.g., 2-10px)
            # We can map the 0-255 threshold to a 0-25px radius for better UI feel
            radius_val = (thr_u8 / 255.0) * 25.0 
            
            seg_params = {
                "radius_thresh_px": float(radius_val) * current_scale,
                "dilate_extra_px": 0.0,
                "min_run_points": 1,
            }
        elif mode == "fastest_ascend":
            seg_params = {
                "sigma": 1.0,               # Default
                "dist_thresh": 0.2,         # Default
                "flux_merge_thresh": 5.0,   # Default
            }
        else:
            seg_params = {"threshold_u8": thr_u8}


        labels = None

        try:
            labels = self.analyzer.segment(mode=mode, params=seg_params)
        except Exception as e:
            self._show_error("Segmentation failed", str(e))
            fr.fail(str(e))
            logger.error("Segmentation failed: %s", e)
            return

        if labels is not None:
            labels_rgb = labels_to_color(labels)
            labels_cache = (id(labels), labels_rgb)
        else:
            labels_rgb = None
            labels_cache = (None, None)

        base_img = fr.orig_img if fr.scale_factor == 1.0 or fr.orig_img_scaled is None else fr.orig_img_scaled
        fr.set_segmentation_outputs(
            labels_2d=labels,
            centerlines_overlay_img=self.analyzer.get_centerlines_image(),
            overlay_img=self._create_overlay(base_img, labels, alpha=0.5, labels_rgb=labels_rgb),
            sampling_overlay=fr.sampling_overlay,
            measurements_overlay=fr.measurements_overlay,
            labels_rgb_cache=labels_cache,
            invalidate_measurements=True,
        )

        if self.cb_view.currentText() == "Original":
            self.cb_view.setCurrentText("Overlay")

        self.action_save_masks_predictions.setEnabled(True)

        self.refresh_view()
        if self._is_batch_processing and not self._batch_collects_measurements:
            self._set_progress_bar(100.0)
        else:
            self._set_progress_bar(70.0)
        logger.info("Segmentation completed successfully.")

    def analyze_fibers(self):
        """
        Computes thickness samples via analyzer and renders overlays/plots.
        """
        fr = self.current_frame
        if self._image_batch is not None and not self._is_batch_processing:
            if self._batch_result is not None and self._batch_results_have_measurements:
                self._set_current_slice(self._current_z_index, update_view=True)
                return
            self._run_batch_analysis(compute_measurements=True)
            return
        
        self.sync_ui_to_config()
        self.sync_ui_to_sample_properties()
        fr.error_message = None
        fr.clear_measurements()
        
        # We need a quick way to convert the config's "1x" string to a float 
        try:
            target_scale = float(self.config.scale_label.lower().replace("x", "").replace("×", ""))
        except ValueError:
            target_scale = 1.0
    
        current_scale = float(fr.scale_factor)
        scale_mismatch = (
            self.analyzer is not None
            and fr.has_segmentation
            and self.analyzer.dt_predicted()
            and not math.isclose(current_scale, target_scale, rel_tol=1e-6, abs_tol=1e-6)
        )

        if scale_mismatch or not fr.has_segmentation:
            if scale_mismatch:
                logger.info(
                    "Scale factor changed from %.3f to %.3f; rerunning segmentation for analysis.",
                    current_scale,
                    target_scale,
                )
            self.segment()
            if not fr.has_segmentation:
                self._show_info("No segmentation", "Segmentation could not be performed.")
                return
        logger.info("Starting fiber analysis.")
        fr.overlay_samples = []

        # Params
        spacing_px_input = max(1.0, self.config.sample_spacing)
        current_scale = fr.scale_factor if fr.scale_factor and np.isfinite(fr.scale_factor) else 1.0
        spacing_px_scaled = max(1.0, spacing_px_input * current_scale)
        
        pixel_size = self.sample_props.effective_pixel_size_xy()
        units_label = self.sample_props.unit if self.sample_props.unit else "px"

        # compute via analyzer
        if self.analyzer is None:
            self._show_info("No analyzer", "Segmentation data unavailable.")
            return

        if not self.analyzer.get_centerline_segments():
            self._show_info("No centerlines", "No centerlines available. Run segmentation first.")
            return

        self.analyzer.scale_factor = fr.scale_factor

        radius_samples, angles_samples_deg, overlay_samples = self.analyzer.analyze(
            spacing_px=spacing_px_scaled
        )
        fr.overlay_samples = overlay_samples

        self._set_progress_bar(80.0)

        # Build sample table and stats (keep in GUI)
        if radius_samples:
            radii_px        = np.array([c[2] for c in radius_samples], dtype=np.float32)
            thickness_px    = 2.0 * radii_px
            thickness_units = thickness_px * pixel_size
            xs = np.array([c[1] for c in radius_samples], dtype=np.int32)
            ys = np.array([c[0] for c in radius_samples], dtype=np.int32)
            samples_table = np.column_stack([xs, ys, thickness_px, np.array(angles_samples_deg, dtype=np.float32)]).astype(object)
        else:
            thickness_units = np.array([], dtype=np.float32)
            samples_table = None

        angles_arr = np.array([a for a in angles_samples_deg if np.isfinite(a)], dtype=np.float32)

        # --- pore splitting + sampling ------------------------------------------
        thr_u8 = self.config.seg_threshold

        # Build void mask from your chosen source (DT or grayscale-based threshold).
        if fr.void_mask is None:
            fr.void_mask, fr.pore_dt = self.analyzer.void_mask_and_pore_dt(thr_u8)

        porosity = float(fr.void_mask.mean()) if fr.void_mask is not None else float("nan")

        # --- NEW: sample pores as inscribed circles from DT peaks ---------------------
        min_radius_px = 1.5 * current_scale   # tweakable; ignores tiny specks
        peak_footprint     = 5     # local-max window; 3..7 works; larger -> fewer peaks
        max_pores     = None  # or set an upper cap for speed/UI clarity
        min_peak_px = 3.0 * current_scale

        pores = self.analyzer.sample_inscribed_circles(
            pore_dt=fr.pore_dt,
            void_mask=fr.void_mask,
            min_radius_px=min_radius_px,
            min_peak_px=min_peak_px,          # <- NEW param
            peak_footprint=peak_footprint,
            max_pores=max_pores,
        )

        # convenience arrays for hist/units
        # pixel_size is loaded from per-sample properties above
        pore_scale = fr.scale_factor if fr.scale_factor and np.isfinite(fr.scale_factor) else 1.0

        if pore_scale == 0:
            pore_scale = 1.0
        pore_diams_units = np.array(
            [2.0 * p.get('r_px_orig', p['r_px'] / pore_scale) * pixel_size for p in pores],
            dtype=np.float32,
        )




        # Summary text
        if thickness_units.size:
            mean_thickness = np.mean(thickness_units)
            std_thickness  = np.std(thickness_units)
        else:
            mean_thickness = float('nan')
            std_thickness  = float('nan')
        orientation_index, mean_angle_deg, std_angle_deg = self._compute_orientation_stats(angles_arr)

        info_text = (
            f"Mean diameter: {mean_thickness:.2f} {units_label}\n"
            f"Std diameter: {std_thickness:.2f} {units_label}\n"
            f"Orientation index: {orientation_index:.2f}\n"
            f"Mean angle: {mean_angle_deg:.2f} deg\n"
            f"Std angle: {std_angle_deg:.2f} deg\n"
            f"Porosity (area fraction): {porosity*100:.1f} %"
        )
        fiber_summary = {
            "mean_diameter": float(mean_thickness),
            "std_diameter": float(std_thickness),
            "orientation_index": float(orientation_index),
            "mean_angle_deg": float(mean_angle_deg),
            "std_angle_deg": float(std_angle_deg),
            "porosity": float(porosity),
            "sample_count": len(radius_samples) if radius_samples else 0,
            "units": units_label,
        }
        fr.set_measurement_outputs(
            samples_table=samples_table,
            fiber_summary=fiber_summary,
            overlay_samples=overlay_samples,
            angles_samples_deg=angles_arr,
            thickness_units=thickness_units,
            porosity=porosity,
            pore_diams_units=pore_diams_units,
            void_mask=fr.void_mask,
            pore_dt=fr.pore_dt,
            pores=pores,
            measurements_overlay=fr.measurements_overlay,
            sampling_overlay=fr.sampling_overlay,
        )
        self.text_edit2.setPlainText(info_text)

        # Overlay + plots
        self._render_vector_overlay()
        self._update_distributions_pyqtgraph(units_label)
        self._set_progress_bar(100.0)

        self._show_info(
            "Analysis complete",
            f"Radius samples: {len(radius_samples)} (spacing={int(spacing_px_input)} px)\n"
            f"Stored overlay samples: {len(fr.overlay_samples)}\n"
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
        if not self.current_frame.overlay_samples:
            return
        show_diam   = getattr(self, "rb_show_diam", None) and self.rb_show_diam.isChecked()
        show_direct = getattr(self, "rb_show_direct", None) and self.rb_show_direct.isChecked()
        show_pores  = getattr(self, "rb_show_pores", None) and self.rb_show_pores.isChecked()

        if not show_diam and not show_direct and not show_pores:
            show_diam = True

        if show_pores:
            base = self._pore_dt_to_rgb(self.current_frame.pore_dt, self.current_frame.void_mask)
            if self.current_frame.pores:
                out = base.copy()
                for p in self.current_frame.pores:
                    x, y = int(p['x']), int(p['y'])
                    r = int(round(p['r_px']))
                    if r <= 0:
                        continue
                    self._draw_circle(out, (x, y), r, color=(255, 255, 255), thickness=1)  
                    # optional tiny center mark for readability (kept 1 px too)
                    # draw_circle(out, (x, y), max(1, r // 12), color=(0, 255, 255), thickness=-1)
                self._set_pixmap_from_array(out)
                return

        sampling_base = ensure_rgb_u8(self.current_frame.dt_u8 if self.current_frame.dt_u8 is not None else self.current_frame.dt_normalized)
        if sampling_base is None and self.current_frame.overlay_img is not None:
            sampling_base = ensure_rgb_u8(self.current_frame.overlay_img)
        if sampling_base is None:
            return

        base_img = self.current_frame.orig_img if self.current_frame.scale_factor == 1.0 or self.current_frame.orig_img_scaled is None else self.current_frame.orig_img_scaled
        measurement_base = self.current_frame.overlay_img.copy() if self.current_frame.overlay_img is not None else ensure_rgb_u8(base_img)
        sampling_canvas = sampling_base.copy()
        # For angle coloring paint directly over the original image
        measurement_canvas = (
            ensure_rgb_u8(base_img).copy()
            if show_direct
            else (measurement_base.copy() if measurement_base is not None else sampling_canvas.copy())
        )

        H, W = sampling_canvas.shape[:2]
        MAX_SEGMENTS = 15000
        pts = self.current_frame.overlay_samples if len(self.current_frame.overlay_samples) <= MAX_SEGMENTS else self.current_frame.overlay_samples[::int(np.ceil(len(self.current_frame.overlay_samples)/MAX_SEGMENTS))]

        line_color = (255, 255, 0)
        point_color = (255, 0, 0)
        sample_coords: list[tuple[int, int]] = []
        sample_colors: list[tuple[int, int, int]] = []

        def _angle_to_bgr(angle_deg: float) -> tuple[int, int, int]:
            if not np.isfinite(angle_deg):
                return (200, 200, 200)
            ang = float(np.clip(angle_deg, -90.0, 90.0))
            hue = (ang + 90.0) / 180.0 * 179.0  # OpenCV HSV hue range is [0,179]
            hsv = np.uint8([[[hue, 255, 255]]])
            bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
            return int(bgr[0]), int(bgr[1]), int(bgr[2])

        for idx, (y, x, r, ny, nx, ty, tx, ang_deg) in enumerate(pts):
            if not (0 <= y < H and 0 <= x < W) or r <= 0:
                continue
            if show_direct:
                color = _angle_to_bgr(float(ang_deg))
                p0 = (y - ny * r, x - nx * r)
                p1 = (y + ny * r, x + nx * r)
                self._draw_line(sampling_canvas, p0, p1, color=color)
                self._draw_line(measurement_canvas, p0, p1, color=color)
                sample_coords.append((y, x))
                sample_colors.append(color)
                draw_point_color = color
            elif show_diam:
                p0 = (y - ny * r, x - nx * r)
                p1 = (y + ny * r, x + nx * r)
                self._draw_line(sampling_canvas, p0, p1, color=line_color)
                if (idx % 10) == 0:
                    self._draw_line(measurement_canvas, p0, p1, color=line_color)
                draw_point_color = point_color
            else:
                draw_point_color = point_color
            self._draw_circle(
                sampling_canvas,
                (int(round(x)), int(round(y))),
                radius_px=1,
                color=draw_point_color,
                thickness=-1,
            )

        # Fill watershed regions by nearest angle sample
        if show_direct and self.current_frame.labels_2d is not None and sample_coords:
            mask = (self.current_frame.labels_2d.astype(np.int32) > 0)
            if mask.any():
                ys, xs = np.nonzero(mask)
                coords = np.stack([ys, xs], axis=1)
                tree = cKDTree(np.asarray(sample_coords, dtype=np.float32))
                _, nn_idx = tree.query(coords, k=1)
                colors_arr = np.asarray(sample_colors, dtype=np.uint8)
                fill_vals = colors_arr[nn_idx]
                measurement_canvas[ys, xs] = fill_vals

        self.current_frame.sampling_overlay = sampling_canvas
        self.current_frame.measurements_overlay = measurement_canvas
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

    @staticmethod
    def _default_grad_quality_points() -> tuple[float, float, float, float]:
        return (-0.35, -0.08, 0.08, 0.35)

    def _validated_grad_quality_points(self) -> tuple[float, float, float, float]:
        default = self._default_grad_quality_points()
        raw = getattr(self.config, "grad_quality_points", default)
        try:
            arr = np.asarray(raw, dtype=np.float32).reshape(-1)
        except Exception:
            logger.warning("Invalid grad_quality_points=%r. Using default %s", raw, default)
            return default

        if arr.size != 4 or not np.all(np.isfinite(arr)):
            logger.warning("Invalid grad_quality_points=%r. Using default %s", raw, default)
            return default

        p0, p1, p2, p3 = [float(v) for v in arr]
        if not (p0 < p1 < p2 < p3):
            logger.warning(
                "grad_quality_points must be strictly increasing, got %s. Using default %s",
                [p0, p1, p2, p3],
                default,
            )
            return default
        if not (p1 <= 0.0 <= p2):
            logger.warning(
                "grad_quality_points must satisfy p1 <= 0 <= p2, got %s. Using default %s",
                [p0, p1, p2, p3],
                default,
            )
            return default
        return (p0, p1, p2, p3)

    def _signed_grad_error_rgb(self) -> np.ndarray | None:
        """
        Signed DT-gradient error view using s = |grad(dt)| - 1.
        Colors on valid mask (dt > 0):
          - s <= p0: blue
          - p0..p1: blue -> green
          - p1..p2: green
          - p2..p3: green -> red
          - s >= p3: red
        Outside valid mask is black.
        """
        dt = self.current_frame.dt
        gx = self.current_frame.gx
        gy = self.current_frame.gy
        if dt is None or gx is None or gy is None:
            return None

        dt = np.asarray(dt, dtype=np.float32)
        gx = np.asarray(gx, dtype=np.float32)
        gy = np.asarray(gy, dtype=np.float32)
        if dt.shape != gx.shape or dt.shape != gy.shape:
            logger.warning(
                "Cannot render signed grad error: shape mismatch dt=%s gx=%s gy=%s",
                dt.shape,
                gx.shape,
                gy.shape,
            )
            return None

        valid = dt > 0.0
        h, w = dt.shape
        out = np.zeros((h, w, 3), dtype=np.uint8)
        if not np.any(valid):
            return out

        mag = np.sqrt(gx * gx + gy * gy)
        signed_err = mag - 1.0
        p0, p1, p2, p3 = self._validated_grad_quality_points()

        mask_blue = valid & (signed_err <= p0)
        out[mask_blue] = (0, 0, 255)

        mask_red = valid & (signed_err >= p3)
        out[mask_red] = (255, 0, 0)

        mask_bg = valid & (signed_err > p0) & (signed_err < p1)
        if np.any(mask_bg):
            t = np.clip((signed_err[mask_bg] - p0) / (p1 - p0), 0.0, 1.0)
            rgb = np.stack(
                [
                    np.zeros_like(t),
                    255.0 * t,
                    255.0 * (1.0 - t),
                ],
                axis=-1,
            )
            out[mask_bg] = rgb.astype(np.uint8)

        mask_green = valid & (signed_err >= p1) & (signed_err <= p2)
        out[mask_green] = (0, 255, 0)

        mask_gr = valid & (signed_err > p2) & (signed_err < p3)
        if np.any(mask_gr):
            t = np.clip((signed_err[mask_gr] - p2) / (p3 - p2), 0.0, 1.0)
            rgb = np.stack(
                [
                    255.0 * t,
                    255.0 * (1.0 - t),
                    np.zeros_like(t),
                ],
                axis=-1,
            )
            out[mask_gr] = rgb.astype(np.uint8)

        return out


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
        diameters = (
            self.current_frame.thickness_units
            if self.current_frame.thickness_units is not None
            else np.array([], dtype=np.float32)
        )
        title = "Fiber thickness"
        if self._stack_diameter_histogram_enabled and self._stack_diameter_histogram_data is not None:
            diameters = self._stack_diameter_histogram_data
            title = "Fiber thickness (stack aggregate)"

        self._style_plotwidget(self.graph_diameters, title, "", f"Thickness ({units_label})")
        tu = np.asarray(diameters, dtype=np.float32)
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
            ang_title = "Fiber orientation"
            ang = (
                self.current_frame.angles_samples_deg
                if self.current_frame.angles_samples_deg is not None
                else np.array([], dtype=np.float32)
            )
            if self._stack_diameter_histogram_enabled and self._stack_angle_histogram_data is not None:
                ang = self._stack_angle_histogram_data
                ang_title = "Fiber orientation (stack aggregate)"
            self._style_plotwidget(self.graph_angles, ang_title, "", "Angle (deg, canonical)")
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
            ds = self.current_frame.pore_diams_units if self.current_frame.pore_diams_units is not None else np.array([], dtype=np.float32)
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
            img = ensure_rgb_u8(self.current_frame.orig_img)
        elif view == "DT":
            img = to_uint8_for_view(self.current_frame.dt_normalized)
        elif view == "Laplacian":
            img = to_uint8_for_view(self.current_frame.lapl_img)
        elif view == "Centerlines":
            img = self._centerlines_with_connections()
        elif view == "Segmentation":
            img = self._labels_rgb()
        elif view == "Overlay":
            img = self.current_frame.overlay_img
        elif view == "Sampling":
            img = self.current_frame.sampling_overlay if self.current_frame.sampling_overlay is not None else ensure_rgb_u8(self.current_frame.dt_u8)
        elif view == "Measurements overlay":
            fallback = self.current_frame.overlay_img if self.current_frame.overlay_img is not None else self._labels_rgb()
            img = self.current_frame.measurements_overlay if self.current_frame.measurements_overlay is not None else fallback
        elif view == "gx": 
            img = to_uint8_for_view(self.current_frame.gx)
        elif view == "gy":
            img = to_uint8_for_view(self.current_frame.gy)
        elif view == "DT grad error":
            img = self._signed_grad_error_rgb()

        self._set_pixmap_from_array(img)

    def _configure_z_scroll(self, frame_count: int):
        total = max(1, int(frame_count))
        self.z_scroll.blockSignals(True)
        self.z_scroll.setMinimum(0)
        self.z_scroll.setMaximum(total - 1)
        self.z_scroll.setSingleStep(1)
        self.z_scroll.setPageStep(1)
        self.z_scroll.setValue(0)
        self.z_scroll.blockSignals(False)
        self.z_scroll.setEnabled(total > 1)
        self._update_slice_label()

    def _update_slice_label(self):
        if self._image_batch is not None:
            total = len(self._image_batch)
            current = self._current_z_index + 1 if total > 0 else 0
            kind = self._image_batch.kind
            name = self._image_batch.name
        else:
            total = 1
            current = 0
            kind = "Image"
            name = "Unsaved"
            
        # Determine the correct terminology for the counter
        unit_label = "Slice" if kind == "Stack" else "Image"
        
        # Combine the name info (from before) with the dynamic unit label
        self._slice_label.setText(f"{kind}: {name}  |  {unit_label}: {current}/{total}")

    def _frame_base_name(self, frame_id: str, z_index: int) -> str:
        base = os.path.splitext(str(frame_id))[0]
        if self._image_batch is None:
            return base
        source_type = self._image_batch.metadata.get("source_type")
        if source_type == "stack" and len(self._image_batch) > 1:
            return f"{base}_z{z_index:03d}"
        return base

    def _apply_frame_result(self, frame_result: Frame | None):
        current_orig = self.current_frame.orig_img
        current_frame_id = self.current_frame.frame_id
        current_z_index = self.current_frame.z_index
        if frame_result is None:
            self.current_frame = Frame.empty(
                frame_id=current_frame_id,
                z_index=current_z_index,
                orig_img=current_orig,
            )
            self.text_edit2.clear()
            self.graph_diameters.clear()
            self.graph_angles.clear()
            return

        applied_frame = frame_result.snapshot()
        if applied_frame.orig_img is None:
            applied_frame.orig_img = current_orig
        self.current_frame = applied_frame

        if self.current_frame.fiber_summary:
            units_label = self.current_frame.fiber_summary.get("units", "px")
            info_text = (
                f"Mean diameter: {self.current_frame.fiber_summary.get('mean_diameter', float('nan')):.2f} {units_label}\n"
                f"Std diameter: {self.current_frame.fiber_summary.get('std_diameter', float('nan')):.2f} {units_label}\n"
                f"Orientation index: {self.current_frame.fiber_summary.get('orientation_index', float('nan')):.2f}\n"
                f"Mean angle: {self.current_frame.fiber_summary.get('mean_angle_deg', float('nan')):.2f} deg\n"
                f"Std angle: {self.current_frame.fiber_summary.get('std_angle_deg', float('nan')):.2f} deg\n"
                f"Porosity (area fraction): {self.current_frame.fiber_summary.get('porosity', float('nan'))*100:.1f} %"
            )
            self.text_edit2.setPlainText(info_text)
            self._update_distributions_pyqtgraph(units_label)
        else:
            self.text_edit2.clear()
            self.graph_diameters.clear()
            self.graph_angles.clear()

        if self.current_frame.labels_2d is not None:
            labels_rgb = labels_to_color(self.current_frame.labels_2d)
            self.current_frame.labels_rgb_cache = (id(self.current_frame.labels_2d), labels_rgb)
        else:
            self.current_frame.labels_rgb_cache = (None, None)

    def _set_current_slice(self, index: int, update_view: bool = True):
        if self._image_batch is None:
            return
        frame_count = len(self._image_batch)
        if frame_count == 0:
            return
        index = max(0, min(int(index), frame_count - 1))
        self._current_z_index = index
        if self.z_scroll.value() != index:
            self.z_scroll.blockSignals(True)
            self.z_scroll.setValue(index)
            self.z_scroll.blockSignals(False)
        frame = self._image_batch.frames[index]
        frame_id = (
            self._image_batch.metadata.get("filenames", [str(index)])[index]
            if self._image_batch.metadata.get("filenames")
            else str(index)
        )
        self.current_frame.frame_id = str(frame_id)
        self.current_frame.z_index = int(index)
        self.current_frame.orig_img = frame
        self.current_frame.orig_img_scaled = None
        self.current_image_name = self._frame_base_name(str(frame_id), index)
        self._apply_frame_result(
            self._batch_result.frame_at(index) if self._batch_result is not None else None
        )
        self._update_slice_label()
        if update_view:
            self.refresh_view()

    def _on_z_scroll_changed(self, value: int):
        if self._image_batch is None:
            return
        self._set_current_slice(int(value), update_view=True)

    def _reset_internal_state(self, preserve_original: bool = False, reset_view: bool = True):
        """Clear all derived data, images, analysis artifacts, and charts."""

        if self.analyzer is not None:
            self.analyzer.reset_prediction_state()
        if self._profile_dialog is not None:
            self._profile_dialog.close()

        preserved_orig = self.current_frame.orig_img if preserve_original else None
        preserved_frame_id = self.current_frame.frame_id if preserve_original else ""
        preserved_z = self.current_frame.z_index if preserve_original else 0
        self.current_frame = Frame.empty(
            frame_id=preserved_frame_id,
            z_index=preserved_z,
            orig_img=preserved_orig,
        )
        if not preserve_original:
            self.current_image_name = None

        self._scalebar_pending = False
        self._scalebar_points = []
        self.graphicsView.viewport().unsetCursor()

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

        if reset_view:
            self.cb_view.setCurrentText("Original")

    def _init_device_from_checkbox(self):
        """On startup: if CUDA is available, check the box and use cuda; else uncheck and use cpu."""
        cuda_ok = torch.cuda.is_available()

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
        cuda_ok = torch.cuda.is_available()
        if checked and not cuda_ok:
            QMessageBox.warning(self, "CUDA not available", "No CUDA GPU detected. Using CPU.")
            self.chb_usegpu.setChecked(False)
            self.config.use_gpu = False
        else:
            self.config.use_gpu = checked
            
        if self.analyzer is not None:
            device = torch.device('cuda' if self.config.use_gpu else 'cpu')
            self.analyzer.set_device(device)

    def _on_segmentation_mode_changed(self, checked: bool | None = None):
        if not checked: return # Ignore signals from the button being unchecked
        
        # Sync the radio button state to config
        self.sync_ui_to_config() 
        
        # Update UI states that depend on the mode
        self._update_strict_checkbox_state()
        
        if self.analyzer is not None:
            self.analyzer.set_strict_centerline_filter(self.config.strict_centerlines)

    def _on_strict_checkbox_toggled(self, checked: bool):
        self._set_strict_mode(bool(checked))

    def _on_cluster_toggled(self, checked: bool):
        # Update the config flag immediately
        self.config.cluster_enabled = bool(checked)
        if self.analyzer is not None:
            # Safely pass cluster_params (now defined in AppConfig)
            self.analyzer.enable_fiber_clustering(
                self.config.cluster_enabled, 
                **self.config.cluster_params
            )

    # ------------------- New default image loading -------------------

    def load_default_image(self):
        default_img_path = os.path.join(PROJECT_ROOT, "resources", "test.png")
        if os.path.exists(default_img_path):
            self.open_source(path=default_img_path, source_kind="file")
        else:
            logger.info("Default image not found: resources/test.png")

    # ------------------- Processing configuration -------------------
    def _set_strict_mode(self, enabled: bool) -> None:
        """Set strict centerline mode across UI, analyzer, and config."""
        self.chb_strict.blockSignals(True)
        self.chb_strict.setChecked(bool(enabled))
        self.chb_strict.blockSignals(False)
        
        self.config.strict_centerlines = bool(enabled)
        if self.analyzer is not None:
            self.analyzer.set_strict_centerline_filter(bool(enabled))

    def _update_strict_checkbox_state(self) -> None:
        """Enables the strict centerline checkbox only in Ridge segmentation mode."""
        if not hasattr(self, "config"):
            return

        # Clean check using the config object
        enable = self.config.segmentation_mode == "ridge"
        
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

    def _set_segmentation_mode(self, mode: str) -> None:
        target = str(mode or "watershed").strip().lower()
        if target == "ridge":
            self.rb_ridge.setChecked(True)
            return
        if target == "fastest_ascend" and hasattr(self, "rb_fastest_ascend"):
            self.rb_fastest_ascend.setChecked(True)
            return
        self.rb_watershed.setChecked(True)

    def sync_ui_to_sample_properties(self):
        """Reads sample-specific calibration fields from the UI."""
        if not hasattr(self, "sample_props") or self.sample_props is None:
            self.sample_props = SampleProperties.defaults()

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
            self.le_pixelsize.text(),
            self.sample_props.effective_pixel_size_xy(),
        )
        z = _parse_positive(
            self.le_voxeldepth.text(),
            self.sample_props.voxel_size_z,
        )

        self.sample_props.pixel_size_x = float(px)
        self.sample_props.pixel_size_y = float(px)
        self.sample_props.voxel_size_z = float(z)
        unit_txt = self.cb_units.currentText().strip()
        self.sample_props.unit = unit_txt if unit_txt else "px"
        self.sample_props.source = "manual"

    def sync_sample_properties_to_ui(self):
        """Updates sample-specific calibration widgets from self.sample_props."""
        if not hasattr(self, "sample_props") or self.sample_props is None:
            self.sample_props = SampleProperties.defaults()

        px = self.sample_props.effective_pixel_size_xy()
        self.le_pixelsize.setText(f"{px:.6g}")
        self.le_voxeldepth.setText(f"{self.sample_props.voxel_size_z:.6g}")

        unit_txt = self.sample_props.unit if self.sample_props.unit else "px"
        self.cb_units.blockSignals(True)
        idx = self.cb_units.findText(unit_txt)
        if idx >= 0:
            self.cb_units.setCurrentIndex(idx)
        else:
            self.cb_units.setCurrentText(unit_txt)
        self.cb_units.blockSignals(False)

    def sync_ui_to_config(self):
        """Reads the current state of the UI and stores it in self.config."""
        if not hasattr(self, "config"):
            self.config = AppConfig()

        # Model & Hardware
        self.config.checkpoint_path = self.cb_model.currentText().strip()
        self.config.use_gpu = self.chb_usegpu.isChecked()
        self.config.scale_label = self.cb_scale.currentText().strip()

        # Centerline Params
        # Config-only: preserve value loaded from config file / config editor dialog.
        self.config.max_angle_deg = float(getattr(self.config, "max_angle_deg", 30.0))
        self.config.strict_centerlines = self.chb_strict.isChecked()

        # Segmentation Mode
        if self.rb_ridge.isChecked():
            self.config.segmentation_mode = "ridge"
        elif hasattr(self, "rb_fastest_ascend") and self.rb_fastest_ascend.isChecked():
            self.config.segmentation_mode = "fastest_ascend"
        else:
            self.config.segmentation_mode = "watershed"

        self.config.seg_threshold = int(self.sp_segm_threshold.value())

        # Analysis
        # Config-only: preserve value loaded from config file / config editor dialog.
        self.config.sample_spacing = float(getattr(self.config, "sample_spacing", 5.0))

        # Clustering
        self.config.cluster_enabled = self.chb_cluster.isChecked()

    def sync_config_to_ui(self):
        """Updates the UI widgets to reflect the values in self.config."""
        if not hasattr(self, "config"):
            return

        cfg = self.config

        def set_combo_text(cb, text):
            cb.blockSignals(True)
            idx = cb.findText(text)
            if idx >= 0:
                cb.setCurrentIndex(idx)
            else:
                cb.setCurrentText(text)
            cb.blockSignals(False)

        # --- Model & Hardware ---
        set_combo_text(self.cb_model, cfg.checkpoint_path)
        
        self.chb_usegpu.blockSignals(True)
        self.chb_usegpu.setChecked(bool(cfg.use_gpu))
        self.chb_usegpu.blockSignals(False)
        
        set_combo_text(self.cb_scale, cfg.scale_label)

        # --- Centerline Params ---
        if hasattr(self, "sp_max_cent_angle"):
            self.sp_max_cent_angle.blockSignals(True)
            # PyQt6 QSpinBox strictly requires an integer. Cast the float to int.
            self.sp_max_cent_angle.setValue(int(cfg.max_angle_deg))
            self.sp_max_cent_angle.blockSignals(False)

        self.chb_strict.blockSignals(True)
        self.chb_strict.setChecked(bool(cfg.strict_centerlines))
        self.chb_strict.blockSignals(False)

        # --- Segmentation Mode ---
        self.rb_watershed.blockSignals(True)
        self.rb_ridge.blockSignals(True)
        if hasattr(self, "rb_fastest_ascend"):
            self.rb_fastest_ascend.blockSignals(True)

        if cfg.segmentation_mode == "ridge":
            self.rb_ridge.setChecked(True)
        elif cfg.segmentation_mode == "fastest_ascend" and hasattr(self, "rb_fastest_ascend"):
            self.rb_fastest_ascend.setChecked(True)
        else:
            self.rb_watershed.setChecked(True)

        self.rb_watershed.blockSignals(False)
        self.rb_ridge.blockSignals(False)
        if hasattr(self, "rb_fastest_ascend"):
            self.rb_fastest_ascend.blockSignals(False)

        self.sp_segm_threshold.blockSignals(True)
        # Cast to int to guarantee PyQt6 won't complain
        self.sp_segm_threshold.setValue(int(cfg.seg_threshold))
        self.sp_segm_threshold.blockSignals(False)

        # --- Analysis & Units ---
        if hasattr(self, "le_samspacing"):
            self.le_samspacing.setText(str(cfg.sample_spacing))

        # --- Clustering ---
        self.chb_cluster.blockSignals(True)
        self.chb_cluster.setChecked(bool(cfg.cluster_enabled))
        self.chb_cluster.blockSignals(False)
        
        # Trigger UI state updates that depend on these flags
        if hasattr(self, "_update_strict_checkbox_state"):
            self._update_strict_checkbox_state()
