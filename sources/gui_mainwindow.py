"""Main Qt window coordinating image IO, analysis workflows, and visualization state."""

import os
import math
import json
from datetime import datetime
from typing import Any, Dict
import numpy as np
import cv2

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
from skimage.segmentation import find_boundaries

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

from .gui_dialogs import (
    PlotProfileDialog,
    SaveOptionsDialog,
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
    """Orchestrate labels to color within the current module workflow.
    
    Args:
        labels (np.ndarray): Array-like data carrying labels values for analysis.
    
    Returns:
        np.ndarray: Array containing the labels to color output aligned with downstream expectations."""
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
        """Initialize main window with validated runtime dependencies and defaults.
        
        Args:
            ui_path (str): Filesystem location used by this operation.
            config_path (str): Filesystem location used by this operation."""
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
        self.chb_strict.blockSignals(True)
        self.chb_strict.setChecked(False)
        self.chb_strict.setEnabled(True)
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
        self.action_save_options.triggered.connect(self._on_edit_save_options)
        self.action_edit_config.triggered.connect(self._on_edit_config)
        self.action_load_config.triggered.connect(self._on_load_config)

        self.btn_segment.clicked.connect(self.segment)
        self.btn_analyze_2D.clicked.connect(self.analyze_fibers)
        self.btn_set_scalebar.clicked.connect(self._on_set_scalebar_clicked)
        self.btn_plot_profile.clicked.connect(self._on_plot_profile_clicked)
        self.btn_analyze_3D.clicked.connect(self._on_analyze_3d_clicked)
        self.action_batch_processing.triggered.connect(self._on_batch_processing)
        self.btn_open_3d.clicked.connect(self._on_open_3d_view)

        # Remove connection of radiobutton signals and instead hook up the combobox change
        self.cb_view.currentIndexChanged.connect(self.refresh_view)

        # Set default checkpoint
        self._init_checkpoint_selector()

        # Disable saves initially.
        self._refresh_save_actions()

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
        self.chb_invert.toggled.connect(self._on_invert_toggled)
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

        # Spinboxes
        if hasattr(self, "sp_max_cent_angle"):
            self.sp_max_cent_angle.valueChanged.connect(self.sync_ui_to_config)
        self.sp_segm_threshold.valueChanged.connect(self.sync_ui_to_config)

        # Line Edits (Text boxes)
        # Note: We use `editingFinished` instead of `textChanged`.
        if hasattr(self, "le_samspacing"):
            self.le_samspacing.editingFinished.connect(self.sync_ui_to_config)
        self.le_pixelsize.editingFinished.connect(self.sync_ui_to_sample_properties)
        if hasattr(self, "le_voxeldepth"):
            self.le_voxeldepth.editingFinished.connect(self.sync_ui_to_sample_properties)

    def _init_checkpoint_selector(self) -> None:
        """Provide an internal helper for init checkpoint selector.
        
        Returns:
            None: Result object produced by the init checkpoint selector routine."""
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
        """Handle the heckpoint selection changed event and propagate UI state updates.
        
        Args:
            index (int): Zero-based position used to select an item."""
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
        self._invalidate_prediction()

    def _discover_checkpoint_files(self) -> list[str]:
        """Provide an internal helper for discover checkpoint files.
        
        Returns:
            list[str]: List of discover checkpoint files results produced for downstream iteration."""
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
        """Provide an internal helper for current checkpoint path.
        
        Returns:
            str | None: Result object produced by the current checkpoint path routine."""
        entry = self.cb_model.currentText().strip()
        if not entry:
            return None
        if os.path.isabs(entry):
            return entry
        return os.path.join(PROJECT_ROOT, entry)


    # ------------------- Display helpers -------------------

    def _set_pixmap_from_array(self, img: np.ndarray | None):
        """Provide an internal helper for set pixmap from array.
        
        Args:
            img (np.ndarray | None): Value for img interpreted by this routine."""
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
        """Handle the Qt resize event for this widget.
        
        Args:
            event (Any): Value for event interpreted by this routine."""
        super().resizeEvent(event)
        self._update_reset_button_position()
        if not self._zoom_mode_active:
            self._fit_view_to_pixmap(force=True)

    @staticmethod
    def _numpy_to_qimage(img: np.ndarray) -> QtGui.QImage:
        """Provide an internal helper for numpy to qimage.
        
        Args:
            img (np.ndarray): Value for img interpreted by this routine.
        
        Returns:
            QtGui.QImage: Result object produced by the numpy to qimage routine."""
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
        """Provide an internal helper for fit view to pixmap.
        
        Args:
            force (bool): Boolean flag that toggles force behavior."""
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
        """Provide an internal helper for create overlay.
        
        Args:
            base_img (np.ndarray | None): Value for base img interpreted by this routine.
            labels_img (np.ndarray | None): Array-like data carrying labels img values for analysis.
            alpha (float): Numeric control parameter for alpha during computation.
            labels_rgb (np.ndarray | None): Array-like data carrying labels rgb values for analysis.
        
        Returns:
            np.ndarray | None: Array containing the create overlay output aligned with downstream expectations."""
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

    @staticmethod
    def _build_segmentation_outline_image(
        base_img: np.ndarray | None,
        labels: np.ndarray | None,
    ) -> np.ndarray | None:
        """Build the RGB outline image used by the "Segmentation outline" view."""
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

    def _segmentation_outline_view(self) -> np.ndarray | None:
        """Provide an internal helper for segmentation outline view.
        
        Returns:
            np.ndarray | None: Array containing the segmentation outline view output aligned with downstream expectations."""
        labels = self.current_frame.labels_2d
        if labels is None:
            return None

        fr = self.current_frame
        base_img = fr.orig_img if fr.scale_factor == 1.0 or fr.orig_img_scaled is None else fr.orig_img_scaled
        cache_id, cache_img = fr.segmentation_outline_cache
        current_id = id(labels)
        if cache_id != current_id or cache_img is None:
            cache_img = self._build_segmentation_outline_image(base_img, labels)
            fr.segmentation_outline_cache = (current_id, cache_img)
        return cache_img

    def _segmentation_centerlines_with_connections(self) -> np.ndarray | None:
        """
        Return segmentation-centerline view with optional clustering links.
        """
        img = self.current_frame.segmentation_centerlines_overlay_img
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

    def _measurement_centerlines_view(self) -> np.ndarray | None:
        """Return cached measurement-centerline view."""
        return ensure_rgb_u8(self.current_frame.measurement_centerlines_overlay_img)

    def _centerlines_comparison_view(self) -> np.ndarray | None:
        """Return combined centerline comparison debug view."""
        return ensure_rgb_u8(self.current_frame.centerlines_comparison_overlay_img)

    def _labels_rgb(self) -> np.ndarray | None:
        """Provide an internal helper for labels rgb.
        
        Returns:
            np.ndarray | None: Array containing the labels rgb output aligned with downstream expectations."""
        if self.current_frame.labels_2d is None:
            return None
        cache_id, cache_img = self.current_frame.labels_rgb_cache
        current_id = id(self.current_frame.labels_2d)
        if cache_id != current_id:
            cache_img = labels_to_color(self.current_frame.labels_2d)
            self.current_frame.labels_rgb_cache = (current_id, cache_img)
        return cache_img

    def _effective_scale_factor(self) -> float:
        token = str(getattr(self.config, "scale_label", "1x") or "1x").strip().lower().replace("×", "x")
        if token.endswith("x"):
            token = token[:-1]
        try:
            scale = float(token)
        except Exception:
            scale = 1.0
        if not np.isfinite(scale) or scale <= 0:
            return 1.0
        return float(scale)

    def _processing_image(self) -> np.ndarray | None:
        fr = self.current_frame
        image = fr.orig_img
        if image is None:
            return None

        scale = self._effective_scale_factor()
        if math.isclose(scale, 1.0, rel_tol=1e-9, abs_tol=1e-9):
            fr.scale_factor = 1.0
            fr.orig_img_scaled = None
            if self.analyzer is not None:
                self.analyzer.scale_factor = 1.0
            return image

        needs_rescale = (
            fr.orig_img_scaled is None
            or not math.isclose(float(fr.scale_factor), float(scale), rel_tol=1e-9, abs_tol=1e-9)
        )
        if needs_rescale:
            interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
            fr.orig_img_scaled = cv2.resize(
                image,
                None,
                fx=float(scale),
                fy=float(scale),
                interpolation=interp,
            )
        fr.scale_factor = float(scale)
        if self.analyzer is not None:
            self.analyzer.scale_factor = float(scale)
        return fr.orig_img_scaled

    def _segmentation_params(self) -> tuple[str, dict[str, Any]]:
        mode = str(getattr(self.config, "segmentation_mode", "watershed") or "watershed")
        thr_u8 = int(getattr(self.config, "seg_threshold", 1))
        if mode not in ("watershed", "radial_expansion"):
            mode = "watershed"
        params = (
            {"threshold_u8": thr_u8}
            if mode == "watershed"
            else {"expansion_factor": float(getattr(self.config, "segmentation_expansion_factor", 1.0))}
        )
        return mode, params

    def _prediction_signature(self) -> str | None:
        image = self.current_frame.orig_img
        if image is None:
            return None
        payload = {
            "image_id": int(id(image)),
            "shape": tuple(int(v) for v in image.shape),
            "dtype": str(image.dtype),
            "checkpoint_path": str(getattr(self.config, "checkpoint_path", "")),
            "invert_input": bool(self.chb_invert.isChecked()),
            "scale_factor": float(self._effective_scale_factor()),
        }
        return json.dumps(payload, sort_keys=True, default=str)

    def _segmentation_centerline_signature(self) -> str | None:
        prediction_sig = self._prediction_signature()
        if prediction_sig is None:
            return None
        payload = {
            "prediction_sig": prediction_sig,
            "cluster_enabled": bool(getattr(self.config, "cluster_enabled", False)),
            "cluster_params": dict(getattr(self.config, "cluster_params", {}) or {}),
            "merge_expansion_factor": float(getattr(self.config, "merge_expansion_factor", 0.7)),
            "centerline_kwargs": dict(
                getattr(self.analyzer, "_centerline_kwargs", None)
                or FiberAnalyzer_2D._default_centerline_kwargs()
            ),
        }
        return json.dumps(payload, sort_keys=True, default=str)

    def _measurement_centerline_signature(self) -> str | None:
        segmentation_centerline_sig = self._segmentation_centerline_signature()
        if segmentation_centerline_sig is None:
            return None
        payload = {
            "segmentation_centerline_sig": segmentation_centerline_sig,
            "max_angle_deg": float(getattr(self.config, "max_angle_deg", 30.0)),
            "strict_centerlines": bool(getattr(self.config, "strict_centerlines", False)),
        }
        return json.dumps(payload, sort_keys=True, default=str)

    def _segmentation_signature(self) -> str | None:
        segmentation_centerline_sig = self._segmentation_centerline_signature()
        if segmentation_centerline_sig is None:
            return None
        payload = {
            "segmentation_centerline_sig": segmentation_centerline_sig,
            "segmentation_mode": str(getattr(self.config, "segmentation_mode", "watershed")),
            "seg_threshold": int(getattr(self.config, "seg_threshold", 1)),
            "segmentation_expansion_factor": float(getattr(self.config, "segmentation_expansion_factor", 1.0)),
        }
        return json.dumps(payload, sort_keys=True, default=str)

    def _measurement_signature(self) -> str | None:
        measurement_centerline_sig = self._measurement_centerline_signature()
        if measurement_centerline_sig is None:
            return None
        props = self.sample_props if self.sample_props is not None else SampleProperties.defaults()
        payload = {
            "measurement_centerline_sig": measurement_centerline_sig,
            "sample_spacing": float(getattr(self.config, "sample_spacing", 5.0)),
            "seg_threshold": int(getattr(self.config, "seg_threshold", 1)),
            "pixel_size_x": float(props.pixel_size_x),
            "pixel_size_y": float(props.pixel_size_y),
            "voxel_size_z": float(props.voxel_size_z),
            "unit": str(props.unit),
        }
        return json.dumps(payload, sort_keys=True, default=str)

    def _invalidate_prediction(self) -> None:
        if self.analyzer is not None:
            self.analyzer._reset_prediction_state()
        self.current_frame.clear_prediction()
        self._invalidate_measurements()

    def _invalidate_segmentation(self, invalidate_centerlines: bool = False) -> None:
        fr = self.current_frame
        if invalidate_centerlines:
            fr.clear_segmentation()
        else:
            keep_seg_overlay = fr.segmentation_centerlines_overlay_img
            keep_meas_overlay = fr.measurement_centerlines_overlay_img
            keep_cmp_overlay = fr.centerlines_comparison_overlay_img
            keep_seg_sig = fr.segmentation_centerline_sig
            keep_meas_sig = fr.measurement_centerline_sig
            fr.clear_segmentation()
            fr.segmentation_centerlines_overlay_img = keep_seg_overlay
            fr.measurement_centerlines_overlay_img = keep_meas_overlay
            fr.centerlines_comparison_overlay_img = keep_cmp_overlay
            fr.segmentation_centerline_sig = keep_seg_sig
            fr.measurement_centerline_sig = keep_meas_sig
        self._invalidate_measurements()

    def _invalidate_measurements(self) -> None:
        self.current_frame.clear_measurements()
        self.text_edit2.clear()
        self.graph_diameters.clear()
        self.graph_angles.clear()
        self._refresh_save_actions()

    def _prediction_params(self) -> dict[str, Any]:
        return {
            "max_angle_deg": float(getattr(self.config, "max_angle_deg", 30.0)),
            "centerline_kwargs": dict(
                getattr(self.analyzer, "_centerline_kwargs", None)
                or FiberAnalyzer_2D._default_centerline_kwargs()
            ),
        }

    def _centerline_params(self) -> dict[str, Any]:
        params = self._prediction_params()
        params.update(
            {
                "strict_centerlines": bool(getattr(self.config, "strict_centerlines", False)),
                "cluster_enabled": bool(getattr(self.config, "cluster_enabled", False)),
                "cluster_params": dict(getattr(self.config, "cluster_params", {}) or {}),
                "merge_expansion_factor": float(getattr(self.config, "merge_expansion_factor", 0.7)),
                "segmentation_expansion_factor": float(getattr(self.config, "segmentation_expansion_factor", 1.0)),
            }
        )
        return params

    def _measurement_params(self) -> dict[str, Any]:
        return {
            "sample_spacing": max(1.0, float(getattr(self.config, "sample_spacing", 5.0))),
            "seg_threshold": int(getattr(self.config, "seg_threshold", 1)),
        }

    def _hydrate_analyzer_from_prediction(self, fr: Frame, *, max_angle_deg: float, centerline_kwargs: dict[str, Any]) -> None:
        if self.analyzer is None or fr.dt is None or fr.gx is None or fr.gy is None:
            return
        self.analyzer._reset_prediction_state()
        self.analyzer.scale_factor = float(fr.scale_factor)
        self.analyzer._set_dt_and_gradients(fr.dt, fr.gx, fr.gy)
        self.analyzer._laplacian = fr.lapl_img
        self.analyzer._style_vec = fr.style_vec
        self.analyzer._last_max_angle = float(max_angle_deg)
        self.analyzer._centerline_kwargs = dict(centerline_kwargs)

    def predict(self, image: np.ndarray, prediction_params: dict[str, Any]) -> dict[str, Any]:
        if self.analyzer is None:
            raise RuntimeError("Fiber analyzer has not been initialized.")
        self.analyzer.predict_dt(
            image=self._image_for_inference(image),
            max_angle_deg=float(prediction_params["max_angle_deg"]),
            centerline_kwargs=dict(prediction_params["centerline_kwargs"]),
        )
        dt = self.analyzer.get_dt()
        gx, gy = self.analyzer.get_gradients()
        if dt is None or gx is None or gy is None:
            raise RuntimeError("Distance transform or gradients unavailable.")
        return {
            "dt": np.asarray(dt, dtype=np.float32, copy=False),
            "gx": np.asarray(gx, dtype=np.float32, copy=False),
            "gy": np.asarray(gy, dtype=np.float32, copy=False),
            "dt_u8": np.asarray(self.analyzer.get_dt_u8(), dtype=np.uint8, copy=False),
            "dt_normalized": np.asarray(self.analyzer.get_dt_norm(), dtype=np.float32, copy=False),
            "lapl_img": np.asarray(self.analyzer.get_laplacian(), dtype=np.float32, copy=False),
            "style_vec": np.asarray(self.analyzer.get_style_vector(), dtype=np.float32, copy=False),
        }

    def build_centerlines(self, prediction: Frame, centerline_params: dict[str, Any]) -> dict[str, Any]:
        if self.analyzer is None:
            raise RuntimeError("Fiber analyzer has not been initialized.")
        self._hydrate_analyzer_from_prediction(
            prediction,
            max_angle_deg=float(centerline_params["max_angle_deg"]),
            centerline_kwargs=dict(centerline_params["centerline_kwargs"]),
        )
        self.analyzer.set_strict_centerline_filter(bool(centerline_params["strict_centerlines"]))
        self.analyzer.enable_fiber_clustering(
            bool(centerline_params["cluster_enabled"]),
            **dict(centerline_params["cluster_params"]),
        )
        self.analyzer.process_centerlines(
            max_angle_deg=float(centerline_params["max_angle_deg"]),
            **dict(centerline_params["centerline_kwargs"]),
            merge_expansion_factor=float(centerline_params["merge_expansion_factor"]),
            segmentation_expansion_factor=float(centerline_params["segmentation_expansion_factor"]),
        )
        return {
            "segmentation_centerlines_overlay_img": self.analyzer.get_segmentation_centerlines_image(),
            "measurement_centerlines_overlay_img": self.analyzer.get_measurement_centerlines_image(),
            "centerlines_comparison_overlay_img": self.analyzer.get_centerlines_comparison_image(),
        }

    def segment_stage(
        self,
        prediction: Frame,
        centerlines: dict[str, Any],
        segmentation_params: dict[str, Any],
    ) -> dict[str, Any]:
        if self.analyzer is None:
            raise RuntimeError("Fiber analyzer has not been initialized.")
        labels = self.analyzer.segment(
            mode=str(segmentation_params["mode"]),
            params=dict(segmentation_params["params"]),
        )
        base_img = prediction.orig_img if prediction.scale_factor == 1.0 or prediction.orig_img_scaled is None else prediction.orig_img_scaled
        if labels is not None:
            labels_rgb = labels_to_color(labels)
            labels_cache = (id(labels), labels_rgb)
            outline_cache = (id(labels), self._build_segmentation_outline_image(base_img, labels))
        else:
            labels_rgb = None
            labels_cache = (None, None)
            outline_cache = (None, None)
        return {
            "labels_2d": labels,
            "segmentation_centerlines_overlay_img": centerlines["segmentation_centerlines_overlay_img"],
            "measurement_centerlines_overlay_img": centerlines["measurement_centerlines_overlay_img"],
            "centerlines_comparison_overlay_img": centerlines["centerlines_comparison_overlay_img"],
            "overlay_img": self._create_overlay(base_img, labels, alpha=0.5, labels_rgb=labels_rgb),
            "labels_rgb_cache": labels_cache,
            "segmentation_outline_cache": outline_cache,
        }

    def _ensure_prediction_ready(self) -> bool:
        fr = self.current_frame
        if fr.orig_img is None:
            self._show_info("Load original", "Load an image before using the profile tool.")
            return False
        if self.analyzer is None:
            self._show_error("Analyzer unavailable", "Fiber analyzer has not been initialized.")
            return False

        self.sync_ui_to_config()
        target_sig = self._prediction_signature()
        if target_sig is None:
            return False
        if fr.has_prediction and fr.dt is not None and fr.prediction_sig == target_sig:
            return True

        try:
            processing_img = self._processing_image()
            if processing_img is None:
                raise RuntimeError("Missing input image.")
            prediction = self.predict(processing_img, self._prediction_params())
        except Exception as exc:
            fr.fail(str(exc))
            self._show_error("Prediction failed", str(exc))
            logger.error("Prediction ensure failed: %s", exc)
            return False

        fr.set_prediction_outputs(
            dt=prediction["dt"],
            gx=prediction["gx"],
            gy=prediction["gy"],
            dt_u8=prediction["dt_u8"],
            dt_normalized=prediction["dt_normalized"],
            lapl_img=prediction["lapl_img"],
            style_vec=prediction["style_vec"],
            orig_img_scaled=fr.orig_img_scaled,
            scale_factor=fr.scale_factor,
            invalidate_downstream=True,
        )
        fr.prediction_sig = target_sig
        return True

    def _ensure_segmentation_centerlines_ready(self) -> bool:
        fr = self.current_frame
        if not self._ensure_prediction_ready():
            return False
        if self.analyzer is None:
            return False

        segmentation_sig = self._segmentation_centerline_signature()
        measurement_sig = self._measurement_centerline_signature()
        if segmentation_sig is None or measurement_sig is None:
            return False
        if (
            fr.segmentation_centerline_sig == segmentation_sig
            and fr.measurement_centerline_sig == measurement_sig
            and self.analyzer.get_segmentation_centerline_segments()
            and self.analyzer.get_measurement_centerline_segments()
        ):
            return True

        params = self._centerline_params()
        try:
            outputs = self.build_centerlines(fr, params)
        except Exception as exc:
            fr.fail(str(exc))
            self._show_error("Centerline computation failed", str(exc))
            logger.error("Centerline computation failed: %s", exc)
            return False

        fr.segmentation_centerlines_overlay_img = outputs["segmentation_centerlines_overlay_img"]
        fr.measurement_centerlines_overlay_img = outputs["measurement_centerlines_overlay_img"]
        fr.centerlines_comparison_overlay_img = outputs["centerlines_comparison_overlay_img"]
        fr.segmentation_centerline_sig = segmentation_sig
        fr.measurement_centerline_sig = measurement_sig
        fr.segmentation_sig = None
        fr.measurement_sig = None
        return True

    def _ensure_measurement_centerlines_ready(self) -> bool:
        if not self._ensure_segmentation_centerlines_ready():
            return False
        fr = self.current_frame
        target_sig = self._measurement_centerline_signature()
        if target_sig is None:
            return False
        if (
            fr.measurement_centerline_sig == target_sig
            and self.analyzer is not None
            and self.analyzer.get_measurement_centerline_segments()
        ):
            return True
        # Recompute both centerline sets when measurement filtering settings changed.
        fr.segmentation_centerline_sig = None
        fr.measurement_centerline_sig = None
        return self._ensure_segmentation_centerlines_ready()

    def _ensure_segmentation_ready(self) -> bool:
        fr = self.current_frame
        if fr.orig_img is None:
            self._show_info("Load original", "Please load an original image first.")
            return False
        if self.analyzer is None:
            self._show_error("Analyzer unavailable", "Fiber analyzer has not been initialized.")
            return False

        self.sync_ui_to_config()
        fr.error_message = None

        target_sig = self._segmentation_signature()
        if target_sig is None:
            return False
        if fr.has_segmentation and fr.labels_2d is not None and fr.segmentation_sig == target_sig:
            return True
        if not self._ensure_segmentation_centerlines_ready():
            return False

        self._invalidate_segmentation(invalidate_centerlines=False)
        mode, params = self._segmentation_params()
        try:
            outputs = self.segment_stage(
                fr,
                {
                    "segmentation_centerlines_overlay_img": fr.segmentation_centerlines_overlay_img,
                    "measurement_centerlines_overlay_img": fr.measurement_centerlines_overlay_img,
                    "centerlines_comparison_overlay_img": fr.centerlines_comparison_overlay_img,
                },
                {"mode": mode, "params": params},
            )
        except Exception as exc:
            fr.fail(str(exc))
            self._show_error("Segmentation failed", str(exc))
            logger.error("Segmentation failed: %s", exc)
            return False

        fr.set_segmentation_outputs(
            labels_2d=outputs["labels_2d"],
            segmentation_centerlines_overlay_img=outputs["segmentation_centerlines_overlay_img"],
            measurement_centerlines_overlay_img=outputs["measurement_centerlines_overlay_img"],
            centerlines_comparison_overlay_img=outputs["centerlines_comparison_overlay_img"],
            overlay_img=outputs["overlay_img"],
            sampling_overlay=fr.sampling_overlay,
            labels_rgb_cache=outputs["labels_rgb_cache"],
            segmentation_outline_cache=outputs["segmentation_outline_cache"],
            invalidate_measurements=True,
        )
        fr.segmentation_sig = target_sig
        return True

    def _build_measurement_outputs(
        self,
        *,
        measurement_samples: list[Any],
        pores: list[dict[str, Any]],
        porosity: float,
        pixel_size: float,
        units_label: str,
        pore_scale: float,
    ) -> dict[str, Any]:
        if measurement_samples:
            radii_px = np.array([sample[4] for sample in measurement_samples], dtype=np.float32)
            angles_arr = np.array([sample[5] for sample in measurement_samples], dtype=np.float32)
            thickness_px = 2.0 * radii_px
            thickness_units = thickness_px * float(pixel_size)
            xs = np.array([sample[1] for sample in measurement_samples], dtype=np.int32)
            ys = np.array([sample[0] for sample in measurement_samples], dtype=np.int32)
            samples_table = np.column_stack([xs, ys, thickness_px, angles_arr]).astype(object)
        else:
            angles_arr = np.array([], dtype=np.float32)
            thickness_units = np.array([], dtype=np.float32)
            samples_table = None

        angles_arr = angles_arr[np.isfinite(angles_arr)] if angles_arr.size else angles_arr
        safe_scale = float(pore_scale) if np.isfinite(pore_scale) and pore_scale > 0 else 1.0
        pore_diams_units = np.array(
            [2.0 * p.get("r_px_orig", p["r_px"] / safe_scale) * float(pixel_size) for p in pores],
            dtype=np.float32,
        )

        if thickness_units.size:
            mean_thickness = float(np.mean(thickness_units))
            std_thickness = float(np.std(thickness_units))
        else:
            mean_thickness = float("nan")
            std_thickness = float("nan")
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
            "mean_diameter": mean_thickness,
            "std_diameter": std_thickness,
            "orientation_index": float(orientation_index),
            "mean_angle_deg": float(mean_angle_deg),
            "std_angle_deg": float(std_angle_deg),
            "porosity": float(porosity),
            "sample_count": len(measurement_samples),
            "units": units_label,
        }
        return {
            "samples_table": samples_table,
            "thickness_units": thickness_units,
            "pore_diams_units": pore_diams_units,
            "fiber_summary": fiber_summary,
            "info_text": info_text,
            "mean_thickness": mean_thickness,
            "std_thickness": std_thickness,
            "orientation_index": float(orientation_index),
            "mean_angle_deg": float(mean_angle_deg),
            "std_angle_deg": float(std_angle_deg),
        }

    def measure(
        self,
        prediction: Frame,
        centerlines: dict[str, Any],
        segmentation: dict[str, Any],
        measurement_params: dict[str, Any],
        sample_props: SampleProperties,
    ) -> dict[str, Any]:
        del centerlines, segmentation
        if self.analyzer is None:
            raise RuntimeError("Fiber analyzer has not been initialized.")

        spacing_px_input = float(measurement_params["sample_spacing"])
        current_scale = prediction.scale_factor if prediction.scale_factor and np.isfinite(prediction.scale_factor) else 1.0
        spacing_px_scaled = max(1.0, spacing_px_input * current_scale)
        pixel_size = sample_props.effective_pixel_size_xy()
        units_label = sample_props.unit if sample_props.unit else "px"

        self.analyzer.scale_factor = prediction.scale_factor
        measurement_samples, drawing_samples = self.analyzer.analyze(spacing_px=spacing_px_scaled)
        self._set_progress_bar(80.0)
        thr_u8 = int(measurement_params["seg_threshold"])
        void_mask, pore_dt = self.analyzer.void_mask_and_pore_dt(thr_u8)
        porosity = float(void_mask.mean()) if void_mask is not None else float("nan")
        pores = self.analyzer.sample_inscribed_circles(
            pore_dt=pore_dt,
            void_mask=void_mask,
            min_radius_px=1.5 * current_scale,
            min_peak_px=3.0 * current_scale,
            peak_footprint=5,
            max_pores=None,
        )

        outputs = self._build_measurement_outputs(
            measurement_samples=measurement_samples,
            pores=pores,
            porosity=porosity,
            pixel_size=pixel_size,
            units_label=units_label,
            pore_scale=prediction.scale_factor,
        )
        outputs.update(
            {
                "measurement_samples": measurement_samples,
                "drawing_samples": drawing_samples,
                "porosity": porosity,
                "void_mask": void_mask,
                "pore_dt": pore_dt,
                "pores": pores,
            }
        )
        return outputs

    def _ensure_measurements_ready(self) -> bool:
        fr = self.current_frame
        if not self._ensure_prediction_ready():
            return False
        if self.analyzer is None:
            self._show_info("No analyzer", "Measurement data unavailable.")
            return False

        target_sig = self._measurement_signature()
        if target_sig is None:
            return False
        if fr.has_measurements and fr.fiber_summary is not None and fr.measurement_sig == target_sig:
            return True

        self._invalidate_measurements()
        if not self._ensure_measurement_centerlines_ready() or not self.analyzer.get_measurement_centerline_segments():
            self._show_info("No centerlines", "No measurement centerlines available.")
            return False

        fr.error_message = None
        measurement_params = self._measurement_params()
        try:
            outputs = self.measure(
                prediction=fr,
                centerlines={
                    "segmentation_centerlines_overlay_img": fr.segmentation_centerlines_overlay_img,
                    "measurement_centerlines_overlay_img": fr.measurement_centerlines_overlay_img,
                    "centerlines_comparison_overlay_img": fr.centerlines_comparison_overlay_img,
                },
                segmentation={"labels_2d": fr.labels_2d},
                measurement_params=measurement_params,
                sample_props=self.sample_props,
            )
        except Exception as exc:
            self._show_error("Analysis failed", str(exc))
            fr.fail(str(exc))
            logger.error("Fiber analysis failed: %s", exc)
            return False

        fr.set_measurement_outputs(
            samples_table=outputs["samples_table"],
            fiber_summary=outputs["fiber_summary"],
            measurement_samples=outputs["measurement_samples"],
            drawing_samples=outputs["drawing_samples"],
            thickness_units=outputs["thickness_units"],
            porosity=outputs["porosity"],
            pore_diams_units=outputs["pore_diams_units"],
            void_mask=outputs["void_mask"],
            pore_dt=outputs["pore_dt"],
            pores=outputs["pores"],
            sampling_overlay=fr.sampling_overlay,
        )
        self._build_measurement_overlays()
        self.text_edit2.setPlainText(outputs["info_text"])
        units_label = str(outputs["fiber_summary"].get("units", "px"))
        self._update_distributions_pyqtgraph(units_label)
        fr.measurement_sig = target_sig
        return True

    def _show_info(self, title: str, message: str):
        """Provide an internal helper for show info.
        
        Args:
            title (str): Value for title interpreted by this routine.
            message (str): Value for message interpreted by this routine."""
        if self._suppress_batch_dialogs:
            logger.info("[Batch] %s: %s", title, message)
        else:
            QtWidgets.QMessageBox.information(self, title, message)

    def _image_for_inference(self, image: np.ndarray | None) -> np.ndarray | None:
        if image is None or not bool(self.chb_invert.isChecked()):
            return image
        try:
            return cv2.bitwise_not(image)
        except Exception:
            return np.max(image) - image

    def _show_warning(self, title: str, message: str):
        """Provide an internal helper for show warning.
        
        Args:
            title (str): Value for title interpreted by this routine.
            message (str): Value for message interpreted by this routine."""
        if self._suppress_batch_dialogs:
            logger.warning("[Batch] %s: %s", title, message)
        else:
            QtWidgets.QMessageBox.warning(self, title, message)

    def _show_error(self, title: str, message: str):
        """Provide an internal helper for show error.
        
        Args:
            title (str): Value for title interpreted by this routine.
            message (str): Value for message interpreted by this routine."""
        if self._suppress_batch_dialogs:
            logger.error("[Batch] %s: %s", title, message)
        else:
            QtWidgets.QMessageBox.critical(self, title, message)

    def _apply_runtime_after_config_change(self) -> None:
        """Provide an internal helper for apply runtime after config change.
        
        Returns:
            None: Result object produced by the apply runtime after config change routine."""
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
        """Provide an internal helper for load config from path.
        
        Args:
            path (str): Filesystem location describing where path is read or written.
        
        Returns:
            bool: True when the validation or availability check succeeds; otherwise False."""
        selected_path = os.path.abspath(path)
        try:
            self.config = AppConfig.load_from_json(selected_path, strict=True)
        except Exception as exc:
            self._show_error("Load config failed", str(exc))
            return False

        self.config_filepath = selected_path
        self._apply_runtime_after_config_change()
        return True

    def _on_load_config(self):
        """Handle the oad config event and propagate UI state updates."""
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
        """Handle the dit config event and propagate UI state updates."""
        dialog = ConfigEditorDialog(self.config, self.config_filepath, self)
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return

        self.config = dialog.result_config()
        self.config_filepath = dialog.result_config_path()
        self._apply_runtime_after_config_change()
        self.statusbar.showMessage(f"Config updated: {self.config_filepath}", 5000)

    def _on_edit_save_options(self):
        """Handle the dit save options event and propagate UI state updates."""
        current_options = self._normalize_save_options(getattr(self.config, "save_options", {}))
        dialog = SaveOptionsDialog(self, current_options)
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return

        self.config.save_options = self._normalize_save_options(dialog.options())
        self.statusbar.showMessage("Save options updated.", 4000)


    # ------------------- Drag & drop -------------------

    def _install_drag_drop(self):
        """Provide an internal helper for install drag drop."""
        v = self.graphicsView
        v.setAcceptDrops(True)
        v.viewport().setAcceptDrops(True)
        v.viewport().installEventFilter(self)

    def eventFilter(self, obj, event):
        """Orchestrate event filter within the current module workflow.
        
        Args:
            obj (Any): Value for obj interpreted by this routine.
            event (Any): Value for event interpreted by this routine.
        
        Returns:
            Any: Object containing the event filter outcome for downstream use."""
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
        """Provide an internal helper for is image path.
        
        Args:
            path (str): Filesystem location describing where path is read or written.
        
        Returns:
            bool: True when the validation or availability check succeeds; otherwise False."""
        ext = os.path.splitext(path)[1].lower()
        return ext in self._supported_image_extensions()

    def _is_loadable_path(self, path: str) -> bool:
        """Provide an internal helper for is loadable path.
        
        Args:
            path (str): Filesystem location describing where path is read or written.
        
        Returns:
            bool: True when the validation or availability check succeeds; otherwise False."""
        return os.path.isdir(path) or self._is_image_path(path)

    def _urls_are_loadable(self, urls) -> bool:
        """Provide an internal helper for urls are loadable.
        
        Args:
            urls (Any): Value for urls interpreted by this routine.
        
        Returns:
            bool: True when the validation or availability check succeeds; otherwise False."""
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

    def _current_batch_measurement_signature(self) -> str | None:
        """Provide an internal helper for current batch measurement signature.
        
        Returns:
            str | None: Result object produced by the current batch measurement signature routine."""
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
                "merge_expansion_factor": float(self.config.merge_expansion_factor),
                "segmentation_expansion_factor": float(self.config.segmentation_expansion_factor),
                "seg_threshold": int(self.config.seg_threshold),
                "sample_spacing": float(self.config.sample_spacing),
                "cluster_enabled": bool(self.config.cluster_enabled),
                "cluster_params": cluster_params,
            },
            "runtime": {
                "invert_input": bool(self.chb_invert.isChecked()),
            },
            "sample_properties": {
                "spacing_zyx": spacing_zyx,
                "unit": str(self.sample_props.unit),
            },
        }
        return json.dumps(payload, sort_keys=True, default=str)

    def _batch_measurements_match_current_settings(self) -> bool:
        """Provide an internal helper for batch measurements match current settings.
        
        Returns:
            bool: True when the validation or availability check succeeds; otherwise False."""
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
        """Provide an internal helper for collect stack diameter samples.
        
        Args:
            batch_result (BatchResult | None): Value for batch result interpreted by this routine.
        
        Returns:
            np.ndarray: Array containing the collect stack diameter samples output aligned with downstream expectations."""
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
        """Provide an internal helper for collect stack angle samples.
        
        Args:
            batch_result (BatchResult | None): Value for batch result interpreted by this routine.
        
        Returns:
            np.ndarray: Array containing the collect stack angle samples output aligned with downstream expectations."""
        if batch_result is None:
            return np.array([], dtype=np.float32)

        chunks: list[np.ndarray] = []
        for frame in batch_result.frame_results:
            values = (
                np.array([sample[5] for sample in (frame.measurement_samples or [])], dtype=np.float32)
                if frame.measurement_samples is not None
                else np.array([], dtype=np.float32)
            )
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
        """Provide an internal helper for set image batch.
        
        Args:
            image_batch (ImageBatch): Array-like data carrying image batch values for analysis."""
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
        """Provide an internal helper for apply sample properties from batch.
        
        Args:
            image_batch (ImageBatch): Array-like data carrying image batch values for analysis.
        
        Returns:
            None: Result object produced by the apply sample properties from batch routine."""
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
        """Load an image file/folder/stack source and initialize frame/batch state."""
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
        """Load original from persisted storage and normalize it for use."""
        self.open_source(source_kind="file")

    def load_original_from_path(self, path: str):
        """Load original from path from persisted storage and normalize it for use.
        
        Args:
            path (str): Filesystem location used by this operation."""
        self.open_source(path=path, source_kind="auto")

    @staticmethod
    def _default_save_options() -> Dict[str, Any]:
        """Provide an internal helper for default save options.
        
        Returns:
            Dict[str, Any]: Mapping containing named default save options outputs and metadata."""
        return dict(AppConfig().save_options)

    @classmethod
    def _normalize_save_options(cls, raw_options: Any) -> Dict[str, Any]:
        """Normalize save options into a canonical representation.
        
        Args:
            raw_options (Any): Configuration values controlling runtime behavior.
        
        Returns:
            Dict[str, Any]: Mapping of computed keys and values."""
        defaults = cls._default_save_options()
        if not isinstance(raw_options, dict):
            return defaults

        normalized = dict(defaults)
        for key in cls._save_artifact_keys():
            if key in raw_options:
                normalized[key] = bool(raw_options.get(key))

        image_format = str(raw_options.get("image_format", defaults["image_format"])).strip().lower()
        normalized["image_format"] = image_format if image_format in {"png", "tiff"} else defaults["image_format"]

        collision = str(raw_options.get("collision", defaults["collision"])).strip().lower()
        normalized["collision"] = collision if collision in {"suffix", "overwrite", "error"} else defaults["collision"]

        normalized["write_manifest"] = bool(raw_options.get("write_manifest", defaults["write_manifest"]))
        return normalized

    @staticmethod
    def _save_artifact_keys() -> tuple[str, ...]:
        """Provide an internal helper for save artifact keys.
        
        Returns:
            tuple[str, ...]: Tuple grouping the save artifact keys outputs in a stable positional order."""
        return ("dt", "labels", "overlay", "diameters", "orientations", "outline", "tables", "style_vector")

    def _current_save_layout(self) -> str:
        """Return export layout based on loaded source type."""
        if self._image_batch is None:
            return "single"
        source_type = str((self._image_batch.metadata or {}).get("source_type", "")).strip().lower()
        if source_type == "folder":
            return "frames"
        if source_type == "stack":
            return "stack"
        return "single"

    def _current_source_base_name(self) -> str:
        """Provide an internal helper for current source base name.
        
        Returns:
            str: String value representing the normalized current source base name outcome."""
        if self._image_batch is not None:
            source_path = str((self._image_batch.metadata or {}).get("source_path", "")).strip()
            if source_path:
                return os.path.splitext(os.path.basename(os.path.normpath(source_path)))[0] or "fiberpose"
        return self.current_image_name or "fiberpose"

    @staticmethod
    def _artifact_exclude_from_save_options(options: Dict[str, Any]) -> list[str]:
        """Provide an internal helper for artifact exclude from save options.
        
        Args:
            options (Dict[str, Any]): Configuration values that control how options is processed.
        
        Returns:
            list[str]: List of artifact exclude from save options results produced for downstream iteration."""
        return [name for name in MainWindow._save_artifact_keys() if not bool(options.get(name, False))]

    def _effective_batch_result_for_save(self) -> BatchResult | None:
        """Provide an internal helper for effective batch result for save.
        
        Returns:
            BatchResult | None: Result object produced by the effective batch result for save routine."""
        layout = self._current_save_layout()
        if layout != "single":
            if self._image_batch is None or self._batch_result is None:
                return None
            if len(self._batch_result.frame_results) != len(self._image_batch):
                return None
            return self._batch_result

        if self._batch_result is not None and len(self._batch_result.frame_results) == 1:
            return self._batch_result

        if self.current_frame.orig_img is None:
            return None

        frame_result = self.current_frame.snapshot(
            include_measurements=True,
            frame_id=str(self.current_frame.frame_id),
            z_index=int(self.current_frame.z_index),
        )
        return BatchResult(frame_results=[frame_result])

    def _build_export_config(
        self,
        *,
        kind: str,
        layout: str,
        save_options: Dict[str, Any],
        base_name: str | None = None,
        image_format_override: str | None = None,
    ) -> Dict[str, Any]:
        """Provide an internal helper for build export config.
        
        Args:
            kind (str): Value for kind interpreted by this routine.
            layout (str): Value for layout interpreted by this routine.
            save_options (Dict[str, Any]): Configuration values that control how save options is processed.
            base_name (str | None): Value for base name interpreted by this routine.
            image_format_override (str | None): Array-like data carrying image format override values for analysis.
        
        Returns:
            Dict[str, Any]: Mapping containing named build export config outputs and metadata.
        
        Raises:
            ValueError: If provided values violate validation or domain constraints."""
        if kind not in {"results", "csv"}:
            raise ValueError(f"Unsupported save kind: {kind}")
        image_format = image_format_override or str(save_options.get("image_format", "png"))
        export_config: Dict[str, Any] = {
            "preset": "full" if kind == "results" else "measurements",
            "collision": str(save_options.get("collision", "suffix")),
            "write_manifest": bool(save_options.get("write_manifest", True)),
            "layout": layout,
            "source_base_name": self._current_source_base_name(),
        }
        if kind == "results":
            export_config["exclude"] = self._artifact_exclude_from_save_options(save_options)

        if layout == "single":
            export_config.update(
                {
                    "naming": "frame_id",
                    "collision": "overwrite",
                    "base_name": base_name or self._current_source_base_name(),
                    "write_manifest": False,
                }
            )
            if kind == "results":
                export_config["image_format"] = "tiff" if image_format == "tiff" else "png"
        elif layout == "frames":
            export_config["naming"] = "frame_id_z"
            if kind == "results":
                export_config["image_format"] = "tiff" if image_format == "tiff" else "png"
        elif layout == "stack":
            if base_name:
                export_config["base_name"] = str(base_name)
        else:
            raise ValueError(f"Unsupported save layout: {layout}")

        return export_config

    def _prompt_save_destination(
        self,
        *,
        kind: str,
        layout: str,
        default_base_name: str,
        image_format: str,
    ) -> tuple[str, str | None, str | None] | None:
        """Provide an internal helper for prompt save destination.
        
        Args:
            kind (str): Value for kind interpreted by this routine.
            layout (str): Value for layout interpreted by this routine.
            default_base_name (str): Value for default base name interpreted by this routine.
            image_format (str): Array-like data carrying image format values for analysis.
        
        Returns:
            tuple[str, str | None, str | None] | None: Tuple grouping the prompt save destination outputs in a stable positional order."""
        is_csv = kind == "csv"
        if layout == "single":
            ext = ".csv" if is_csv else (".tiff" if image_format == "tiff" else ".png")
            title_single = "Export CSV (choose file name)" if is_csv else "Export results (choose base name)"
            file_filter = "CSV Files (*.csv);;All Files (*)" if is_csv else "All Files (*)"
            default_path = os.path.join(self.working_dir, f"{default_base_name}{ext}")
            selected, _ = QtWidgets.QFileDialog.getSaveFileName(
                self,
                title_single,
                default_path,
                file_filter,
            )
            if not selected:
                return None
            dest_dir = os.path.dirname(selected) or self.working_dir
            base_name = os.path.splitext(os.path.basename(selected))[0] or default_base_name
            if is_csv:
                return dest_dir, base_name, None
            selected_ext = os.path.splitext(selected)[1].strip().lower()
            if selected_ext in {".tif", ".tiff"}:
                image_format_override = "tiff"
            elif selected_ext == ".png":
                image_format_override = "png"
            else:
                image_format_override = None
            return dest_dir, base_name, image_format_override

        title_multi = "Export CSV to folder" if is_csv else "Export results to folder"
        dest_dir = QFileDialog.getExistingDirectory(
            self,
            title_multi,
            self.working_dir,
        )
        if not dest_dir:
            return None
        return dest_dir, None, None

    def _run_export(
        self,
        *,
        batch_result: BatchResult,
        dest_dir: str,
        export_config: Dict[str, Any],
    ) -> bool:
        """Provide an internal helper for run export.
        
        Args:
            batch_result (BatchResult): Value for batch result interpreted by this routine.
            dest_dir (str): Filesystem location describing where dest dir is read or written.
            export_config (Dict[str, Any]): Configuration values that control how export config is processed.
        
        Returns:
            bool: True when the validation or availability check succeeds; otherwise False."""
        try:
            export_result = batch_result.save_all(dest_dir, export_config)
        except Exception as exc:
            self._show_error("Error", f"Failed to export:\n{exc}")
            return False

        self.working_dir = dest_dir
        error_count = len(export_result.get("errors", []))
        if error_count:
            self._show_warning(
                "Exported with issues",
                f"Exported {export_result.get('written_count', 0)} file(s) to:\n{dest_dir}\n"
                f"Encountered {error_count} export error(s).",
            )
            return False

        self._show_info(
            "Export complete",
            f"Exported {export_result.get('written_count', 0)} file(s) to:\n{dest_dir}",
        )
        return True

    @staticmethod
    def _has_export_data(kind: str, batch_result: BatchResult | None) -> bool:
        """Provide an internal helper for has export data.
        
        Args:
            kind (str): Value for kind interpreted by this routine.
            batch_result (BatchResult | None): Value for batch result interpreted by this routine.
        
        Returns:
            bool: True when the validation or availability check succeeds; otherwise False."""
        if batch_result is None:
            return False
        if kind == "results":
            return any(fr.has_prediction or fr.has_segmentation for fr in batch_result.frame_results)
        if kind == "csv":
            return any(
                fr.has_measurements and fr.samples_table is not None and len(fr.samples_table) > 0
                for fr in batch_result.frame_results
            )
        return False

    def _refresh_save_actions(self) -> None:
        """Provide an internal helper for refresh save actions.
        
        Returns:
            None: Result object produced by the refresh save actions routine."""
        batch_result = self._effective_batch_result_for_save()
        has_results = self._has_export_data("results", batch_result)
        has_csv = self._has_export_data("csv", batch_result)
        self.action_save_masks_predictions.setEnabled(has_results)
        self.action_save_measurements.setEnabled(has_csv)

    def _save_by_kind(self, kind: str) -> None:
        """Provide an internal helper for save by kind.
        
        Args:
            kind (str): Value for kind interpreted by this routine.
        
        Returns:
            None: Result object produced by the save by kind routine."""
        batch_result = self._effective_batch_result_for_save()
        if not self._has_export_data(kind, batch_result):
            message = "Run segmentation first." if kind == "results" else "Run Analyze fibers first."
            self._show_info("Nothing to export", message)
            return

        save_options = self._normalize_save_options(getattr(self.config, "save_options", {}))
        if kind == "results" and not any(bool(save_options.get(key, False)) for key in self._save_artifact_keys()):
            self._show_warning(
                "Nothing selected",
                "All artifacts are disabled in File -> Save options.",
            )
            return

        layout = self._current_save_layout()
        default_base_name = (self.current_image_name or "fiberpose") + (
            "_results" if kind == "results" else "_samples"
        )
        target = self._prompt_save_destination(
            kind=kind,
            layout=layout,
            default_base_name=default_base_name,
            image_format=str(save_options.get("image_format", "png")),
        )
        if target is None:
            return
        dest_dir, base_name, image_format_override = target

        export_config = self._build_export_config(
            kind=kind,
            layout=layout,
            save_options=save_options,
            base_name=base_name,
            image_format_override=image_format_override,
        )
        self._run_export(
            batch_result=batch_result,
            dest_dir=dest_dir,
            export_config=export_config,
        )

    def save_predictions(self):
        """Persist predictions using the configured export conventions."""
        self._save_by_kind("results")

    def save_measurements(self):
        """Persist measurements using the configured export conventions."""
        self._save_by_kind("csv")


    # ------------------- Scalebar -------------------

    def _on_set_scalebar_clicked(self):
        """Handle the et scalebar clicked event and propagate UI state updates."""
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
        """Provide an internal helper for handle scalebar mouse.
        
        Args:
            event (QtGui.QMouseEvent): Value for event interpreted by this routine."""
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
        """Provide an internal helper for finalize scalebar selection."""
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
        self._invalidate_measurements()

        self._show_info(
            "Scalebar Set",
            f"Pixel size set to {pixel_size:.6g} {self._scalebar_unit} per pixel.",
        )
        self._cancel_scalebar_selection()

    def _cancel_scalebar_selection(self, reset_cursor: bool = True, clear_points: bool = True):
        """Provide an internal helper for cancel scalebar selection.
        
        Args:
            reset_cursor (bool): Value for reset cursor interpreted by this routine.
            clear_points (bool): Value for clear points interpreted by this routine."""
        self._scalebar_pending = False
        if clear_points:
            self._scalebar_points = []
        if reset_cursor:
            self.graphicsView.viewport().unsetCursor()

    # ------------------- Zoom helpers -------------------

    def reset_zoom(self):
        """Orchestrate reset zoom within the current module workflow."""
        self._fit_view_to_pixmap(force=True)

    def _update_reset_button_position(self):
        """Provide an internal helper for update reset button position."""
        viewport = self.graphicsView.viewport()
        if viewport is None:
            return
        margin = 8
        hint = self.btn_reset_view.sizeHint()
        x = viewport.width() - hint.width() - margin
        y = margin
        self.btn_reset_view.move(max(margin, x), y)

    def _update_reset_button_visibility(self):
        """Provide an internal helper for update reset button visibility."""
        self.btn_reset_view.setVisible(self._zoom_mode_active)

    def _handle_wheel_zoom(self, event: QtGui.QWheelEvent) -> bool:
        """Provide an internal helper for handle wheel zoom.
        
        Args:
            event (QtGui.QWheelEvent): Value for event interpreted by this routine.
        
        Returns:
            bool: True when the validation or availability check succeeds; otherwise False."""
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
        """Provide an internal helper for supported image extensions.
        
        Returns:
            set[str]: Result object produced by the supported image extensions routine."""
        return {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp", ".webp"}

    def _set_progress_bar(self, sample_progress_percent: float) -> None:
        """Provide an internal helper for set progress bar.
        
        Args:
            sample_progress_percent (float): Value for sample progress percent interpreted by this routine.
        
        Returns:
            None: Result object produced by the set progress bar routine."""
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
        """Provide an internal helper for run batch analysis.
        
        Args:
            compute_measurements (bool): Value for compute measurements interpreted by this routine.
            image_batch (ImageBatch | None): Array-like data carrying image batch values for analysis.
            error_title (str | None): Value for error title interpreted by this routine.
        
        Returns:
            BatchResult | None: Result object produced by the run batch analysis routine.
        
        Raises:
            RuntimeError: If runtime state prevents the operation from completing."""
        active_batch = image_batch if image_batch is not None else self._image_batch
        if active_batch is None or len(active_batch) == 0:
            self._show_info("Load original", "Please load an image or folder first.")
            return None

        # Build user-facing error text per processing mode unless explicitly provided.
        title = error_title
        if not title:
            title = "Batch analysis failed" if compute_measurements else "Batch segmentation failed"
        target_view = "Measurements overlay" if compute_measurements else "Overlay"
        if self.cb_view.currentText() != target_view:
            self.cb_view.setCurrentText(target_view)

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
            cached_batch = (
                self._batch_result
                if (
                    self._batch_result is not None
                    and active_batch is self._image_batch
                    and len(self._batch_result.frame_results) == len(active_batch)
                )
                else None
            )

            # Enter internal batch execution mode so frame callbacks can call segment/analyze safely.
            batch_analyzer = BatchAnalyzer()
            previous_batch_state = self._suppress_batch_dialogs
            previous_processing_state = self._is_batch_processing
            self._is_batch_processing = True
            if len(active_batch) > 1:
                self._suppress_batch_dialogs = True

            processed_images = 0

            def _process_frame(frame: np.ndarray, frame_id: str, z_index: int) -> Frame:
                """Provide an internal helper for process frame.
                
                Args:
                    frame (np.ndarray): Array-like data carrying frame values for analysis.
                    frame_id (str): Array-like data carrying frame id values for analysis.
                    z_index (int): Coordinate component representing the z index axis value.
                
                Returns:
                    Frame: Result object produced by the process frame routine."""
                nonlocal processed_images
                self.statusbar.showMessage(f"Processing {processed_images + 1}/{total_images} images")
                self._set_progress_bar(0.0)
                QtWidgets.QApplication.processEvents()
                cached_frame = None
                if cached_batch is not None and 0 <= int(z_index) < len(cached_batch.frame_results):
                    cached_frame = cached_batch.frame_at(int(z_index))
                if cached_frame is None:
                    self.current_frame = Frame.empty(
                        frame_id=str(frame_id),
                        z_index=int(z_index),
                        orig_img=frame,
                    )
                else:
                    seeded_frame = cached_frame.snapshot(
                        include_measurements=True,
                        frame_id=str(frame_id),
                        z_index=int(z_index),
                    )
                    seeded_frame.orig_img = frame
                    self.current_frame = seeded_frame

                self.current_image_name = self._frame_base_name(frame_id, z_index)
                ok = self._run_current_frame_pipeline(compute_measurements)
                if not ok and not self.current_frame.error_message:
                    self.current_frame.fail("Processing failed.")
                self.refresh_view()
                QtWidgets.QApplication.processEvents()
                result = self.current_frame.snapshot(
                    include_measurements=True,
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
        measurement_sig = self._current_batch_measurement_signature()
        measurements_valid = measurement_sig is not None and all(
            fr.has_measurements for fr in batch_result.frame_results
        )
        self._batch_results_have_measurements = measurements_valid
        self._last_batch_measurements_signature = measurement_sig if measurements_valid else None
        if not measurements_valid:
            self._stack_diameter_histogram_data = None
            self._stack_angle_histogram_data = None
            self._stack_diameter_histogram_enabled = False
        self._set_current_slice(self._current_z_index, update_view=True)
        self._refresh_save_actions()
        logger.info(
            "Batch %s completed successfully for %d frame(s).",
            "analysis" if compute_measurements else "segmentation",
            len(active_batch),
        )
        return batch_result

    def _on_batch_processing(self):
        """Handle the atch processing event and propagate UI state updates."""
        self._run_batch_processing_for_folder(None)

    def _run_batch_processing_for_folder(self, folder_path: str | None):
        """Provide an internal helper for run batch processing for folder.
        
        Args:
            folder_path (str | None): Filesystem location describing where folder path is read or written."""
        image_batch = self.open_source(path=folder_path, source_kind="folder")
        if image_batch is None:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        directory = self.working_dir
        output_root = os.path.join(directory, f"fiberpose_batch_{timestamp}")
        os.makedirs(output_root, exist_ok=True)

        batch_result = self._run_batch_analysis(
            compute_measurements=True,
            image_batch=image_batch,
            error_title="Batch processing failed",
        )
        if batch_result is None:
            return

        save_options = self._normalize_save_options(getattr(self.config, "save_options", {}))
        export_config = self._build_export_config(
            kind="results",
            layout="frames",
            save_options=save_options,
            base_name=None,
            image_format_override=None,
        )
        export_result = batch_result.save_all(output_root, export_config)

        failures: list[tuple[str, str]] = []
        processed = 0
        for result in batch_result.frame_results:
            if not result.has_measurements:
                failures.append((result.frame_id, "Measurement stage did not produce results"))
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
        """Handle the pen 3d view event and propagate UI state updates."""
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
        """Handle the nalyze 3d clicked event and propagate UI state updates."""
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
        self._refresh_save_actions()

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
        """Provide an internal helper for ensure profile dialog.
        
        Returns:
            PlotProfileDialog: Result object produced by the ensure profile dialog routine."""
        if self._profile_dialog is None:
            self._profile_dialog = PlotProfileDialog(self)
            self._profile_dialog.clearRequested.connect(self._on_profile_clear_requested)
            self._profile_dialog.closed.connect(self._on_profile_dialog_closed)
        return self._profile_dialog

    def _on_plot_profile_clicked(self):
        """Handle the lot profile clicked event and propagate UI state updates."""
        if not self._ensure_prediction_ready():
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
        """Handle the rofile clear requested event and propagate UI state updates."""
        self._reset_profile_selection(keep_dialog=True)

    def _on_profile_dialog_closed(self):
        """Handle the rofile dialog closed event and propagate UI state updates."""
        self._profile_mode_active = False
        self._profile_dialog = None
        self._reset_profile_selection()

    def _reset_profile_selection(self, keep_dialog: bool = False):
        """Provide an internal helper for reset profile selection.
        
        Args:
            keep_dialog (bool): Value for keep dialog interpreted by this routine."""
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
        """Provide an internal helper for remove profile line."""
        if self._profile_line_item is not None and self._profile_line_item.scene() is not None:
            self._scene.removeItem(self._profile_line_item)
        self._profile_line_item = None

    def _remove_profile_items(self):
        """Provide an internal helper for remove profile items."""
        self._remove_profile_line()
        if not self._profile_point_items:
            self._profile_point_items = []
            return
        for item in self._profile_point_items:
            if item is not None and item.scene() is not None:
                self._scene.removeItem(item)
        self._profile_point_items = []

    def _add_profile_marker(self, x: float, y: float):
        """Provide an internal helper for add profile marker.
        
        Args:
            x (float): Coordinate component representing the x axis value.
            y (float): Coordinate component representing the y axis value."""
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
        """Provide an internal helper for draw profile line."""
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
        """Provide an internal helper for scene to image coords.
        
        Args:
            scene_x (float): Value for scene x interpreted by this routine.
            scene_y (float): Value for scene y interpreted by this routine.
        
        Returns:
            tuple[float, float] | None: Tuple grouping the scene to image coords outputs in a stable positional order."""
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
        """Provide an internal helper for handle profile mouse.
        
        Args:
            event (QtGui.QMouseEvent): Value for event interpreted by this routine."""
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
        """Provide an internal helper for finalize profile selection."""
        if len(self._profile_points_scene) < 2 or len(self._profile_points_img) < 2:
            return
        self._draw_profile_line()
        self._update_profile_plot()

    def _update_profile_plot(self):
        """Provide an internal helper for update profile plot."""
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

    def _run_current_frame_pipeline(self, compute_measurements: bool) -> bool:
        if compute_measurements:
            return self._ensure_measurements_ready()
        return self._ensure_segmentation_ready()

    def segment(self):
        """Run the segmentation workflow for the current frame or active batch."""
        if self._image_batch is not None and not self._is_batch_processing:
            self._run_batch_analysis(compute_measurements=False)
            return
        self._set_progress_bar(0.0)
        if not self._run_current_frame_pipeline(compute_measurements=False):
            return

        if self.cb_view.currentText() == "Original":
            self.cb_view.setCurrentText("Overlay")

        self._refresh_save_actions()
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
        if self._image_batch is not None and not self._is_batch_processing:
            if self.cb_view.currentText() != "Measurements overlay":
                self.cb_view.setCurrentText("Measurements overlay")
            if self._batch_measurements_match_current_settings():
                self._set_progress_bar(100.0)
                self._set_current_slice(self._current_z_index, update_view=True)
                return
            self._run_batch_analysis(compute_measurements=True)
            return

        self._set_progress_bar(0.0)
        if not self._run_current_frame_pipeline(compute_measurements=True):
            self._set_progress_bar(0.0)
            return
        self._set_progress_bar(100.0)
        self._render_vector_overlay()

        fr = self.current_frame
        summary = fr.fiber_summary or {}
        units_label = str(summary.get("units", "px"))
        mean_thickness = float(summary.get("mean_diameter", float("nan")))
        std_thickness = float(summary.get("std_diameter", float("nan")))
        orientation_index = float(summary.get("orientation_index", float("nan")))
        mean_angle_deg = float(summary.get("mean_angle_deg", float("nan")))
        std_angle_deg = float(summary.get("std_angle_deg", float("nan")))
        spacing_px_input = max(1.0, float(getattr(self.config, "sample_spacing", 5.0)))
        self._show_info(
            "Analysis complete",
            f"Radius samples: {int(summary.get('sample_count', 0))} (spacing={int(spacing_px_input)} px)\n"
            f"Stored drawing samples: {len(fr.drawing_samples) if fr.drawing_samples is not None else 0}\n"
            f"Mean diameter: {mean_thickness:.2f} {units_label}\n"
            f"Std diameter: {std_thickness:.2f} {units_label}\n"
            f"Orientation index: {orientation_index:.2f}\n"
            f"Mean angle: {mean_angle_deg:.2f} deg\n"
            f"Std angle: {std_angle_deg:.2f} deg"
        )
        self._refresh_save_actions()
        logger.info("Fiber analysis completed successfully.")

    def _build_measurement_overlays(self):
        fr = self.current_frame
        sampling_base = ensure_rgb_u8(fr.dt_u8 if fr.dt_u8 is not None else fr.dt_normalized)
        if sampling_base is None and fr.overlay_img is not None:
            sampling_base = ensure_rgb_u8(fr.overlay_img)

        base_img = fr.orig_img if fr.scale_factor == 1.0 or fr.orig_img_scaled is None else fr.orig_img_scaled
        base_rgb = ensure_rgb_u8(base_img)

        fr.sampling_overlay = sampling_base.copy() if sampling_base is not None else None
        fr.diameters_overlay = (
            fr.overlay_img.copy()
            if fr.overlay_img is not None
            else (base_rgb.copy() if base_rgb is not None else None)
        )
        fr.orientations_overlay = base_rgb.copy() if base_rgb is not None else None
        fr.pores_overlay = self._pore_dt_to_rgb(fr.pore_dt, fr.void_mask)

        pts = list(fr.drawing_samples or [])
        if len(pts) > 15000:
            step = int(np.ceil(len(pts) / 15000))
            pts = pts[::step]

        if pts and fr.sampling_overlay is not None:
            height, width = fr.sampling_overlay.shape[:2]
            for idx, (y, x, ny, nx, r, _angle_deg) in enumerate(pts):
                if not (0 <= y < height and 0 <= x < width) or r <= 0:
                    continue

                p0 = (y - ny * r, x - nx * r)
                p1 = (y + ny * r, x + nx * r)
                self._draw_line(fr.sampling_overlay, p0, p1, color=(255, 255, 0))
                self._draw_circle(
                    fr.sampling_overlay,
                    (int(round(x)), int(round(y))),
                    radius_px=1,
                    color=(255, 0, 0),
                    thickness=-1,
                )
                if fr.diameters_overlay is not None and (idx % 10) == 0:
                    self._draw_line(fr.diameters_overlay, p0, p1, color=(255, 255, 0))

        self._fill_orientation_overlay_by_label()

        if fr.pores_overlay is not None and fr.pores:
            fr.pores_overlay = fr.pores_overlay.copy()
            for pore in fr.pores:
                x, y = int(pore["x"]), int(pore["y"])
                r = int(round(pore["r_px"]))
                if r > 0:
                    self._draw_circle(fr.pores_overlay, (x, y), r, color=(255, 255, 255), thickness=1)

    def _fill_orientation_overlay_by_label(self):
        fr = self.current_frame
        img = fr.orientations_overlay
        labels = fr.labels_2d
        samples = fr.drawing_samples or []
        if img is None or labels is None or not samples:
            return

        sample_map: dict[int, list[tuple[float, float, float]]] = {}
        height, width = labels.shape[:2]
        for y, x, _ny, _nx, _r, angle_deg in samples:
            iy = int(round(float(y)))
            ix = int(round(float(x)))
            if iy < 0 or iy >= height or ix < 0 or ix >= width:
                continue
            label_id = int(labels[iy, ix])
            if label_id <= 0 or not np.isfinite(angle_deg):
                continue
            sample_map.setdefault(label_id, []).append((float(y), float(x), float(angle_deg)))

        for label_id, label_samples in sample_map.items():
            mask = labels == label_id
            ys, xs = np.nonzero(mask)
            if ys.size == 0:
                continue

            sample_arr = np.asarray(label_samples, dtype=np.float32)
            tree = cKDTree(sample_arr[:, :2])
            query_coords = np.stack([ys, xs], axis=1).astype(np.float32)

            k = min(4, sample_arr.shape[0])
            distances, indices = tree.query(query_coords, k=k)

            angles = sample_arr[:, 2]
            pixel_angles = self._weighted_axial_average_angles(angles, distances, indices)
            pixel_colors = np.asarray(
                [self._angle_to_bgr(float(angle)) for angle in pixel_angles],
                dtype=np.uint8,
            )
            img[ys, xs] = pixel_colors


    @staticmethod
    def _weighted_axial_average_angles(
        sample_angles_deg: np.ndarray,
        distances: np.ndarray,
        indices: np.ndarray,
    ) -> np.ndarray:
        distances = np.asarray(distances, dtype=np.float32)
        indices = np.asarray(indices, dtype=np.int32)

        if distances.ndim == 1:
            distances = distances[:, None]
            indices = indices[:, None]

        neighbor_angles = sample_angles_deg[indices]
        weights = 1.0 / (distances + 1e-6)

        theta2 = np.deg2rad(2.0 * neighbor_angles)
        c = np.sum(weights * np.cos(theta2), axis=1)
        s = np.sum(weights * np.sin(theta2), axis=1)

        return 0.5 * np.rad2deg(np.arctan2(s, c))

    def _render_vector_overlay(self):
        fr = self.current_frame
        img = fr.diameters_overlay
        if self.rb_show_direct.isChecked():
            img = fr.orientations_overlay
        if self.rb_show_pores.isChecked():
            img = fr.pores_overlay

        if img is None:
            img = fr.overlay_img if fr.overlay_img is not None else self._labels_rgb()
        if img is None:
            return
        if self.cb_view.currentText() not in ("Measurements overlay", "Sampling"):
            self.cb_view.setCurrentText("Measurements overlay")
        self._set_pixmap_from_array(img)

    @staticmethod
    def _angle_to_bgr(angle_deg: float) -> tuple[int, int, int]:
        if not np.isfinite(angle_deg):
            return (200, 200, 200)
        ang = float(np.clip(angle_deg, -90.0, 90.0))
        hue = (ang + 90.0) / 180.0 * 179.0
        hsv = np.uint8([[[hue, 255, 255]]])
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
        return int(bgr[0]), int(bgr[1]), int(bgr[2])

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
        """Provide an internal helper for default grad quality points.
        
        Returns:
            tuple[float, float, float, float]: Tuple grouping the default grad quality points outputs in a stable positional order."""
        return (-0.35, -0.08, 0.08, 0.35)

    def _validated_grad_quality_points(self) -> tuple[float, float, float, float]:
        """Provide an internal helper for validated grad quality points.
        
        Returns:
            tuple[float, float, float, float]: Tuple grouping the validated grad quality points outputs in a stable positional order."""
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
        """Provide an internal helper for style plotwidget.
        
        Args:
            pw (pg.PlotWidget): Value for pw interpreted by this routine.
            title (str): Value for title interpreted by this routine.
            xlab (str): Value for xlab interpreted by this routine.
            ylab (str): Value for ylab interpreted by this routine."""
        pw.setBackground("w")
        pw.clear()
        pw.setTitle(title, color="k", size="12pt")
        pw.showGrid(x=True, y=True, alpha=0.25)
        ax = pw.getAxis("bottom"); ax.setPen("k"); ax.setTextPen("k"); ax.setLabel(xlab, color="k")
        ay = pw.getAxis("left");   ay.setPen("k"); ay.setTextPen("k"); ax.setLabel(ylab, color="k")
        pw.setAntialiasing(True)

    def _update_distributions_pyqtgraph(self, units_label: str):
        # Thickness
        """Provide an internal helper for update distributions pyqtgraph.
        
        Args:
            units_label (str): Value for units label interpreted by this routine."""
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

        show_pores = self.rb_show_pores.isChecked()

        if not show_pores:
            # Angles (-90..90)
            ang_title = "Fiber orientation"
            ang = (
                np.array([sample[5] for sample in (self.current_frame.measurement_samples or [])], dtype=np.float32)
                if self.current_frame.measurement_samples is not None
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
        """Orchestrate refresh view within the current module workflow."""
        img = None
        # Use the combo box text to select appropriate view.
        view = self.cb_view.currentText()
        if view == "Original":
            img = ensure_rgb_u8(self.current_frame.orig_img)
        elif view == "DT":
            img = to_uint8_for_view(self.current_frame.dt_normalized)
        elif view == "Laplacian":
            img = to_uint8_for_view(self.current_frame.lapl_img)
        elif view in ("Segmentation centerlines", "Centerlines"):
            img = self._segmentation_centerlines_with_connections()
        elif view == "Measurement centerlines":
            img = self._measurement_centerlines_view()
        elif view == "Centerline comparison":
            img = self._centerlines_comparison_view()
        elif view == "Segmentation":
            img = self._labels_rgb()
        elif view == "Segmentation outline":
            img = self._segmentation_outline_view()
        elif view == "Overlay":
            img = self.current_frame.overlay_img
        elif view == "Sampling":
            img = self.current_frame.sampling_overlay if self.current_frame.sampling_overlay is not None else ensure_rgb_u8(self.current_frame.dt_u8)
        elif view == "Measurements overlay":
            fallback = self.current_frame.overlay_img if self.current_frame.overlay_img is not None else self._labels_rgb()
            if self.rb_show_pores.isChecked():
                img = self.current_frame.pores_overlay
            elif self.rb_show_direct.isChecked():
                img = self.current_frame.orientations_overlay
            else:
                img = self.current_frame.diameters_overlay
            if img is None:
                img = fallback
        elif view == "gx": 
            img = to_uint8_for_view(self.current_frame.gx)
        elif view == "gy":
            img = to_uint8_for_view(self.current_frame.gy)
        elif view == "DT grad error":
            img = self._signed_grad_error_rgb()

        self._set_pixmap_from_array(img)

    def _configure_z_scroll(self, frame_count: int):
        """Provide an internal helper for configure z scroll.
        
        Args:
            frame_count (int): Array-like data carrying frame count values for analysis."""
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
        """Provide an internal helper for update slice label."""
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
        """Provide an internal helper for frame base name.
        
        Args:
            frame_id (str): Array-like data carrying frame id values for analysis.
            z_index (int): Coordinate component representing the z index axis value.
        
        Returns:
            str: String value representing the normalized frame base name outcome."""
        base = os.path.splitext(str(frame_id))[0]
        if self._image_batch is None:
            return base
        source_type = self._image_batch.metadata.get("source_type")
        if source_type == "stack" and len(self._image_batch) > 1:
            return f"{base}_z{z_index:03d}"
        return base

    def _apply_frame_result(self, frame_result: Frame | None):
        """Provide an internal helper for apply frame result.
        
        Args:
            frame_result (Frame | None): Array-like data carrying frame result values for analysis."""
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
            self.current_frame.segmentation_outline_cache = (None, None)

    def _set_current_slice(self, index: int, update_view: bool = True):
        """Provide an internal helper for set current slice.
        
        Args:
            index (int): Zero-based index selecting which index to operate on.
            update_view (bool): Value for update view interpreted by this routine."""
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
        """Handle the z scroll changed event and propagate UI state updates.
        
        Args:
            value (int): Value for value interpreted by this routine."""
        if self._image_batch is None:
            return
        self._set_current_slice(int(value), update_view=True)

    def _reset_processing_state(self, preserve_original: bool = False):
        """Reset analyzer + frame processing state, without touching UI widgets/charts."""
        if self.analyzer is not None:
            self.analyzer._reset_prediction_state()

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

    def _reset_internal_state(self, preserve_original: bool = False, reset_view: bool = True):
        """Clear all derived data, images, analysis artifacts, and charts."""
        self._reset_processing_state(preserve_original=preserve_original)
        if self._profile_dialog is not None:
            self._profile_dialog.close()

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

        self._refresh_save_actions()

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
        """Handle the usegpu toggled event and propagate UI state updates.
        
        Args:
            checked (bool): Value for checked interpreted by this routine."""
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

    def _on_invert_toggled(self, checked: bool):
        self._invalidate_prediction()

    def _on_segmentation_mode_changed(self, checked: bool | None = None):
        """Handle the segmentation mode changed event and propagate UI state updates.
        
        Args:
            checked (bool | None): Value for checked interpreted by this routine."""
        if not checked: return # Ignore signals from the button being unchecked
        
        # Sync the radio button state to config
        self.sync_ui_to_config() 
        
        # Update UI states that depend on the mode
        self._update_strict_checkbox_state()
        
        if self.analyzer is not None:
            self.analyzer.set_strict_centerline_filter(self.config.strict_centerlines)

    def _on_strict_checkbox_toggled(self, checked: bool):
        """Handle the strict checkbox toggled event and propagate UI state updates.
        
        Args:
            checked (bool): Value for checked interpreted by this routine."""
        self._set_strict_mode(bool(checked))

    def _on_cluster_toggled(self, checked: bool):
        # Update the config flag immediately
        """Handle the cluster toggled event and propagate UI state updates.
        
        Args:
            checked (bool): Value for checked interpreted by this routine."""
        self.config.cluster_enabled = bool(checked)
        if self.analyzer is not None:
            # Safely pass cluster_params (now defined in AppConfig)
            self.analyzer.enable_fiber_clustering(
                self.config.cluster_enabled, 
                **self.config.cluster_params
            )

    # ------------------- New default image loading -------------------

    def load_default_image(self):
        """Load default image from persisted storage and normalize it for use."""
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
        """Strict filtering now controls measurement-centerline filtering and is always available."""
        self.chb_strict.setEnabled(True)

    def _set_segmentation_mode(self, mode: str) -> None:
        """Provide an internal helper for set segmentation mode.
        
        Args:
            mode (str): Value for mode interpreted by this routine.
        
        Returns:
            None: Result object produced by the set segmentation mode routine."""
        target = str(mode or "watershed").strip().lower()
        if target == "radial_expansion":
            self.rb_ridge.setChecked(True)
            return
        self.rb_watershed.setChecked(True)

    def sync_ui_to_sample_properties(self):
        """Reads sample-specific calibration fields from the UI."""
        if not hasattr(self, "sample_props") or self.sample_props is None:
            self.sample_props = SampleProperties.defaults()

        prev_px = float(getattr(self.sample_props, "pixel_size_x", 1.0))
        prev_py = float(getattr(self.sample_props, "pixel_size_y", 1.0))
        prev_z = float(getattr(self.sample_props, "voxel_size_z", 1.0))
        prev_unit = str(getattr(self.sample_props, "unit", "px"))

        def _parse_positive(text: str, fallback: float) -> float:
            """Provide an internal helper for parse positive.
            
            Args:
                text (str): Value for text interpreted by this routine.
                fallback (float): Value for fallback interpreted by this routine.
            
            Returns:
                float: Result object produced by the parse positive routine."""
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

        changed = (
            not math.isclose(prev_px, float(self.sample_props.pixel_size_x), rel_tol=1e-9, abs_tol=1e-9)
            or not math.isclose(prev_py, float(self.sample_props.pixel_size_y), rel_tol=1e-9, abs_tol=1e-9)
            or not math.isclose(prev_z, float(self.sample_props.voxel_size_z), rel_tol=1e-9, abs_tol=1e-9)
            or prev_unit != str(self.sample_props.unit)
        )
        if changed:
            self._invalidate_measurements()

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

        prev_checkpoint = str(getattr(self.config, "checkpoint_path", ""))
        prev_scale_label = str(getattr(self.config, "scale_label", "1x"))
        prev_max_angle = float(getattr(self.config, "max_angle_deg", 30.0))
        prev_strict = bool(getattr(self.config, "strict_centerlines", False))
        prev_mode = str(getattr(self.config, "segmentation_mode", "watershed"))
        prev_merge_expansion_factor = float(getattr(self.config, "merge_expansion_factor", 0.7))
        prev_segmentation_expansion_factor = float(getattr(self.config, "segmentation_expansion_factor", 1.0))
        prev_threshold = int(getattr(self.config, "seg_threshold", 1))
        prev_spacing = float(getattr(self.config, "sample_spacing", 5.0))
        prev_cluster = bool(getattr(self.config, "cluster_enabled", False))
        prev_cluster_params = json.dumps(
            dict(getattr(self.config, "cluster_params", {}) or {}),
            sort_keys=True,
            default=str,
        )

        # Model & Hardware
        self.config.checkpoint_path = self.cb_model.currentText().strip()
        self.config.use_gpu = self.chb_usegpu.isChecked()
        self.config.scale_label = self.cb_scale.currentText().strip()

        # Centerline Params
        if hasattr(self, "sp_max_cent_angle"):
            self.config.max_angle_deg = float(self.sp_max_cent_angle.value())
        else:
            self.config.max_angle_deg = float(getattr(self.config, "max_angle_deg", 30.0))
        self.config.strict_centerlines = self.chb_strict.isChecked()

        # Segmentation Mode
        if self.rb_ridge.isChecked():
            self.config.segmentation_mode = "radial_expansion"
        else:
            self.config.segmentation_mode = "watershed"

        self.config.seg_threshold = int(self.sp_segm_threshold.value())
        self.config.merge_expansion_factor = float(getattr(self.config, "merge_expansion_factor", 0.7))
        self.config.segmentation_expansion_factor = float(getattr(self.config, "segmentation_expansion_factor", 1.0))

        # Analysis
        if hasattr(self, "le_samspacing"):
            text = self.le_samspacing.text().strip()
            try:
                spacing = float(text)
            except Exception:
                spacing = prev_spacing
            if spacing > 0 and np.isfinite(spacing):
                self.config.sample_spacing = float(spacing)
            else:
                self.config.sample_spacing = float(prev_spacing)
        else:
            self.config.sample_spacing = float(getattr(self.config, "sample_spacing", 5.0))

        # Clustering
        self.config.cluster_enabled = self.chb_cluster.isChecked()

        new_cluster_params = json.dumps(
            dict(getattr(self.config, "cluster_params", {}) or {}),
            sort_keys=True,
            default=str,
        )

        prediction_changed = (
            prev_checkpoint != str(self.config.checkpoint_path)
            or prev_scale_label != str(self.config.scale_label)
        )
        max_angle_changed = not math.isclose(
            prev_max_angle,
            float(self.config.max_angle_deg),
            rel_tol=1e-9,
            abs_tol=1e-9,
        )
        segmentation_changed = (
            max_angle_changed
            or prev_strict != bool(self.config.strict_centerlines)
            or prev_mode != str(self.config.segmentation_mode)
            or not math.isclose(
                prev_merge_expansion_factor,
                float(self.config.merge_expansion_factor),
                rel_tol=1e-9,
                abs_tol=1e-9,
            )
            or not math.isclose(
                prev_segmentation_expansion_factor,
                float(self.config.segmentation_expansion_factor),
                rel_tol=1e-9,
                abs_tol=1e-9,
            )
            or prev_threshold != int(self.config.seg_threshold)
            or prev_cluster != bool(self.config.cluster_enabled)
            or prev_cluster_params != new_cluster_params
        )
        measurements_changed = not math.isclose(
            prev_spacing,
            float(self.config.sample_spacing),
            rel_tol=1e-9,
            abs_tol=1e-9,
        )

        if prediction_changed:
            self._invalidate_prediction()
        elif segmentation_changed:
            self._invalidate_segmentation(
                invalidate_centerlines=(
                    max_angle_changed
                    or prev_strict != bool(self.config.strict_centerlines)
                    or not math.isclose(
                        prev_merge_expansion_factor,
                        float(self.config.merge_expansion_factor),
                        rel_tol=1e-9,
                        abs_tol=1e-9,
                    )
                    or prev_cluster != bool(self.config.cluster_enabled)
                    or prev_cluster_params != new_cluster_params
                )
            )
        elif measurements_changed:
            self._invalidate_measurements()

    def sync_config_to_ui(self):
        """Updates the UI widgets to reflect the values in self.config."""
        if not hasattr(self, "config"):
            return

        cfg = self.config

        def set_combo_text(cb, text):
            """Update combo text and synchronize dependent runtime state.
            
            Args:
                cb (Any): Value for cb interpreted by this routine.
                text (Any): Value for text interpreted by this routine."""
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

        if str(cfg.segmentation_mode).strip().lower() == "radial_expansion":
            self.rb_ridge.setChecked(True)
        else:
            self.rb_watershed.setChecked(True)

        self.rb_watershed.blockSignals(False)
        self.rb_ridge.blockSignals(False)

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
