"""Main Qt window coordinating image IO, analysis workflows, and visualization state."""

import os
from datetime import datetime
from typing import Any, Dict
import numpy as np

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

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

from .support import (
    PlotProfileDialog,
    ConfigEditorDialog,
)
from ..ml import onnx_cuda_available
from ..domain import BatchResult, Frame, ImageBatch, AppConfig, SampleProperties
from ..render import render
from ..utils import to_uint8_for_view
from ..pipeline import PipelineController, PipelineOutcome
from .support import UiConfigBinder
from ..pipeline import BatchService, BatchCallbacks
from ..io_export import ExportService
from .support import ScalebarTool, ProfileTool
from ..paths import DEFAULT_UI_PATH, EXAMPLE_IMAGE_PATH, default_config_path
from ..models import DEFAULT_MODEL, discover_models, models_dir, normalize_model_entry, resolve_model_path

# ------------------- MainWindow -------------------


class MainWindow(QMainWindow):
    """Main Fiberpose window: loads sources, runs the pipeline, and shows views, plots, and exports."""
    def __init__(self, ui_path: str = str(DEFAULT_UI_PATH), config_path: str | None = None):
        super().__init__()
        loadUi(ui_path, self)
        self.setWindowTitle("Fiberpose — fiber segmentation and analysis tool")
        self.config_filepath = os.path.abspath(config_path or default_config_path())

        # Load config and sync to UI
        self.config = AppConfig.load_from_json(self.config_filepath)
        self.config.model_path = normalize_model_entry(self.config.model_path)
        self.sample_props = SampleProperties.defaults()
        self._ui_binder = UiConfigBinder()
        self._scalebar_tool = ScalebarTool()
        self._profile_tool = ProfileTool()
        self._pipeline_service = PipelineController()
        self._batch_service = BatchService()
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
        self.current_image_name: str | None = None
        self._pixmap_source_shape: tuple[int, int] | None = None

        # View/zoom state
        self._zoom_mode_active: bool = False
        self._base_scale: float = 1.0
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
        self._image_batch: ImageBatch | None = None
        self._batch_result: BatchResult | None = None
        self._current_z_index: int = 0
        self.z_scroll.setEnabled(False)
        self.z_scroll.valueChanged.connect(self._on_z_scroll_changed)
        self._update_slice_label()

        # Profile plotting tool state
        self._profile_line_item: QGraphicsItem | None = None
        self._profile_point_items: list[QGraphicsItem] = []
        self._profile_dialog: PlotProfileDialog | None = None

        # Menu & actions
        self.action_load_image.triggered.connect(self.load_original)
        self.action_save_masks_predictions.triggered.connect(self.save_predictions)
        self.action_save_measurements.triggered.connect(self.save_measurements)
        self.action_edit_config.triggered.connect(self._on_edit_config)
        self.action_load_config.triggered.connect(self._on_load_config)

        self.btn_load_image.clicked.connect(self.load_original)
        self.btn_segment.clicked.connect(self.segment)
        self.btn_measure.clicked.connect(self.analyze_fibers)
        self.btn_set_scalebar.clicked.connect(self._on_set_scalebar_clicked)
        self.btn_plot_profile.clicked.connect(self._on_plot_profile_clicked)
        self.btn_export_results.clicked.connect(self._on_export_results_clicked)
        self.btn_save_screenshot.clicked.connect(self._on_save_screenshot_clicked)
        self.btn_export_csv.clicked.connect(self.save_measurements)
        self.action_batch_processing.triggered.connect(self._on_batch_processing)

        # Collapsible "Advanced" sections start folded.
        for button, panel in (
            (self.btn_segmentation_advanced, self.widget_segmentation_advanced),
            (self.btn_measurement_advanced, self.widget_measurement_advanced),
            (self.btn_export_advanced, self.widget_export_advanced),
        ):
            panel.setVisible(False)
            button.toggled.connect(
                lambda checked, button=button, panel=panel: self._toggle_advanced(button, panel, checked)
            )

        self.cb_view.currentIndexChanged.connect(self.refresh_view)
        self.cb_view.setMinimumHeight(self.btn_load_image.sizeHint().height())
        # Units belong to the plots: hidden with them, but keep the row's space so plots don't shift.
        for widget in (self.label_units, self.cb_units):
            policy = widget.sizePolicy()
            policy.setRetainSizeWhenHidden(True)
            widget.setSizePolicy(policy)
        self._clear_results()

        # Set default model
        self._init_model_selector()

        # Disable saves initially.
        self._refresh_save_actions()

        self.resize(1100, 800)
        self.refresh_view()
        
        # Delay loading default image until the UI is fully shown.
        QTimer.singleShot(0, self.load_default_image)

        # Use CUDA by default when available; the model loads on the first prediction.
        self.chb_usegpu.setChecked(onnx_cuda_available())
        self._pipeline_service.set_model(
            self._current_model_path() or str(models_dir() / DEFAULT_MODEL),
            "cuda" if self.chb_usegpu.isChecked() else "cpu",
        )

        # react to checkbox changes at runtime
        self.chb_usegpu.toggled.connect(self._on_usegpu_toggled)
        self.chb_invert.toggled.connect(self._on_invert_toggled)

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
        self.cb_units.currentTextChanged.connect(self._on_units_changed)

        # Checkboxes and Radio Buttons
        self.chb_usegpu.toggled.connect(self.sync_ui_to_config)
        self.chb_strict.toggled.connect(self.sync_ui_to_config)
        
        self.rb_watershed.toggled.connect(self.sync_ui_to_config)
        self.rb_ridge.toggled.connect(self.sync_ui_to_config)

        # Spinboxes
        self.sp_mask_threshold.valueChanged.connect(self.sync_ui_to_config)
        self.sp_foreground_threshold.valueChanged.connect(self.sync_ui_to_config)

        # Line Edits (Text boxes)
        # Note: We use `editingFinished` instead of `textChanged`.
        self.le_pixelsize.editingFinished.connect(self.sync_ui_to_sample_properties)
        self.le_voxeldepth.editingFinished.connect(self.sync_ui_to_sample_properties)

        # Export options live in config.save_options, not in the binder.
        for checkbox in (*self._export_checkboxes().values(), self.chb_write_manifest):
            checkbox.toggled.connect(self._sync_ui_to_save_options)
        self.cb_image_format.currentIndexChanged.connect(self._sync_ui_to_save_options)
        self.cb_collision.currentIndexChanged.connect(self._sync_ui_to_save_options)

    def _init_model_selector(self) -> None:
        models = self._discover_model_files()
        configured_entry = normalize_model_entry(self.config.model_path)
        rel_default = DEFAULT_MODEL
        self.cb_model.blockSignals(True)
        self.cb_model.clear()
        if models:
            for rel_path in models:
                self.cb_model.addItem(rel_path)
        else:
            self.cb_model.addItem(rel_default)

        # Keep explicit config choice first; fallback to default, then first available model.
        if configured_entry:
            if self.cb_model.findText(configured_entry) < 0:
                self.cb_model.addItem(configured_entry)
            configured_index = self.cb_model.findText(configured_entry)
            if configured_index >= 0:
                self.cb_model.setCurrentIndex(configured_index)
            else:
                self.cb_model.setCurrentIndex(0)
        else:
            default_index = self.cb_model.findText(rel_default)
            if default_index >= 0:
                self.cb_model.setCurrentIndex(default_index)
            else:
                self.cb_model.setCurrentIndex(0)
        self.cb_model.blockSignals(False)
        self.cb_model.currentIndexChanged.connect(self._on_model_selection_changed)

    def _on_model_selection_changed(self, index: int):
        """Handle model selection changed and propagate UI state updates.
        
        Args:
            index (int): Zero-based position used to select an item."""
        new_path = self._current_model_path()
        if not new_path:
            return
        if not os.path.isfile(new_path):
            QMessageBox.warning(
                self,
                "Model missing",
                f"Selected model was not found:\n{new_path}",
            )
            return
        if new_path == self._pipeline_service.model_path:
            return
        self._pipeline_service.set_model(new_path, self._pipeline_service.device)
        self._invalidate_prediction()

    def _discover_model_files(self) -> list[str]:
        return discover_models()

    def _current_model_path(self) -> str | None:
        entry = self.cb_model.currentText().strip()
        if not entry:
            return None
        try:
            return resolve_model_path(entry)
        except Exception as exc:
            logger.warning("Could not download model %s: %s", entry, exc)
            return str(models_dir() / normalize_model_entry(entry))


    # ------------------- Display helpers -------------------

    def _set_pixmap_from_array(self, img: np.ndarray | None):
        prev_shape = self._pixmap_source_shape
        if img is None:
            self._pixitem.setPixmap(QtGui.QPixmap())
            self._pixmap_source_shape = None
            self._zoom_mode_active = False
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
            self._zoom_mode_active = False
            self._update_reset_button_visibility()

    def _invalidate_prediction(self) -> None:
        self.current_frame.clear_prediction()
        self._invalidate_measurements()

    def _invalidate_segmentation(self) -> None:
        self.current_frame.clear_segmentation()
        self._invalidate_measurements()

    def _invalidate_measurements(self) -> None:
        self.current_frame.clear_measurements()
        self._clear_results()
        self._refresh_save_actions()

    def _ensure_prediction_ready(self) -> bool:
        self.sync_ui_to_config()
        outcome = self._pipeline_service.ensure_prediction(
            self.current_frame,
            self.config,
            invert_input=bool(self.chb_invert.isChecked()),
        )
        return self._handle_pipeline_outcome(outcome)

    def _ensure_segmentation_ready(self) -> bool:
        self.sync_ui_to_config()
        outcome = self._pipeline_service.ensure_segmentation(
            self.current_frame,
            self.config,
            invert_input=bool(self.chb_invert.isChecked()),
        )
        return self._handle_pipeline_outcome(outcome)

    def _ensure_measurements_ready(self) -> bool:
        self.sync_ui_to_config()
        self.sync_ui_to_sample_properties()
        outcome = self._pipeline_service.ensure_measurements(
            self.current_frame,
            self.config,
            self.sample_props,
            invert_input=bool(self.chb_invert.isChecked()),
        )
        if not self._handle_pipeline_outcome(outcome):
            return False
        summary = self.current_frame.fiber_summary or {}
        units_label = str(summary.get("units", "px"))
        info_text = (
            f"Diam Mean: {summary.get('diam_mean', float('nan')):.2f} {units_label}\n"
            f"Diam SD: {summary.get('diam_sd', float('nan')):.2f} {units_label}\n"
            f"Diam Mode: {summary.get('diam_mode', float('nan')):.2f} {units_label}\n"
            f"Diam Median: {summary.get('diam_median', float('nan')):.2f} {units_label}\n"
            f"Orientation Index: {summary.get('orientation_index', float('nan')):.2f}\n"
            f"Angle Mean: {summary.get('mean_angle_deg', float('nan')):.2f} deg\n"
            f"Angle SD: {summary.get('std_angle_deg', float('nan')):.2f} deg\n"
            f"Porosity: {summary.get('porosity', float('nan'))*100:.1f} %\n"
            f"Centerline Fragments: {int(summary.get('fragments_with_centerlines', 0))}"
        )
        self.text_edit2.setPlainText(info_text)
        self._update_distributions_pyqtgraph(units_label)
        return True

    def _show_info(self, title: str, message: str):
        if self._suppress_batch_dialogs:
            logger.info("[Batch] %s: %s", title, message)
        else:
            QtWidgets.QMessageBox.information(self, title, message)

    def _handle_pipeline_outcome(self, outcome: PipelineOutcome) -> bool:
        if outcome.ok:
            return True
        if outcome.error_title and outcome.error_message:
            self._show_error(outcome.error_title, outcome.error_message)
        elif outcome.info_title and outcome.info_message:
            self._show_info(outcome.info_title, outcome.info_message)
        return False

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
        model_entry = normalize_model_entry(self.config.model_path)
        self.config.model_path = model_entry
        if model_entry and self.cb_model.findText(model_entry) < 0:
            self.cb_model.addItem(model_entry)

        self.sync_config_to_ui()
        self.sync_ui_to_config()

        self._on_model_selection_changed(self.cb_model.currentIndex())
        self._on_usegpu_toggled(self.chb_usegpu.isChecked())

    def _load_config_from_path(self, path: str) -> bool:
        selected_path = os.path.abspath(path)
        try:
            self.config = AppConfig.load_from_json(selected_path)
        except Exception as exc:
            self._show_error("Load config failed", str(exc))
            return False

        self.config_filepath = selected_path
        self._apply_runtime_after_config_change()
        return True

    def _on_load_config(self):
        start_dir = os.path.dirname(self.config_filepath) if self.config_filepath else os.path.expanduser("~")
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

    def _export_checkboxes(self) -> Dict[str, QtWidgets.QCheckBox]:
        """Export-panel checkboxes keyed by their save_options artifact name."""
        return {key: getattr(self, f"chb_export_{key}") for key in self._save_artifact_keys()}

    def _sync_save_options_to_ui(self) -> None:
        """Show config.save_options in the export panel without echoing back into the config."""
        options = self._normalize_save_options(self.config.save_options)
        widgets = (
            *self._export_checkboxes().values(),
            self.chb_write_manifest,
            self.cb_image_format,
            self.cb_collision,
        )
        for widget in widgets:
            widget.blockSignals(True)
        try:
            for key, checkbox in self._export_checkboxes().items():
                checkbox.setChecked(bool(options[key]))
            self.chb_write_manifest.setChecked(bool(options["write_manifest"]))
            self.cb_image_format.setCurrentIndex(max(0, self.cb_image_format.findText(options["image_format"])))
            self.cb_collision.setCurrentIndex(max(0, self.cb_collision.findText(options["collision"])))
        finally:
            for widget in widgets:
                widget.blockSignals(False)

    def _sync_ui_to_save_options(self, *_args) -> None:
        """Store the export panel state in config.save_options."""
        options: Dict[str, Any] = {
            key: checkbox.isChecked() for key, checkbox in self._export_checkboxes().items()
        }
        options["image_format"] = self.cb_image_format.currentText()
        options["collision"] = self.cb_collision.currentText()
        options["write_manifest"] = self.chb_write_manifest.isChecked()
        self.config.save_options = self._normalize_save_options(options)


    # ------------------- Drag & drop -------------------

    def _install_drag_drop(self):
        v = self.graphicsView
        v.setAcceptDrops(True)
        v.viewport().setAcceptDrops(True)
        v.viewport().installEventFilter(self)
        # Cmd/Ctrl+C copies the image only while the view itself has focus.
        copy_shortcut = QtGui.QShortcut(QtGui.QKeySequence(QtGui.QKeySequence.StandardKey.Copy), v)
        copy_shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        copy_shortcut.activated.connect(self._copy_active_image)

    def eventFilter(self, obj, event):
        if obj is self.graphicsView.viewport():
            et = event.type()
            if et == QEvent.Type.Resize:
                self._update_reset_button_position()
            elif et == QEvent.Type.Wheel:
                if isinstance(event, QtGui.QWheelEvent):
                    if self._handle_wheel_zoom(event):
                        return True
            if self._scalebar_tool.pending and et == QEvent.Type.MouseButtonPress:
                if isinstance(event, QtGui.QMouseEvent):
                    self._handle_scalebar_mouse(event)
                    return True
            if self._profile_tool.active and et == QEvent.Type.MouseButtonPress:
                if isinstance(event, QtGui.QMouseEvent):
                    self._handle_profile_mouse(event)
                    return True
            if et == QEvent.Type.ContextMenu:
                if isinstance(event, QtGui.QContextMenuEvent):
                    self._show_view_context_menu(event.globalPos())
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
                                    # Keep drag-and-drop behavior aligned with the Load action:
                                    # dropping a folder loads it, while processing is triggered
                                    # explicitly via Segment/Analyze actions.
                                    self.open_source(path=p, source_kind="folder")
                                else:
                                    self.open_source(path=p, source_kind="auto")
                                event.acceptProposedAction()
                                return True
        return super().eventFilter(obj, event)

    def _show_view_context_menu(self, global_pos: QPoint) -> None:
        """Show right-click context actions for the current graphics viewport."""
        if self._pixitem.pixmap().isNull():
            return
        menu = QtWidgets.QMenu(self)
        action_copy = menu.addAction("Copy Image")
        action_copy.setShortcut(QtGui.QKeySequence(QtGui.QKeySequence.StandardKey.Copy))
        action_save = menu.addAction("Save Image As\u2026")
        selected = menu.exec(global_pos)
        if selected is action_copy:
            self._copy_active_image()
        elif selected is action_save:
            self._save_active_image_png()

    def _copy_active_image(self) -> None:
        """Copy the full-resolution image of the selected view mode to the clipboard."""
        if self.current_frame.orig_img is None:
            return
        img = self.refresh_view()
        if img is None:
            self._show_info("No image", "No active image is available to copy for this view mode.")
            return
        # Detach from the numpy buffer: the clipboard outlives this array.
        QtWidgets.QApplication.clipboard().setImage(self._numpy_to_qimage(img).copy())
        self.statusbar.showMessage("Image copied to clipboard", 5000)

    def _on_export_results_clicked(self) -> None:
        self.save_predictions()

    def _on_save_screenshot_clicked(self) -> None:
        if self.current_frame.orig_img is not None:
            self._save_active_image_png()

    def _save_active_image_png(self) -> None:
        """Save the full active image for the selected view mode as PNG."""
        img = self.refresh_view()
        if img is None:
            self._show_info("No image", "No active image is available to save for this view mode.")
            return

        view_label = str(self.cb_view.currentText() or "view").strip().lower()
        view_slug = "".join(ch if ch.isalnum() else "_" for ch in view_label).strip("_") or "view"
        base_name = (self.current_image_name or "fiberpose").strip() or "fiberpose"
        default_name = f"{base_name}_{view_slug}.png"
        default_path = os.path.join(self.working_dir, default_name)

        selected_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save Image As",
            default_path,
            "PNG Files (*.png);;All Files (*)",
        )
        if not selected_path:
            return
        if not selected_path.lower().endswith(".png"):
            selected_path = f"{selected_path}.png"

        qimg = self._numpy_to_qimage(img)
        pixmap = QtGui.QPixmap.fromImage(qimg)
        ok = pixmap.save(selected_path, "PNG")
        if not ok:
            self._show_error("Save failed", "Could not save the active image as PNG.")
            return

        self.working_dir = os.path.dirname(selected_path) or self.working_dir
        self.statusbar.showMessage(f"Saved active image: {selected_path}", 5000)

    def _is_image_path(self, path: str) -> bool:
        ext = os.path.splitext(path)[1].lower()
        return ext in self._supported_image_extensions()

    def _is_loadable_path(self, path: str) -> bool:
        return os.path.isdir(path) or self._is_image_path(path)

    def _urls_are_loadable(self, urls) -> bool:
        return any(u.isLocalFile() and self._is_loadable_path(u.toLocalFile()) for u in urls)

    def _batch_measurements_match_current_settings(self) -> bool:
        if self._batch_result is None or self._image_batch is None:
            return False
        if len(self._batch_result.frame_results) != len(self._image_batch):
            return False
        invert_input = bool(self.chb_invert.isChecked())
        return all(
            frame.has_measurements
            and frame.fiber_summary is not None
            and frame.measurement_sig
            == self._pipeline_service.measurement_signature(
                frame,
                self.config,
                self.sample_props,
                invert_input=invert_input,
            )
            for frame in self._batch_result.frame_results
        )

    def _set_image_batch(self, image_batch: ImageBatch):
        self._image_batch = image_batch
        self._batch_result = None
        self._current_z_index = 0
        self._configure_z_scroll(len(image_batch))
        self._set_current_slice(0, update_view=True)
        # New source invalidates previous export availability until
        # segmentation/analysis is run on the newly loaded data.
        self._refresh_save_actions()

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

    @classmethod
    def _normalize_save_options(cls, raw_options: Any) -> Dict[str, Any]:
        """Normalize save options into a canonical representation.
        
        Args:
            raw_options (Any): Configuration values controlling runtime behavior.
        
        Returns:
            Dict[str, Any]: Mapping of computed keys and values."""
        return ExportService.normalize_save_options(raw_options)

    @staticmethod
    def _save_artifact_keys() -> tuple[str, ...]:
        return ExportService.save_artifact_keys()

    def _current_save_layout(self) -> str:
        """Return export layout based on loaded source type."""
        return ExportService.current_save_layout(self._image_batch)

    def _current_source_base_name(self) -> str:
        return ExportService.current_source_base_name(self._image_batch, self.current_image_name)

    def _effective_batch_result_for_save(self) -> BatchResult | None:
        return ExportService.effective_batch_result_for_save(
            layout=self._current_save_layout(),
            image_batch=self._image_batch,
            batch_result=self._batch_result,
            current_frame=self.current_frame,
        )

    def _build_export_config(
        self,
        *,
        kind: str,
        layout: str,
        save_options: Dict[str, Any],
        base_name: str | None = None,
        image_format_override: str | None = None,
    ) -> Dict[str, Any]:
        return ExportService.build_export_config(
            kind=kind,
            layout=layout,
            save_options=save_options,
            source_base_name=self._current_source_base_name(),
            base_name=base_name,
            image_format_override=image_format_override,
        )

    def _prompt_save_destination(
        self,
        *,
        kind: str,
        layout: str,
        default_base_name: str,
        image_format: str,
    ) -> tuple[str, str | None, str | None] | None:
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
        return ExportService.has_export_data(kind, batch_result)

    def _refresh_save_actions(self) -> None:
        batch_result = self._effective_batch_result_for_save()
        has_results = self._has_export_data("results", batch_result)
        has_csv = self._has_export_data("csv", batch_result)
        self.action_save_masks_predictions.setEnabled(has_results)
        self.action_save_measurements.setEnabled(has_csv)
        self.btn_export_results.setEnabled(has_results)
        self.btn_export_csv.setEnabled(has_csv)
        self.btn_save_screenshot.setEnabled(self.current_frame.orig_img is not None)

    def _save_by_kind(self, kind: str) -> None:
        batch_result = self._effective_batch_result_for_save()
        if not self._has_export_data(kind, batch_result):
            message = "Run segmentation first." if kind == "results" else "Run Analyze fibers first."
            self._show_info("Nothing to export", message)
            return

        save_options = self._normalize_save_options(self.config.save_options)
        if kind == "results" and not any(bool(save_options.get(key, False)) for key in self._save_artifact_keys()):
            self._show_warning(
                "Nothing selected",
                "All outputs are unchecked in the Export results panel.",
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
        spin.setValue(self._scalebar_tool.length_value)
        form.addRow("Length", spin)

        combo = QtWidgets.QComboBox(dialog)
        units = ["nm", "um", "mm", "m"]
        combo.addItems(units)
        combo.setCurrentText(self._scalebar_tool.unit)
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

        self._scalebar_tool.start(length_value, combo.currentText())
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

        pos = event.position()
        scene_pos = self.graphicsView.mapToScene(QPoint(int(round(pos.x())), int(round(pos.y()))))
        x = float(scene_pos.x())
        y = float(scene_pos.y())

        pix = self._pixitem.pixmap()
        width = pix.width()
        height = pix.height()
        if x < 0 or y < 0 or x >= width or y >= height:
            return

        px_size = self._scalebar_tool.add_point(x, y)
        if px_size is not None:
            self._finalize_scalebar_selection(pixel_size=px_size)
        elif not self._scalebar_tool.pending:  # two clicks on the same point
            self._cancel_scalebar_selection()

    def _finalize_scalebar_selection(self, pixel_size: float):
        self.sample_props.pixel_size_x = float(pixel_size)
        self.sample_props.pixel_size_y = float(pixel_size)
        self.sample_props.unit = str(self._scalebar_tool.unit or "px")
        self.sample_props.source = "manual"
        self.sync_sample_properties_to_ui()
        self._invalidate_measurements()

        self._show_info(
            "Scalebar Set",
            f"Pixel size set to {pixel_size:.6g} {self._scalebar_tool.unit} per pixel.",
        )
        self._cancel_scalebar_selection()

    def _cancel_scalebar_selection(self):
        self._scalebar_tool.cancel()
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
        self._update_reset_button_visibility()
        return True

    # ------------------- Batch processing -------------------

    @staticmethod
    def _supported_image_extensions() -> set[str]:
        return {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp", ".webp"}

    def _run_batch_analysis(
        self,
        compute_measurements: bool,
        image_batch: ImageBatch | None = None,
        error_title: str | None = None,
    ) -> BatchResult | None:
        """Run batch processing via `BatchService` and mirror progress in the UI."""
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

        cached_batch = (
            self._batch_result
            if (
                self._batch_result is not None
                and active_batch is self._image_batch
                and len(self._batch_result.frame_results) == len(active_batch)
            )
            else None
        )

        # Read settings once; the window stays disabled so they cannot change mid-batch.
        self.sync_ui_to_config()
        self.sync_ui_to_sample_properties()
        invert = bool(self.chb_invert.isChecked())

        QtWidgets.QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        previous_batch_state = self._suppress_batch_dialogs
        if len(active_batch) > 1:
            self._suppress_batch_dialogs = True
        try:
            self.setEnabled(False)
            def _on_started(total: int) -> None:
                self.statusbar.showMessage(f"Processing 0/{total} images")

            def _on_frame_started(index: int, total: int) -> None:
                self.statusbar.showMessage(f"Processing {index + 1}/{total} images")
                QtWidgets.QApplication.processEvents()

            def _on_frame_finished(_processed: int, _total: int, result: Frame) -> None:
                self.current_frame = result.snapshot()  # a copy: later slice changes mutate current_frame
                self.refresh_view()
                QtWidgets.QApplication.processEvents()

            callbacks = BatchCallbacks(
                on_started=_on_started,
                on_frame_started=_on_frame_started,
                on_frame_finished=_on_frame_finished,
                on_finished=lambda _result: None,
            )

            def _process_frame(
                frame: np.ndarray,
                frame_id: str,
                z_index: int,
                cached_frame: Frame | None,
                do_measurements: bool,
            ) -> Frame:
                if cached_frame is None:
                    result = Frame.empty(frame_id=frame_id, z_index=z_index, orig_img=frame)
                else:
                    result = cached_frame.snapshot(frame_id=frame_id, z_index=z_index)
                    result.orig_img = frame
                if do_measurements:
                    outcome = self._pipeline_service.ensure_measurements(
                        result, self.config, self.sample_props, invert_input=invert
                    )
                else:
                    outcome = self._pipeline_service.ensure_segmentation(result, self.config, invert_input=invert)
                if not self._handle_pipeline_outcome(outcome) and not result.error_message:
                    result.fail("Processing failed.")
                return result

            batch_result = self._batch_service.run_batch(
                active_batch,
                compute_measurements,
                process_frame=_process_frame,
                callbacks=callbacks,
                cached_result=cached_batch,
            )
        except Exception as exc:
            self._show_error(title, str(exc))
            logger.error("%s: %s", title, exc)
            return None
        finally:
            self._suppress_batch_dialogs = previous_batch_state
            self.setEnabled(True)
            QtWidgets.QApplication.restoreOverrideCursor()
            self.statusbar.clearMessage()

        # Publish batch outputs back into the GUI and refresh the active slice.
        self._batch_result = batch_result
        self._set_current_slice(self._current_z_index, update_view=True)
        self._refresh_save_actions()
        logger.info(
            "Batch %s completed successfully for %d frame(s).",
            "analysis" if compute_measurements else "segmentation",
            len(active_batch),
        )
        return batch_result

    def _on_batch_processing(self):
        self._run_batch_processing_for_folder(None)

    def _run_batch_processing_for_folder(self, folder_path: str | None):
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

        save_options = self._normalize_save_options(self.config.save_options)
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

    # ------------------- Profile plot tool -------------------

    def _ensure_profile_dialog(self) -> PlotProfileDialog:
        if self._profile_dialog is None:
            self._profile_dialog = PlotProfileDialog(self)
            self._profile_dialog.clearRequested.connect(self._on_profile_clear_requested)
            self._profile_dialog.closed.connect(self._on_profile_dialog_closed)
        return self._profile_dialog

    def _on_plot_profile_clicked(self):
        if not self._ensure_prediction_ready():
            return
        dialog = self._ensure_profile_dialog()
        if not dialog.isVisible():
            dialog.reset_plot()
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()
        self._profile_tool.start()
        self._reset_profile_selection(keep_dialog=True)

    def _on_profile_clear_requested(self):
        self._reset_profile_selection(keep_dialog=True)

    def _on_profile_dialog_closed(self):
        self._profile_tool.stop()
        self._profile_dialog = None
        self._reset_profile_selection()

    def _reset_profile_selection(self, keep_dialog: bool = False):
        self._profile_tool.reset()
        self._remove_profile_items()
        dialog = self._profile_dialog
        if dialog is not None and (keep_dialog or self._profile_tool.active):
            dialog.reset_plot()
        if self._profile_tool.active:
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
        if len(self._profile_tool.points_scene) < 2 or self._scene is None:
            return
        self._remove_profile_line()
        (x0, y0), (x1, y1) = self._profile_tool.points_scene[:2]
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

        pos = event.position()
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

        complete = self._profile_tool.add_point((sx, sy), (ix, iy))
        self._add_profile_marker(sx, sy)

        if complete:
            self._finalize_profile_selection()

    def _finalize_profile_selection(self):
        if len(self._profile_tool.points_scene) < 2 or len(self._profile_tool.points_img) < 2:
            return
        self._draw_profile_line()
        self._update_profile_plot()

    def _update_profile_plot(self):
        if not self.current_frame.has_prediction or len(self._profile_tool.points_img) < 2:
            return
        (x0, y0), (x1, y1) = self._profile_tool.points_img[:2]
        profile = self._profile_tool.compute_profile(
            self.current_frame.dt,
            (x0, y0),
            (x1, y1),
            scale_factor=float(self.current_frame.scale_factor),
        )
        if profile is None:
            return
        distances, profile_vals_true, summary = profile
        dialog = self._ensure_profile_dialog()
        dialog.update_profile(distances, profile_vals_true, summary)

    # ------------------- Segmentation -------------------

    def _run_current_frame_pipeline(self, compute_measurements: bool) -> bool:
        if compute_measurements:
            return self._ensure_measurements_ready()
        return self._ensure_segmentation_ready()

    def segment(self):
        """Run the segmentation workflow for the current frame or active batch."""
        if self._image_batch is not None:
            self._run_batch_analysis(compute_measurements=False)
            return
        if not self._run_current_frame_pipeline(compute_measurements=False):
            return

        if self.cb_view.currentText() == "Original":
            self.cb_view.setCurrentText("Overlay")

        self._refresh_save_actions()
        self.refresh_view()
        logger.info("Segmentation completed successfully.")

    def analyze_fibers(self):
        """
        Computes thickness samples and renders overlays/plots.
        """
        if self._image_batch is not None:
            if self.cb_view.currentText() != "Measurements overlay":
                self.cb_view.setCurrentText("Measurements overlay")
            if self._batch_measurements_match_current_settings():
                self._set_current_slice(self._current_z_index, update_view=True)
                return
            self._run_batch_analysis(compute_measurements=True)
            return

        if not self._run_current_frame_pipeline(compute_measurements=True):
            return
        self._render_vector_overlay()

        fr = self.current_frame
        summary = fr.fiber_summary or {}
        units_label = str(summary.get("units", "px"))
        diam_mean = float(summary.get("diam_mean", float("nan")))
        diam_sd = float(summary.get("diam_sd", float("nan")))
        diam_mode = float(summary.get("diam_mode", float("nan")))
        diam_median = float(summary.get("diam_median", float("nan")))
        orientation_index = float(summary.get("orientation_index", float("nan")))
        mean_angle_deg = float(summary.get("mean_angle_deg", float("nan")))
        std_angle_deg = float(summary.get("std_angle_deg", float("nan")))
        spacing_px_input = max(1.0, float(self.config.sample_spacing))
        self._show_info(
            "Analysis complete",
            f"Measurement Samples: {int(summary.get('sample_count', 0))} (spacing={int(spacing_px_input)} px)\n"
            f"Drawing Samples: {len(fr.drawing_samples) if fr.drawing_samples is not None else 0}\n"
            f"Diam Mean: {diam_mean:.2f} {units_label}\n"
            f"Diam SD: {diam_sd:.2f} {units_label}\n"
            f"Diam Mode: {diam_mode:.2f} {units_label}\n"
            f"Diam Median: {diam_median:.2f} {units_label}\n"
            f"Orientation Index: {orientation_index:.2f}\n"
            f"Angle Mean: {mean_angle_deg:.2f} deg\n"
            f"Angle SD: {std_angle_deg:.2f} deg"
        )
        self._refresh_save_actions()
        logger.info("Fiber analysis completed successfully.")

    def _render_vector_overlay(self):
        if self.current_frame.labels_2d is None:
            return
        if self.cb_view.currentText() not in ("Measurements overlay", "Sampling"):
            self.cb_view.setCurrentText("Measurements overlay")
        self.refresh_view()

    def _style_plotwidget(self, pw: pg.PlotWidget, title: str, xlab: str, ylab: str):
        pw.setBackground("w")
        pw.showAxis("bottom")
        pw.showAxis("left")
        pw.clear()
        pw.setTitle(title, color="k", size="12pt")
        pw.showGrid(x=True, y=True, alpha=0.25)
        ax = pw.getAxis("bottom"); ax.setPen("k"); ax.setTextPen("k"); ax.setLabel(xlab, color="k")
        ay = pw.getAxis("left");   ay.setPen("k"); ay.setTextPen("k"); ay.setLabel(ylab, color="k")
        pw.setAntialiasing(True)

    def _update_distributions_pyqtgraph(self, units_label: str):
        self.text_edit2.setVisible(True)
        self.label_units.setVisible(True)
        self.cb_units.setVisible(True)
        # Thickness
        diameters = (
            self.current_frame.thickness_units
            if self.current_frame.thickness_units is not None
            else np.array([], dtype=np.float32)
        )
        self._style_plotwidget(self.graph_diameters, "Diameter Distribution", f"Diameter ({units_label})", "Count")
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
            ang = (
                np.array([sample[5] for sample in (self.current_frame.measurement_samples or [])], dtype=np.float32)
                if self.current_frame.measurement_samples is not None
                else np.array([], dtype=np.float32)
            )
            self._style_plotwidget(self.graph_angles, "Orientation Distribution", "Angle (deg)", "Count")
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
            self._style_plotwidget(self.graph_angles, "Pore Diameter Distribution", f"Diameter ({units_label})", "Count")
            ds = self.current_frame.pore_diams_units if self.current_frame.pore_diams_units is not None else np.array([], dtype=np.float32)
            if ds.size:
                hist, edges = np.histogram(ds, bins=40)
                centers = 0.5 * (edges[:-1] + edges[1:])
                widths  = (edges[1:] - edges[:-1])
                bg = pg.BarGraphItem(x=centers, height=hist, width=widths,
                                    brush=pg.mkBrush("#70AD47"),
                                    pen=pg.mkPen(color="#507E2D", width=1))
                self.graph_angles.addItem(bg)

        # Fit the complete histograms even after the user has zoomed or panned.
        self.graph_diameters.autoRange()
        self.graph_angles.autoRange()

    # ------------------- View & reset -------------------

    def refresh_view(self) -> np.ndarray | None:
        """Show the image of the selected view and return it."""
        view = self.cb_view.currentText()
        measurement_views = ("Pores", "Orientations", "Diameters")
        if view == "Measurements overlay":
            view = measurement_views[0 if self.rb_show_pores.isChecked() else 1 if self.rb_show_direct.isChecked() else 2]
        img = render(self.current_frame, view, self.config.grad_quality_points)
        if img is None and view in measurement_views:
            img = render(self.current_frame, "Overlay")
        self._set_pixmap_from_array(img)
        return img

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
        if self._image_batch is None:
            self.label_image_name.setText("Image: none")
            self.label_image_name.setToolTip("")
            return
        total = len(self._image_batch)
        current = self._current_z_index + 1 if total > 0 else 0
        kind = self._image_batch.kind
        if kind == "Folder":
            filename = self._image_batch.metadata["filenames"][self._current_z_index]
            text = f"Image: {filename} ({current}/{total})"
        elif kind == "Stack":
            text = f"Stack: {self._image_batch.name} (slice {current}/{total})"
        else:
            text = f"Image: {self._image_batch.name}"
        self.label_image_name.setText(text)
        self.label_image_name.setToolTip(text)

    def _frame_base_name(self, frame_id: str, z_index: int) -> str:
        base = os.path.splitext(str(frame_id))[0]
        if self._image_batch is None:
            return base
        source_type = self._image_batch.metadata.get("source_type")
        if source_type == "Stack" and len(self._image_batch) > 1:
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
            self._clear_results()
            return

        applied_frame = frame_result.snapshot()
        applied_frame.orig_img = current_orig
        self.current_frame = applied_frame

        if self.current_frame.fiber_summary:
            units_label = self.current_frame.fiber_summary.get("units", "px")
            info_text = (
                f"Diam Mean: {self.current_frame.fiber_summary.get('diam_mean', float('nan')):.2f} {units_label}\n"
                f"Diam SD: {self.current_frame.fiber_summary.get('diam_sd', float('nan')):.2f} {units_label}\n"
                f"Diam Mode: {self.current_frame.fiber_summary.get('diam_mode', float('nan')):.2f} {units_label}\n"
                f"Diam Median: {self.current_frame.fiber_summary.get('diam_median', float('nan')):.2f} {units_label}\n"
                f"Orientation Index: {self.current_frame.fiber_summary.get('orientation_index', float('nan')):.2f}\n"
                f"Angle Mean: {self.current_frame.fiber_summary.get('mean_angle_deg', float('nan')):.2f} deg\n"
                f"Angle SD: {self.current_frame.fiber_summary.get('std_angle_deg', float('nan')):.2f} deg\n"
                f"Porosity: {self.current_frame.fiber_summary.get('porosity', float('nan'))*100:.1f} %\n"
                f"Centerline Fragments: {int(self.current_frame.fiber_summary.get('fragments_with_centerlines', 0))}"
            )
            self.text_edit2.setPlainText(info_text)
            self._update_distributions_pyqtgraph(units_label)
        else:
            self._clear_results()

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

    def _reset_processing_state(self, preserve_original: bool = False):
        """Reset frame processing state, without touching UI widgets/charts."""
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

        self._scalebar_tool.cancel()
        self.graphicsView.viewport().unsetCursor()

        self._profile_tool.stop()
        self._remove_profile_items()
        self.btn_reset_view.hide()

        # ---- results (stats + pyqtgraph charts) ----
        self._clear_results()

        # ---- UI actions / state ----
        self._pixitem.setPixmap(QtGui.QPixmap())

        self._refresh_save_actions()

        if reset_view:
            self.cb_view.setCurrentText("Original")

    def _on_usegpu_toggled(self, checked: bool):
        cuda_ok = onnx_cuda_available()
        if checked and not cuda_ok:
            QMessageBox.warning(self, "CUDA not available", "ONNX CUDA provider not available. Using CPU.")
            self.chb_usegpu.setChecked(False)
            self.config.use_gpu = False
        else:
            self.config.use_gpu = checked
            
        self._pipeline_service.set_model(self._pipeline_service.model_path, "cuda" if self.config.use_gpu else "cpu")

    def _on_invert_toggled(self, checked: bool):
        self._invalidate_prediction()

    # ------------------- New default image loading -------------------

    def load_default_image(self):
        """Load default image from persisted storage and normalize it for use."""
        default_img_path = str(EXAMPLE_IMAGE_PATH)
        if os.path.exists(default_img_path):
            self.open_source(path=default_img_path, source_kind="file")
        else:
            logger.info("Default image not found: %s", default_img_path)


    def sync_ui_to_sample_properties(self):
        """Reads sample-specific calibration fields from the UI."""
        self.sample_props, changed = self._ui_binder.sync_ui_to_sample_properties(self, self.sample_props)
        if changed:
            self._invalidate_measurements()

    def sync_sample_properties_to_ui(self):
        """Updates sample-specific calibration widgets from self.sample_props."""
        self._ui_binder.sync_sample_properties_to_ui(self, self.sample_props)
        self._update_pixel_size_unit_label()

    def sync_ui_to_config(self):
        """Reads the current state of the UI and stores it in self.config."""
        self.config, flags = self._ui_binder.sync_ui_to_config(self, self.config)
        if flags.prediction_changed:
            self._invalidate_prediction()
        elif flags.segmentation_changed:
            self._invalidate_segmentation()

    def sync_config_to_ui(self):
        """Updates the UI widgets to reflect the values in self.config."""
        self._ui_binder.sync_config_to_ui(self, self.config)
        self._sync_save_options_to_ui()

    # ------------------- Panel helpers -------------------

    def _clear_results(self) -> None:
        """Keep the results strip reserved but visually empty until measurements exist."""
        self.text_edit2.clear()
        self.text_edit2.setVisible(False)
        self.label_units.setVisible(False)
        self.cb_units.setVisible(False)
        for pw in (self.graph_diameters, self.graph_angles):
            pw.clear()
            pw.setTitle(None)
            pw.hideAxis("bottom")
            pw.hideAxis("left")
            pw.setBackground(self.palette().color(QtGui.QPalette.ColorRole.Window))

    @staticmethod
    def _toggle_advanced(button: QToolButton, panel: QtWidgets.QWidget, checked: bool) -> None:
        panel.setVisible(checked)
        button.setArrowType(Qt.ArrowType.DownArrow if checked else Qt.ArrowType.RightArrow)

    def _update_pixel_size_unit_label(self) -> None:
        self.label_pixel_unit.setText(self.cb_units.currentText().strip() or "px")

    def _on_units_changed(self, _text: str) -> None:
        """Units are picked next to the plots: re-measure so plots and stats follow the new unit."""
        had_measurements = self._batch_result is not None and any(
            frame.has_measurements for frame in self._batch_result.frame_results
        )
        self.sync_ui_to_sample_properties()
        self._update_pixel_size_unit_label()
        if had_measurements:
            view = self.cb_view.currentText()
            self._run_batch_analysis(compute_measurements=True)
            self.cb_view.setCurrentText(view)
