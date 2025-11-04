# app/ui_main.py
import os
import numpy as np
import cv2

from PyQt6 import QtGui, QtWidgets
from PyQt6.QtCore import Qt, QEvent, QTimer
from PyQt6.QtWidgets import QMainWindow, QFileDialog, QMessageBox, QGraphicsScene, QGraphicsPixmapItem
from PyQt6.uic import loadUi

import imageio.v3 as iio
import pyqtgraph as pg
import torch

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

from .fiber_analyzer import FiberAnalyzer
from .utils import ensure_rgb_u8, to_uint8_for_view

# ------------------- Paths -------------------

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
DEFAULT_UI_PATH = os.path.join(PROJECT_ROOT, "resources", "fiber_tool.ui")
DEFAULT_CKPT_PATH = os.path.join(PROJECT_ROOT, "checkpoints", "fiber_2000_epoch.pt")

# ------------------- Image helpers -------------------

def labels_to_color(labels: np.ndarray) -> np.ndarray:
    if labels is None:
        return None
    labels = labels.astype(np.int64, copy=False)
    h, w = labels.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    if labels.size == 0:
        return out
    unique_vals = [v for v in np.unique(labels) if v != 0]
    if not unique_vals:
        return out
    rng = np.random.default_rng()
    total_colors = 256 ** 3
    colors_flat = rng.choice(total_colors, size=len(unique_vals), replace=False)
    colors = np.stack(
        (
            (colors_flat >> 16) & 0xFF,
            (colors_flat >> 8) & 0xFF,
            colors_flat & 0xFF,
        ),
        axis=1,
    ).astype(np.uint8)
    for idx, v in enumerate(unique_vals):
        out[labels == v] = colors[idx]
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
        self.dt = None
        self.gray_u8 = None          # CPNet DT (0..255)
        self.gray_img = None         # normalized float [0..1]
        self.lapl_img = None
        self.gx = None
        self.gy = None
        self.analyzer: FiberAnalyzer | None = None

        self.labels_img = None       # watershed labels (uint16)
        self.overlay_img = None
        self.debug_overlay = None

        self.samples_table = None
        self.fiber_summary = None
        self._viz_points = None
        self._angles_deg = None
        self._thickness_units = None
        self._porosity = None
        self._pore_diams_units = None
        self._labels_rgb_cache: tuple[int | None, np.ndarray | None] = (None, None)

        self.high_conf_fib = False
        self.chb_high_conf.setChecked(False)

        # Menu & actions: updated UI element names (rb_mask1_cent -> rb_centerlines; rb_mask2_segm -> rb_segmentation;
        # sp_threshold1 -> sp_max_cent_angle; sp_threshold2 -> sp_segm_threshold)
        self.action_load_image.triggered.connect(self.load_original)
        self.action_save_masks_predictions.triggered.connect(self.save_predictions)
        self.action_save_measurements.triggered.connect(self.save_measurements)

        self.btn_segment.clicked.connect(self.segment)
        self.btn_analyze_fibers.clicked.connect(self.analyze_fibers)
        self.btn_save_csv.clicked.connect(self.save_measurements)

        # Remove connection of radiobutton signals and instead hook up the combobox change
        if hasattr(self, "cb_view"):
            self.cb_view.currentIndexChanged.connect(self.refresh_view)

        # GUI defaults
        self.le_minarea.setText("50")
        self.le_mineccentricity.setText("0.9")
        self.le_minskeletonlen.setText("20")
        self.le_samspacing.setText("5")
        self.le_pixelsize.setText("")

        if hasattr(self, "cb_model"):
            self.cb_model.clear()
            self.cb_model.addItem("checkpoints/fiber_2000_epoch.pt")

        # Disable saves initially
        self.action_save_masks_predictions.setEnabled(False)
        self.action_save_measurements.setEnabled(False)

        self.resize(1100, 800)
        self.refresh_view()
        
        # Delay loading default image until the UI is fully shown.
        QTimer.singleShot(0, self.load_default_image)

        self.ckpt_path = DEFAULT_CKPT_PATH

        # ---- device / CUDA handling ----
        self.device = torch.device('cpu')   # default
        self._init_device_from_checkbox()   # sets checkbox + self.device accordingly

        # Analyzer instantiated once and reused
        self.analyzer = FiberAnalyzer(
            ckpt_path=self.ckpt_path,
            device=self.device,
            high_conf_fib=self.high_conf_fib,
        )

        # react to checkbox changes at runtime
        self.chb_usegpu.toggled.connect(self._on_usegpu_toggled)
        self.chb_high_conf.toggled.connect(self._on_high_conf_toggled)

        # Connect toggled signals of rb_show_diam and rb_show_direct to update overlay
        self.rb_show_diam.toggled.connect(self._render_vector_overlay)
        self.rb_show_direct.toggled.connect(self._render_vector_overlay)
        self.rb_show_pores.toggled.connect(self._render_vector_overlay)


    # ------------------- Display helpers -------------------

    def _set_pixmap_from_array(self, img: np.ndarray | None):
        if img is None:
            self._pixitem.setPixmap(QtGui.QPixmap())
            return
        qimg = self._numpy_to_qimage(img)
        pix = QtGui.QPixmap.fromImage(qimg)
        # Scale pixmap if it is smaller than the viewport
        view_size = self.graphicsView.viewport().size()
        if pix.width() < view_size.width() or pix.height() < view_size.height():
            pix = pix.scaled(view_size, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self._pixitem.setPixmap(pix)
        self._fit_view_to_pixmap()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._fit_view_to_pixmap()


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
        
    def _fit_view_to_pixmap(self):
        pm = self._pixitem.pixmap()
        if pm is not None and not pm.isNull():
            self.graphicsView.fitInView(self._pixitem, Qt.AspectRatioMode.KeepAspectRatio)
            self.graphicsView.centerOn(self._pixitem)

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


    # ------------------- Drag & drop -------------------

    def _install_drag_drop(self):
        v = self.graphicsView
        v.setAcceptDrops(True)
        v.viewport().setAcceptDrops(True)
        v.viewport().installEventFilter(self)

    def eventFilter(self, obj, event):
        if obj is self.graphicsView.viewport():
            et = event.type()
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
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to read image:\n{e}")
            logger.error(f"Failed to read image: {e}")
            return
        
        logger.info(f"Image loaded successfully: {path}")

        if img.ndim not in (2, 3):
            QtWidgets.QMessageBox.warning(self, "Unsupported", "Original must be 2D or 3D (H×W×C).")
            return
        self.orig_img = img
        self.refresh_view()

    def save_predictions(self):
        if self.gray_u8 is None and self.labels_img is None and self.overlay_img is None:
            QtWidgets.QMessageBox.information(self, "Nothing to save", "Run segmentation first.")
            return
        base, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save predictions (choose base name)",
            os.path.join(self.working_dir, "fiberpose_predictions.png"),
            "PNG (*.png);;TIFF (*.tif *.tiff)"
        )
        if not base:
            return
        root, _ = os.path.splitext(base)
        logger.info(f"Saving predictions with base name: {root}")

        try:
            if self.gray_u8 is not None:
                iio.imwrite(root + "_dt.png", np.asarray(self.gray_u8, dtype=np.uint8))
            if self.labels_img is not None and self.labels_img.max() > 0:
                iio.imwrite(root + "_labels.png", labels_to_color(self.labels_img))
            if self.overlay_img is not None:
                iio.imwrite(root + "_overlay.png", self.overlay_img.astype(np.uint8))
            if self.debug_overlay is not None:
                iio.imwrite(root + "_debug.png", self.debug_overlay.astype(np.uint8))
            self.working_dir = os.path.dirname(base)
            QtWidgets.QMessageBox.information(self, "Saved", f"Saved to:\n{os.path.dirname(base)}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to save:\n{e}")

    def save_measurements(self):
        if self.samples_table is None or len(self.samples_table) == 0:
            QtWidgets.QMessageBox.information(self, "Nothing to save", "Run Analyze fibers first.")
            return
        base, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save measurements (CSV)",
            os.path.join(self.working_dir, "fiberpose_measurements.csv"),
            "CSV (*.csv)"
        )
        if not base:
            return
        root, _ = os.path.splitext(base)
        logger.info(f"Saving measurements to base: {root}")

        try:
            with open(root + "_samples.csv", "w", newline="") as f:
                # Write header with both thickness in pixels and in units.
                f.write("x,y,thickness_px,thickness_units,fiber_id\n")
                # Check if a pixel size was provided (if not, leave thickness_units blank).
                units_set = bool(self.le_pixelsize.text().strip())
                for i, row in enumerate(self.samples_table):
                    x, y, t = row[:3]
                    if units_set and self._thickness_units is not None and len(self._thickness_units) > i:
                        units_value = f"{self._thickness_units[i]:.6f}"
                    else:
                        units_value = ""
                    f.write(f"{int(x)},{int(y)},{float(t):.6f},{units_value},0\n")
        except Exception as e:
             QtWidgets.QMessageBox.critical(self, "Error", f"Failed to save CSVs:\n{e}")
             return
        QtWidgets.QMessageBox.information(self, "Saved", f"Saved CSV(s) to:\n{os.path.dirname(base)}")


    # ------------------- Segmentation -------------------

    def segment(self):
        if self.orig_img is None:
            QMessageBox.information(self, "Load original", "Please load an original image first.")
            logger.info("Segment: No original image loaded.")
            return
        if self.analyzer is None:
            QMessageBox.critical(self, "Analyzer unavailable", "Fiber analyzer has not been initialized.")
            logger.error("Segmentation failed: analyzer not initialized.")
            return

        #Dt not predicted, predict dt
        if self.analyzer.dt_predicted() is False:

            logger.info("Starting Distance transform prediction.")
            self.progress_bar.setValue(0)

            try:
                dt, gx, gy = self.analyzer.predict_dt(
                    image=self.orig_img,
                    max_angle_deg=float(self.sp_max_cent_angle.value()),
                )
            except Exception as e:
                QMessageBox.critical(self, "Prediction failed", str(e))
                logger.error("Segmentation failed: %s", e)
                return

            self.dt = np.asarray(dt, dtype=np.float32, copy=False)
            self.gx = np.asarray(gx, dtype=np.float32, copy=False)
            self.gy = np.asarray(gy, dtype=np.float32, copy=False)
            self.gray_u8 = self.analyzer.get_dt_u8()
            self.gray_img = self.analyzer.get_dt_norm()
            self.lapl_img = self.analyzer.get_laplacian()
            self._centerlines_dt = self.analyzer.get_centerlines_image()

        thr_u8 = int(self.sp_segm_threshold.value())
        radius_thresh_px: float | None = float(self.sp_segm_threshold.value()) if self.high_conf_fib else None
        try:
            self.labels_img = self.analyzer.segment(
                thr_u8=thr_u8,
                radius_thresh_px=radius_thresh_px,
            )
        except Exception as e:
            QMessageBox.critical(self, "Segmentation failed", str(e))
            logger.error("Watershed failed: %s", e)
            return

        if self.labels_img is not None:
            labels_rgb = labels_to_color(self.labels_img)
            self._labels_rgb_cache = (id(self.labels_img), labels_rgb)
        else:
            labels_rgb = None
            self._labels_rgb_cache = (None, None)

        self.overlay_img = self._create_overlay(self.orig_img, self.labels_img, alpha=0.5, labels_rgb=labels_rgb)

        if self.cb_view.currentText() not in ["Overlay", "Segmentation", "Measurements"]:
            self.cb_view.setCurrentText("Overlay")

        if hasattr(self, "action_save_masks_predictions"):
            self.action_save_masks_predictions.setEnabled(True)

        self.refresh_view()
        self.progress_bar.setValue(80)
        logger.info("Segmentation completed successfully.")

    def analyze_fibers(self):
        """
        Computes thickness samples via analyzer and renders overlays/plots.
        """
        if self.gray_u8 is None or self.labels_img is None:
            self.segment()
            if self.gray_u8 is None or self.labels_img is None:
                QtWidgets.QMessageBox.information(self, "No segmentation", "Segmentation could not be performed.")
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
        spacing_px = max(1.0, float(_getf(self.le_samspacing, 5)))
        pixel_size  = float(_getf(self.le_pixelsize, 1.0))
        if hasattr(self, "cb_units") and self.cb_units.currentText().strip() and self.le_pixelsize.text().strip():
            units_label = self.cb_units.currentText()
        else:
            units_label = "px"

        # compute via analyzer
        if self.analyzer is None:
            QtWidgets.QMessageBox.information(self, "No analyzer", "Segmentation data unavailable.")
            return

        if not self.analyzer.get_centerline_segments():
            QtWidgets.QMessageBox.information(self, "No centerlines", "No centerlines available. Run segmentation first.")
            return

        hist_centers, viz_points, angles_deg = self.analyzer.analyze(
            spacing_px=spacing_px
        )
        self._viz_points = viz_points

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
        thr_u8 = int(self.sp_segm_threshold.value()) if hasattr(self, "sp_segm_threshold") else 128

        # Build void mask from your chosen source (DT or grayscale-based threshold).
        # If you already have self._void_mask, reuse it.
        void_mask_u8 = getattr(self, "_void_mask", None)

        if void_mask_u8 is None:
            # If your pipeline stores grayscale in self.gray_u8, pass that DT/threshold you use
            self._void_mask, self._pore_dt = self.analyzer.void_mask_and_pore_dt(thr_u8)

        self._porosity = float(self._void_mask.mean())

        # --- NEW: sample pores as inscribed circles from DT peaks ---------------------
        min_radius_px = 1.5   # tweakable; ignores tiny specks
        peak_footprint     = 5     # local-max window; 3..7 works; larger -> fewer peaks
        max_pores     = None  # or set an upper cap for speed/UI clarity
        min_peak_px = 3.0

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
        self._pore_diams_units = np.array([2.0 * p['r_px'] * pixel_size for p in self._pores], dtype=np.float32)




        # Summary text
        if self._thickness_units.size:
            mean_thickness = np.mean(self._thickness_units)
            std_thickness  = np.std(self._thickness_units)
        else:
            mean_thickness = float('nan')
            std_thickness  = float('nan')
        if self._angles_deg.size:
            angles_rad = np.deg2rad(self._angles_deg)
            mean_cos = np.mean(np.cos(angles_rad))
            mean_sin = np.mean(np.sin(angles_rad))
            orientation_index = np.hypot(mean_cos, mean_sin)
        else:
            orientation_index = float('nan')

        info_text = (
            f"Mean diameter: {mean_thickness:.2f} {units_label}\n"
            f"Std diameter: {std_thickness:.2f} {units_label}\n"
            f"Orientation index: {orientation_index:.2f}\n"
            f"Porosity (area fraction): {self._porosity*100:.1f} %"
        )
        if hasattr(self, "text_edit2"):
            self.text_edit2.setPlainText(info_text)

        # Overlay + plots
        self._render_vector_overlay()
        self._update_distributions_pyqtgraph(units_label)
        self.progress_bar.setValue(100)

        QtWidgets.QMessageBox.information(
            self, "Analysis complete",
            f"Histogram samples: {len(hist_centers)} (spacing={int(spacing_px)} px)\n"
            f"Stored viz points: {len(self._viz_points)}\n"
            f"Mean diameter: {mean_thickness:.2f} {units_label}\n"
            f"Std diameter: {std_thickness:.2f} {units_label}\n"
            f"Orientation index: {orientation_index:.2f}"
        )

        self.action_save_measurements.setEnabled(True)
        logger.info("Fiber analysis completed successfully.")

    # ---- single overlay renderer (kept once) ----
    def _render_vector_overlay(self):
        if not getattr(self, "_viz_points", None):
            return
        base = self.overlay_img.copy() if self.overlay_img is not None else ensure_rgb_u8(self.gray_u8)
        dbg  = base.copy()
        H, W = dbg.shape[:2]

        show_diam   = getattr(self, "rb_show_diam", None) and self.rb_show_diam.isChecked()
        show_direct = getattr(self, "rb_show_direct", None) and self.rb_show_direct.isChecked()
        show_pores  = getattr(self, "rb_show_pores", None) and self.rb_show_pores.isChecked()

        if not show_diam and not show_direct and not show_pores:
            show_diam = True

        MAX_SEGMENTS = 15000
        pts = self._viz_points if len(self._viz_points) <= MAX_SEGMENTS else self._viz_points[::int(np.ceil(len(self._viz_points)/MAX_SEGMENTS))]

        if not show_pores:
            for (y, x, r, ny, nx, ty, tx) in pts:
                if not (0 <= y < H and 0 <= x < W) or r <= 0:
                    continue
                if show_diam:
                    p0 = (y - ny * r, x - nx * r)
                    p1 = (y + ny * r, x + nx * r)
                    self._draw_line(dbg, p0, p1, color=(255, 255, 0))
                else:
                    self._draw_arrow_centered(dbg, float(y), float(x), float(tx), float(ty), length_px=2.0 * r, color=(255, 255, 0))

        else:
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

    


        self.debug_overlay = dbg
        # Set view to Measurements after rendering vector overlay
        if hasattr(self, "cb_view"):
            self.cb_view.setCurrentText("Measurements")
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
        if not hasattr(self, "graph_diameters") or not hasattr(self, "graph_angles"):
            return

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
        if hasattr(self, "cb_view"):
            view = self.cb_view.currentText()
        else:
            view = "Original"
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
        elif view == "Measurements":
            seg_rgb = self._labels_rgb()
            img = self.debug_overlay if self.debug_overlay is not None else seg_rgb
        elif view == "gx": 
            img = to_uint8_for_view(self.gx) if hasattr(self, "gx") else None
        elif view == "gy":
            img = to_uint8_for_view(self.gy) if hasattr(self, "gy") else None

        self._set_pixmap_from_array(img)

    def _reset_internal_state(self, preserve_original: bool = False):
        """Clear all derived data, images, analysis artifacts, and charts."""
        
        self.analyzer.reset_prediction_state()
        # ---- raw / processing buffers ----
        if not preserve_original:
            self.orig_img = None
        self.dt = None
        self.gray_img = None
        self.gray_u8 = None
        self.lapl_img = None
        self.gx = None
        self.gy = None

        # ---- segmentation outputs ----
        self.labels_img = None
        self.overlay_img = None
        self._centerlines_dt = None

        # ---- analysis / debug ----
        self.samples_table = None
        self.fiber_summary = None
        self.debug_overlay = None
        self._viz_points = None
        self._angles_deg = None
        self._thickness_units = None
        self._void_mask = None
        self._pore_dt = None
        self._pores = None
        self._porosity = None
        self._pore_diams_units = None
        self._labels_rgb_cache = (None, None)

        # ---- charts (pyqtgraph) ----
        if hasattr(self, "graph_diameters") and isinstance(self.graph_diameters, pg.PlotWidget):
            self.graph_diameters.clear()
            self.graph_diameters.setTitle("")
            self.graph_diameters.showGrid(x=False, y=False)
        if hasattr(self, "graph_angles") and isinstance(self.graph_angles, pg.PlotWidget):
            self.graph_angles.clear()
            self.graph_angles.setTitle("")
            self.graph_angles.showGrid(x=False, y=False)

        # ---- UI actions / state ----
        if hasattr(self, "_pixitem"):
            self._pixitem.setPixmap(QtGui.QPixmap())

        if hasattr(self, "action_save_masks_predictions"):
            self.action_save_masks_predictions.setEnabled(False)
        if hasattr(self, "action_save_measurements"):
            self.action_save_measurements.setEnabled(False)

        if hasattr(self, "cb_view"):
            self.cb_view.setCurrentText("Original")

    def _init_device_from_checkbox(self):
        """On startup: if CUDA is available, check the box and use cuda; else uncheck and use cpu."""
        cuda_ok = False
        try:
            cuda_ok = torch.cuda.is_available()
        except Exception:
            cuda_ok = False

        if hasattr(self, "chb_usegpu"):
            if cuda_ok:
                # user prefers GPU by default when available
                self.chb_usegpu.setChecked(True)
                self.device = torch.device('cuda')
            else:
                self.chb_usegpu.setChecked(False)
                self.device = torch.device('cpu')
                # notify once at init if the UI is asking for GPU (edge case: pre-checked in .ui)
                if getattr(self.chb_usegpu, "isChecked", lambda: False)():
                    QtWidgets.QMessageBox.information(
                        self, "CUDA not available",
                        "No CUDA-capable GPU detected. Falling back to CPU."
                    )
        else:
            self.device = torch.device('cuda' if cuda_ok else 'cpu')

    def _on_usegpu_toggled(self, checked: bool):
        """When user toggles the GPU checkbox, validate and set device; notify if not available."""
        cuda_ok = torch.cuda.is_available()
        if checked and not cuda_ok:
            QtWidgets.QMessageBox.warning(
                self, "CUDA not available",
                "CUDA GPU not detected on this system. Using CPU instead."
            )
            # force UI to reflect reality
            self.chb_usegpu.setChecked(False)
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda' if checked else 'cpu')

        if self.analyzer is not None:
            self.analyzer.set_device(self.device)

    def _on_high_conf_toggled(self, checked: bool):
        self.high_conf_fib = checked
        if self.analyzer is not None:
            self.analyzer.set_high_conf_filter(checked)

    # ------------------- New default image loading -------------------

    def load_default_image(self):
        default_img_path = os.path.join(PROJECT_ROOT, "resources", "test.png")
        if os.path.exists(default_img_path):
            self.load_original_from_path(default_img_path)
        else:
            logger.info("Default image not found: resources/test.png")


