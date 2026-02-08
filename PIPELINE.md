# Fiberpose Pipeline

## Scope
- Document runtime pipeline used by `MainWindow` + `FiberAnalyzer`.
- Cover single-image, batch, and stack/3D visualization paths.

## Modes
- `segmentation.mode=watershed`: uses `FiberAnalyzer.segment_watershed(thr_u8)`.
- `segmentation.mode=ridge`: uses `FiberAnalyzer.segment_strict_ridge(radius_thresh_px, ...)`.
- `strict_centerlines`: affects centerline filtering only; in GUI it is enabled only in ridge mode.
- `clustering.enabled`: toggles fragment stitching before marker/render generation.
- `pores view`: uses pore circles + pore diameter histogram instead of orientation histogram.

## Core Data Products
- `image`: `(H,W)` or `(H,W,C)` input frame.
- `dt`: `(H,W) float32`, model DT prediction (channel 2).
- `dt_u8`: `(H,W) uint8`, clipped DT.
- `gx, gy`: `(H,W) float32`, DT gradients.
- `fragments`: `list[FiberFragment(points_xy, radii, parent_fiber_id)]`.
- `markers`: `(H,W) int32`, watershed seeds.
- `labels`: `(H,W) uint16`, segmentation map.
- `hist_centers`: thickness samples from centerlines.
- `void_mask`, `pore_dt`, `pores`: pore products for porosity and pore-size distribution.

## Single-Frame Flow (Image -> Distributions)
1. `MainWindow.segment()`
2. `FiberAnalyzer.predict_dt(image, max_angle_deg, centerline_kwargs)`
3. `_run_model()` -> `FiberposeModel.predict()` -> `ResUnet.forward()` -> `dt`
4. DT postprocess: smooth + gradients + laplacian; cache state
5. `FiberAnalyzer.process_centerlines(...)`
6. `compute_centerlines(...)` -> `RidgeDetector.detect_ridges(...)` -> split/trim -> sliding-window filter -> marker rendering
7. `mode switch`:
- watershed: `FiberAnalyzer.segment_watershed(thr_u8)`
- ridge: `FiberAnalyzer.segment_strict_ridge(radius_thresh_px, dilate_extra_px, min_run_points)`
8. `MainWindow.analyze_fibers()`
9. `FiberAnalyzer.analyze(spacing_px)` -> `hist_centers`, `viz_points`, `angles_deg`
10. Thickness distribution: `np.histogram(_thickness_units, bins=40)`
11. Pores: `void_mask_and_pore_dt(thr_u8)` + `sample_inscribed_circles(...)`
12. Pore distribution (when pores view ON): `np.histogram(_pore_diams_units, bins=40)`

## Batch Flow
1. Build `ImageBatch` via `ImageBatch.from_path(...)` or `ImageBatch.from_folder(...)`.
2. `MainWindow._run_batch_analysis(...)` creates per-frame callback.
3. `BatchAnalyzer.analyze(...)` loops frames and resets analyzer state each frame.
4. Per frame callback:
- set `orig_img`
- `_reset_internal_state(preserve_original=True)`
- run `segment()` or `analyze_fibers()`
- capture `FrameResult` via `_capture_frame_result(...)`
5. Return `BatchResult(frame_results=[...])`.
6. Optional exports via `_save_batch_outputs(...)` (`dt`, `labels`, `overlay`, `debug`, `samples`, `style_vector`).
7. Slice navigation rehydrates cached data with `_apply_frame_result(frame_result)`.

## Stack / 3D Flow
1. `ImageBatch._normalize_stack(...)` maps input to ordered frame list:
- `HxW` -> single frame
- `HxWxC` -> single frame
- `ZxHxW` -> grayscale stack
- `ZxHxWxC` -> color stack
2. `Stack3DWindow` builds aligned stack from `BatchResult.frame_results`.
3. 2D projection panel: `Max | Mean | Sum` over Z.
4. 3D panel:
- `Centerlines`: sparse point cloud (`GLScatterPlotItem`)
- `Original/Labels/DT`: volume render (`GLVolumeItem`)

## Class Roles
- `MainWindow`: UI state, config sync, orchestrates segmentation/analysis, batch, and slice replay.
- `FiberAnalyzer`: core compute engine and cached data owner.
- `FiberposeModel`: model/device/AMP wrapper over `ResUnet` inference.
- `RidgeDetector`: ridge/junction extractor.
- `FiberFragment` + `FragmentEndpoint`: centerline fragment representation (and clustering graph semantics).
- `ImageBatch` / `FrameResult` / `BatchResult` / `BatchAnalyzer`: batch input and result transport.
- `Stack3DWindow`: visualization of stack-derived outputs.

## Notes / Constraints
- In GUI, strict checkbox is disabled outside ridge mode (`watershed` forces strict off).
- Clustering is optional and only changes fragment grouping and marker/overlay identities.
- `StackAnalyzer3D` exists (cross-slice label merge by IoU) but is not currently invoked by `MainWindow` flow.
