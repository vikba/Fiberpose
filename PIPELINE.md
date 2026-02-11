# Fiberpose Pipeline

## Scope
- Documents runtime behavior in `sources/gui_mainwindow.py`, `sources/fiber_analyzer.py`, and `sources/batch_pipeline.py`.
- Covers segmentation, measurement, batch execution, and stack/3D visualization.

## Runtime Modes and Flags
- `segmentation.mode=watershed` calls `FiberAnalyzer.segment(mode="watershed", params={"threshold_u8": ...})`.
- `segmentation.mode=ridge` calls `FiberAnalyzer.segment(mode="ridge", params={"radius_thresh_px": ..., "dilate_extra_px": ..., "min_run_points": ...})`.
- `segmentation.mode=fastest_ascend` calls `FiberAnalyzer.segment(mode="fastest_ascend", params={"sigma": ..., "dist_thresh": ..., "flux_merge_thresh": ...})`.
- `strict_centerlines` affects centerline filtering in `process_centerlines`; GUI enables this checkbox only in ridge mode.
- `clustering.enabled` toggles fragment stitching (`stitch_and_cluster`) before marker/render generation.
- `scale factor` (`cb_scale`) rescales input for prediction/segmentation and rescales thresholds/measurements back to original units.
- `pores view` changes overlay/histogram to pore circles and pore diameter distribution.

## Core Runtime Data
- `ImageBatch`: ordered list of frames plus metadata (`filenames`, `z_indices`, `source_type`).
- `Frame`: per-slice state container (`orig_img`, prediction outputs, segmentation outputs, measurement outputs).
- `BatchResult`: `frame_results: list[Frame]`.
- `dt`: model output distance transform (`float32`).
- `dt_u8`, `dt_norm`, `gx`, `gy`, `laplacian`, `style_vec`: cached prediction products.
- `fragments`: `list[FiberFragment]` from centerline extraction.
- `markers`: `int32` seed image used by watershed.
- `labels_2d`: `uint16` segmentation labels.
- `samples_table`, `thickness_units`, `angles_deg`, `porosity`, `pores`, `pore_diams_units`: measurement outputs.

## Entry Flow
1. `MainWindow.open_source(...)` loads file/folder into `ImageBatch` via `ImageBatch.from_source/from_path/from_folder`.
2. `_set_image_batch(...)` sets current batch and selects slice 0.
3. User action calls `segment()` or `analyze_fibers()`.
4. If `_image_batch` exists and not already inside batch callback, GUI routes through `_run_batch_analysis(...)` (even single-frame cases).

## Batch Execution Flow
1. `_run_batch_analysis(...)` builds a `BatchAnalyzer`.
2. `BatchAnalyzer.analyze(...)` loops frames and calls `self._analyzer.reset_prediction_state()` per frame.
3. Per-frame callback in `_run_batch_analysis(...)`:
- `_reset_internal_state(preserve_original=False)`
- `current_frame.begin_frame(frame, frame_id, z_index)`
- call `segment()` or `analyze_fibers()` (with `_batch_processing_active=True`, so no recursive batch dispatch)
- capture result with `current_frame.snapshot(include_measurements=...)`
- optionally write artifacts via `_save_batch_outputs(...)`
4. Collected outputs become `BatchResult(frame_results=[...])`.
5. GUI stores `_batch_result` and rehydrates selected slice with `_apply_frame_result(...)`.

## Per-Frame Segmentation Flow
1. Validate `orig_img` and `analyzer`.
2. Clear downstream frame state (`clear_segmentation`, `clear_measurements`).
3. Resolve current scale:
- if scale changed: `analyzer.reset_prediction_state()` and `frame.clear_prediction()`
- build `orig_img_scaled` when needed (`cv2.resize`)
4. Sync UI config into `pipeline_config`.
5. If no cached DT prediction, run:
- `FiberAnalyzer.predict_dt(...)`
- internally: `_run_model()` -> `FiberposeModel.predict(...)`
- then smooth DT, compute gradients/laplacian, cache outputs
6. Always run `FiberAnalyzer.process_centerlines(...)`:
- `compute_centerlines(...)` ->
- `RidgeDetector.detect_ridges(...)` ->
- contour split/trim ->
- sliding window filter (optional strict checks) ->
- `FiberFragment` list + markers
- optional clustering merge and centerline overlay render
7. Copy analyzer prediction caches into `Frame` via `set_prediction_outputs(...)`.
8. Segment by mode through unified dispatch:
- `FiberAnalyzer.segment(mode=..., params=...)`
- watershed uses threshold_u8
- ridge uses radius threshold and optional dilation/min-run params (radius and dilation scale with active scale)
- fastest_ascend uses sigma/dist-threshold/flux-merge params
9. Build label colors + overlay and store via `set_segmentation_outputs(...)`.

## Per-Frame Measurement Flow
1. `analyze_fibers()` ensures valid segmentation at current scale; reruns `segment()` if missing or scale mismatch.
2. Sets `analyzer.scale_factor = frame.scale_factor`.
3. Calls `FiberAnalyzer.analyze(spacing_px_scaled)` returning:
- `hist_centers` (center/radius mapped back to original scale)
- `viz_points`
- `angles_deg`
4. Builds thickness arrays/sample table and orientation stats in GUI.
5. Computes porosity and pore sampling:
- `void_mask_and_pore_dt(thr_u8)`
- `sample_inscribed_circles(...)`
6. Stores outputs in `Frame` via `set_measurement_outputs(...)`.
7. Renders overlays and histograms:
- thickness histogram always
- angles histogram or pore size histogram depending on pores view

## Stack / 3D Visualization
1. Stack input normalization in `ImageBatch._normalize_stack(...)`:
- `HxW` -> single frame
- `HxWxC` -> single frame
- `ZxHxW` -> grayscale stack
- `ZxHxWxC` -> color stack
2. `Stack3DWindow` reads `BatchResult.frame_results`.
3. Chooses imaging stack from `Original | Labels | Centerlines | DT`.
4. Shows 2D projection (`Max | Mean | Sum`).
5. Shows 3D view:
- centerlines as sparse scatter
- other modes as volume render

## Class Roles
- `MainWindow` (`gui_mainwindow.py`): orchestration, UI state, view updates, CSV/image export.
- `FiberAnalyzer`: prediction, centerline extraction, segmentation, pore and measurement math.
- `BatchAnalyzer` (`batch_pipeline.py`): frame iteration orchestration.
- `StackAnalyzer_3D` (`batch_pipeline.py`): 3D label volume build, cross-slice label merge, and skeletonization.
- `ImageBatch` / `Frame` / `BatchResult` (`datastructures.py`): runtime data transport/state.
- `RidgeDetector`: ridge/junction extraction backend.
- `Stack3DWindow` (`gui_dialogs.py`): stack projection + 3D visualization.

## Notes
- `pipeline_config["ridge_detection"]["centerline_kwargs"]` defaults from `FiberAnalyzer._default_centerline_kwargs()` and is used by `segment()`.
