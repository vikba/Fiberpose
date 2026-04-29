# Fiberpose Pipeline

## Scope
- Documents runtime behavior in `sources/gui_mainwindow.py`, `sources/fiber_analyzer_2D.py`, and `sources/batch_pipeline.py`.
- Covers segmentation, measurement, batch execution, and stack/3D visualization.

## Runtime Modes and Flags
- `segmentation.mode=watershed` calls `FiberAnalyzer.segment(mode="watershed", params={"threshold_u8": ...})`.
- `segmentation.mode=radial_expansion` calls `FiberAnalyzer.segment(mode="radial_expansion", params={"expansion_factor": ...})`.
- `radial_expansion_factor` controls ridge expansion as a fraction of local DT (default `0.7`).
- `strict_centerlines` affects the measurement-centerline filtering stage.
- `keep_segments_without_measurement_centerlines` keeps segmentation fragments even when no measurement centerline survives filtering.
- `clustering.enabled` toggles optional centerline stitching links for debug visualization.
- `scale factor` (`cb_scale`) rescales input for prediction/segmentation and rescales thresholds/measurements back to original units.
- `pores view` changes overlay/histogram to pore circles and pore diameter distribution.

## Core Runtime Data
- `ImageBatch`: ordered list of frames plus metadata (`filenames`, `z_indices`, `source_type`).
- `Frame`: per-slice state container (`orig_img`, prediction outputs, segmentation outputs, measurement outputs).
- `BatchResult`: `frame_results: list[Frame]`.
- `dt`: model output distance transform (`float32`).
- `dt_u8`, `dt_normalized`, `gx`, `gy`, `laplacian`: cached prediction products.
- `segmentation_centerlines`: `list[FiberFragment]` built directly from raw ridges.
- `measurement_centerlines`: `list[FiberFragment]` filtered from `segmentation_centerlines`.
- `markers`: `int32` marker image rendered from `segmentation_centerlines` and used by watershed.
- `labels_2d`: `uint16` segmentation labels.
- `samples_table`, `thickness_units`, `angles_samples_deg`, `porosity`, `pores`, `pore_diams_units`: measurement outputs.

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
- call `segment()` or `analyze_fibers()` (with `_is_batch_processing=True`, so no recursive batch dispatch)
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
- internally: `_run_model()` -> `OnnxFiberposeModel.predict(...)`
- then smooth DT, compute gradients/laplacian, cache outputs
6. Always run `FiberAnalyzer.process_centerlines(...)`:
- `RidgeDetector.detect_ridges(...)` (raw ridge lines)
- `_ridge_lines_to_segmentation_fragments(...)`:
  - sample local DT radius on raw ridges
  - expand ridges by `radial_expansion_factor * DT`
  - merge overlapping expanded regions and assign stable label ids
- `_filter_measurement_centerlines(...)` derives measurement centerlines from segmentation centerlines
- render segmentation/measurement/comparison centerline overlays
7. Copy analyzer prediction caches into `Frame` via `set_prediction_outputs(...)`.
8. Segment by mode through unified dispatch:
- `FiberAnalyzer.segment(mode=..., params=...)`
- watershed uses threshold_u8
- radial_expansion uses the precomputed expanded ridge labels
9. Build label colors + overlay and store via `set_segmentation_outputs(...)`.

## Per-Frame Measurement Flow
1. `analyze_fibers()` ensures prediction + measurement centerlines are ready.
2. Sets `analyzer.scale_factor = frame.scale_factor`.
3. Calls `FiberAnalyzer.analyze(spacing_px_scaled)` returning:
- `radius_samples` (center/radius mapped back to original scale)
- `angles_samples_deg`
- `overlay_samples`
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
3. Chooses imaging stack from `Original | DT | Semantic segmentation`.
4. Shows 2D projection (`Max | Mean | Sum`).
5. Shows 3D view:
- semantic segmentation as volume render
- optional semantic centerline overlay
- optional semantic junction overlay

## Class Roles
- `MainWindow` (`gui_mainwindow.py`): orchestration, UI state, view updates, CSV/image export.
- `FiberAnalyzer`: prediction, centerline extraction, segmentation, pore and measurement math.
- `BatchAnalyzer` (`batch_pipeline.py`): frame iteration orchestration.
- `StackAnalyzer_3D` (`batch_pipeline.py`): semantic 3D volume merge from per-slice labels + native-grid 3D skeletonization.
- `ImageBatch` / `Frame` / `BatchResult` (`datastructures.py`): runtime data transport/state.
- `RidgeDetector`: ridge/junction extraction backend.
- `Stack3DWindow` (`gui_dialogs.py`): stack projection + 3D visualization.

## Notes
- `centerline_kwargs` defaults from `FiberAnalyzer_2D._default_centerline_kwargs()` and are used by `process_centerlines()`.
