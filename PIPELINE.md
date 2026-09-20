# Fiberpose Pipeline

## Scope
- Documents runtime behavior in `run.py`, `sources/ui_main.py`, `sources/pipeline.py`, and `sources/fiber_analyzer_2D.py`.
- Covers GUI and CLI startup, segmentation, measurement, and batch execution/export.

## Runtime Modes and Flags
- `segmentation.mode=watershed` calls `FiberAnalyzer_2D.segment(mode="watershed", params={"gradient_threshold": ...})`.
- `segmentation.mode=radial_expansion` calls `FiberAnalyzer_2D.segment(mode="radial_expansion", params={"expansion_factor": ...})`.
- `merge_expansion_factor` controls overlap-based ridge-fragment merging (default `0.7`).
- `segmentation_expansion_factor` controls radial label expansion (default `1.0`).
- `strict_centerlines` affects the measurement-centerline filtering stage.
- `keep_segments_without_measurement_centerlines` keeps segmentation fragments even when no measurement centerline survives filtering.
- `scale factor` (`cb_scale`) rescales input for prediction/segmentation and rescales thresholds/measurements back to original units.
- `pores view` changes overlay/histogram to pore circles and pore diameter distribution.

## Core Runtime Data
- `ImageBatch`: ordered list of frames plus metadata (`filenames`, `z_indices`, `source_type`).
- `Frame`: per-slice state container (`orig_img`, prediction outputs, segmentation outputs, measurement outputs).
- `BatchResult`: `frame_results: list[Frame]`.
- `dt`: model output distance transform (`float32`).
- `foreground_prob`: optional foreground probability (`float32`) from `nout == 2` checkpoints.
- `dt_u8`, `dt_normalized`, `gx`, `gy`, `laplacian`: cached prediction products.
- `segmentation_centerlines`: `list[FiberFragment]` built directly from raw ridges.
- `measurement_centerlines`: `list[FiberFragment]` filtered from `segmentation_centerlines`.
- `markers`: `int32` marker image rendered from `segmentation_centerlines` and used by watershed.
- `labels_2d`: `uint16` segmentation labels.
- `samples_table`, `thickness_units`, `angles_samples_deg`, `porosity`, `pores`, `pore_diams_units`: measurement outputs.

## Entry Flow
1. `run.py` dispatches mode:
- GUI default (`python run.py` or `python run.py --gui`)
- CLI (`python run.py --cli --input <path> ...`)
2. GUI path:
- `MainWindow.open_source(...)` loads input into `ImageBatch`.
- User action calls `segment()` or `analyze_fibers()`.
- If `_image_batch` exists and not already in callbacks, GUI routes through `_run_batch_analysis(...)`.
3. CLI path:
- `CliRunner` loads config and source via `ImageBatch.from_source(...)`.
- It instantiates analyzer + `PipelineController`, processes each frame, builds `BatchResult`, and exports.

## Batch Execution Flow
1. `_run_batch_analysis(...)` configures `BatchCallbacks` and calls `BatchService.run_batch(...)`.
2. `BatchService.run_batch(...)` loops frames and invokes the provided per-frame callback.
3. Per-frame callback in `_run_batch_analysis(...)`:
- `_reset_internal_state(preserve_original=False)`
- create or hydrate `current_frame` with `Frame.empty(...)` or `Frame.snapshot(...)`
- call `segment()` or `analyze_fibers()` (with `_is_batch_processing=True`, so no recursive batch dispatch)
- capture result with `current_frame.snapshot(include_measurements=...)`
4. Collected outputs become `BatchResult(frame_results=[...])`.
5. GUI stores `_batch_result` and rehydrates selected slice with `_apply_frame_result(...)`.
6. CLI uses the same controller stages (`ensure_segmentation` / `ensure_measurements`) and writes exports via `BatchResult.save_all(...)`.

## Per-Frame Segmentation Flow
1. Validate `orig_img` and `analyzer`.
2. Clear downstream frame state (`clear_segmentation`, `clear_measurements`).
3. Resolve current scale and processing image in `PipelineController`.
4. Read runtime config from `AppConfig` (+ `invert_input` flag from GUI checkbox or CLI argument).
5. If no cached DT prediction, run:
- `FiberAnalyzer_2D.predict_dt(...)`
- internally: `_run_model()` -> `OnnxFiberposeModel.predict(...)`
- `predict(...)` returns `(dt, foreground_prob)`; `nout != 2` uses the last channel as DT and has no foreground
- then smooth DT, compute gradients/laplacian, cache outputs
6. Always run `FiberAnalyzer_2D.process_centerlines(...)`:
- `RidgeDetector.detect_ridges(...)` (raw ridge lines)
- `_group_detected_polylines_into_segmentation_fragments(...)`:
  - sample local DT radius on raw ridges
  - merge locally expanded ridge regions using `merge_expansion_factor * DT`
  - merge overlapping expanded regions and assign stable label ids
- `_filter_measurement_centerlines(...)` derives measurement centerlines from segmentation centerlines
- render segmentation/measurement/comparison centerline overlays
7. Copy analyzer prediction caches into `Frame` via `set_prediction_outputs(...)`.
8. Segment by mode through unified dispatch:
- `FiberAnalyzer_2D.segment(mode=..., params=...)`
- watershed thresholds the cached DT gradient magnitude, unions two 3x3 dilations of detected centerline markers, then applies erode -> dilate to the combined mask
- radial_expansion uses the precomputed expanded ridge labels
- foreground, when available, prunes labels after pure-DT segmentation
9. Build label colors + overlay and store via `set_segmentation_outputs(...)`.

## Per-Frame Measurement Flow
1. GUI `analyze_fibers()` or CLI measurement mode ensures prediction + measurement centerlines are ready.
2. Sets `analyzer.scale_factor = frame.scale_factor`.
3. Calls `FiberAnalyzer_2D.analyze(spacing_px=...)` returning measurement and drawing samples.
4. Builds thickness arrays/sample table and orientation stats in `PipelineController`.
5. Computes porosity and pore sampling:
- `void_mask_and_pore_dt(thr_u8)`
- `sample_inscribed_circles(...)`
6. Stores outputs in `Frame` via `set_measurement_outputs(...)`.
7. GUI renders overlays/histograms for display:
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
- `run.py`: top-level GUI/CLI dispatch and argument parsing.
- `MainWindow` (`ui_main.py`): GUI orchestration, UI state, view updates, and user-triggered batch runs.
- `CliRunner` (`cli_runner.py`): CLI orchestration for batch processing/export.
- `PipelineController` (`pipeline.py`): shared processing engine for prediction, centerlines, segmentation, and measurements.
- `BatchService` (`pipeline.py`): frame iteration orchestration using callbacks.
- `FiberAnalyzer_2D`: prediction, centerline extraction, segmentation, pore and measurement math.
- `ImageBatch` / `Frame` / `BatchResult` (`domain.py`): runtime data transport/state.
- `RidgeDetector`: ridge/junction extraction backend.

## Notes
- `centerline_kwargs` defaults from `FiberAnalyzer_2D._default_centerline_kwargs()` and are used by `process_centerlines()`.
