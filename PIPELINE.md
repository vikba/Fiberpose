# Fiberpose Pipeline

## Scope
- Documents runtime behavior in `src/fiberpose/cli.py`, `src/fiberpose/gui/main_window.py`, `src/fiberpose/pipeline.py`, and `src/fiberpose/analysis.py`.
- Covers GUI and CLI startup, segmentation, measurement, and batch execution/export.

## Runtime Modes and Flags
- `segmentation_mode=watershed` makes `analysis.segment(...)` label the fiber pieces with the alignment-merge watershed (`watershed.segment_fibers(dt)`); `watershed_mask_threshold` (default `0.5`) is the DT value above which pixels are labelled.
- `segmentation_mode=radial_expansion` makes `analysis.segment(...)` rasterize the expanded ridge fragments.
- `merge_expansion_factor` controls overlap-based ridge-fragment merging (default `0.7`).
- `segmentation_expansion_factor` controls radial label expansion (default `1.0`).
- `strict_centerlines` affects the measurement-centerline filtering stage.
- `keep_segments_without_measurement_centerlines` keeps segmentation fragments even when no measurement centerline survives filtering.
- `scale factor` (`cb_scale`) rescales input for prediction/segmentation and rescales thresholds/measurements back to original units.
- `pores view` changes overlay/histogram to pore circles and pore diameter distribution.

## Core Runtime Data
- `ImageBatch`: ordered list of frames plus metadata (`filenames`, `z_indices`, `source_type`).
- `Frame`: per-slice state container (`orig_img`, prediction outputs, segmentation outputs, measurement outputs). It is the only owner of per-image state; `analysis` functions are stateless and `PipelineController` keeps only the loaded model.
- `BatchResult`: `frame_results: list[Frame]`.
- `dt`: model output distance transform (`float32`).
- `foreground_prob`: optional foreground probability (`float32`) from `nout == 2` checkpoints.
- `dt_u8`, `dt_normalized`, `gx`, `gy`, `laplacian`: cached prediction products.
- `fragments`: `list[FiberFragment]` that own a label; each carries its segmentation and measurement centerlines.
- `labels_2d`: `uint16` segmentation labels.
- `samples_table`, `thickness_units`, `angles_samples_deg`, `porosity`, `pores`, `pore_diams_units`: measurement outputs.

## Entry Flow
1. `fiberpose.cli.main()` (the `fiberpose` command) dispatches mode:
- GUI default (`fiberpose` or `fiberpose --gui`)
- CLI (`fiberpose --cli --input <path> ...`)
2. GUI path:
- `MainWindow.open_source(...)` loads input into `ImageBatch`.
- User action calls `segment()` or `analyze_fibers()`.
- If `_image_batch` exists and not already in callbacks, GUI routes through `_run_batch_analysis(...)`.
3. CLI path:
- `CliRunner` loads config and source via `ImageBatch.from_source(...)`.
- It instantiates `PipelineController(model_path, device)`, processes each frame, builds `BatchResult`, and exports.

## Batch Execution Flow
1. `_run_batch_analysis(...)` configures `BatchCallbacks` and calls `BatchService.run_batch(...)`.
2. `BatchService.run_batch(...)` loops frames and invokes the provided per-frame callback.
3. Per-frame callback in `_run_batch_analysis(...)`:
- create or hydrate `current_frame` with `Frame.empty(...)` or a `snapshot()` of the cached frame result
- run `_run_current_frame_pipeline(...)`, which calls `ensure_segmentation` or `ensure_measurements` (`_is_batch_processing=True` keeps `segment()` / `analyze_fibers()` from re-dispatching a batch)
- capture result with `current_frame.snapshot(frame_id=..., z_index=...)`
4. Collected outputs become `BatchResult(frame_results=[...])`.
5. GUI stores `_batch_result` and rehydrates selected slice with `_apply_frame_result(...)`.
6. CLI uses the same controller stages (`ensure_segmentation` / `ensure_measurements`) and writes exports via `BatchResult.save_all(...)`.

## Per-Frame Segmentation Flow
1. `ensure_prediction`: if the frame has no prediction for the current signature, resolve the scaled processing image and run `analysis.predict(model, image)`:
- pads to the model stride, runs `OnnxFiberposeModel.predict(...)` -> `(dt, foreground_prob)`; `nout != 2` uses the last channel as DT and has no foreground
- smooths DT, computes gradients/laplacian; stored via `Frame.set_prediction_outputs(...)`
2. If the frame's segmentation signature (prediction signature + `segmentation_params(config)`) matches, stop.
3. Otherwise `clear_segmentation()` and run `analysis.segment(dt, dt_u8, gx, gy, foreground_prob, **segmentation_params)`:
- `RidgeDetector.detect_ridges(...)` (raw ridge lines)
- `_cut_polylines_by_foreground(...)` cuts each ridge where the foreground probability crosses `min_segment_foreground_mean`, so a ridge running from a fiber into a rejected object is split there (no foreground: nothing is cut)
- `_group_detected_polylines_into_segmentation_fragments(...)` merges ridges whose `merge_expansion_factor * DT` disks overlap; pieces inside and outside the foreground are never merged
- fragments outside the foreground are not measured or returned
- `_filter_measurement_centerlines(...)` derives measurement centerlines
- `_filter_fragments_by_grad_quality(...)` drops fragments with unreliable DT gradients
- watershed (`watershed.segment_fibers`) uses the DT alone: it floods `dt > watershed_mask_threshold` from the DT maxima into basins, then merges two touching basins when the vector joining their seed points runs along the fiber axis (DT structure tensor) on their collision line and at both seeds, and the line spans a fiber; regions under 40 px are dropped. Each fragment then takes the label under most of its centerline points. radial_expansion rasterizes the expanded fragments
- foreground, when available, prunes labels; then the labels that hold no remaining fragment are removed; only fragments that keep a label are returned
4. Store the labels and fragments via `set_segmentation_outputs(...)`. The pipeline draws no images.

## Per-Frame Measurement Flow
1. GUI `analyze_fibers()` or CLI measurement mode ensures prediction + segmentation are ready.
2. Calls `analysis.sample_centerlines(dt, gx, gy, frame.fragments, spacing_px, frame.scale_factor)` returning measurement and drawing samples.
3. Builds thickness arrays/sample table and orientation stats in `PipelineController`.
4. Computes porosity and pore sampling:
- `analysis.void_mask_and_pore_dt(dt_u8, thr_u8)`
- `analysis.sample_inscribed_circles(...)`
5. Stores outputs in `Frame` via `set_measurement_outputs(...)`.
6. GUI shows the selected view and histograms:
- thickness histogram always
- angles histogram or pore size histogram depending on pores view

## Stack Input
Stack input normalization in `ImageBatch._normalize_stack(...)`:
- `HxW` -> single frame
- `HxWxC` -> single frame
- `ZxHxW` -> grayscale stack
- `ZxHxWxC` -> color stack

## Class Roles
- `cli.py`: top-level GUI/CLI dispatch and argument parsing (`fiberpose` command).
- `analyze()` (`api.py`): public single-image entry point wrapping `PipelineController`.
- `MainWindow` (`gui/main_window.py`): GUI orchestration, UI state, view updates, and user-triggered batch runs.
- `CliRunner` (`cli_runner.py`): CLI orchestration for batch processing/export.
- `PipelineController` (`pipeline.py`): shared processing engine; owns the lazily loaded model and caches each stage on `Frame` by signature.
- `BatchService` (`pipeline.py`): frame iteration orchestration using callbacks.
- `analysis.py`: stateless prediction, centerline extraction, segmentation, pore and measurement functions.
- `render(frame, view)` (`render.py`): builds every displayed or exported image from frame data on demand and caches it in `Frame.views` until a stage is recomputed.
- `ImageBatch` / `Frame` / `BatchResult` (`domain.py`): runtime data transport/state.
- `RidgeDetector`: ridge/junction extraction backend.
