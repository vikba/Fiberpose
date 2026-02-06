1) Glossary (Data Products)
- `image`: (H,W) or (H,W,C) float/uint8, input micrograph intensity
- `dt`: (H,W) float32, predicted distance transform in [0..255]
- `dt_u8`: (H,W) uint8, clipped `dt` in [0..255]
- `gx, gy`: (H,W) float32, gradients of smoothed `dt`
- `laplacian`: (H,W) float32, Laplacian of smoothed `dt`
- `ridge_lines`: list[RidgeLine], polylines in (row,col)
- `junctions`: list[Junction], ridge intersections
- `fragments`: list[FiberFragment], each with `points_xy` (N,2), `radii` (N,), `parent_fiber_id`
- `markers`: (H,W) int32, watershed seeds labeled by fiber id
- `labels`: (H,W) uint16, segmentation labels
- `centerlines_rgb`: (H,W,3) uint8, DT background + centerlines/viz
- `void_mask`: (H,W) uint8 {0,1}, pore region mask
- `pore_dt`: (H,W) float32, distance transform inside voids
- `fiber_metrics`: {hist_centers, viz_points, angles_deg} lists, per-sample measurements

2) Algorithm Cards
### DT Inference (FiberposeModel.predict)
Consumes: `image`
Produces: `dt` (and optional style_vec)
Assumes: checkpoint exists; input normalized per-image min/max
Steps:
1. Normalize channels and scale to [0,1]
2. Run ResUNet, extract channel 2 as `dt`
Params: `ckpt_path`, `nbase`, `nout`, `amp`, `device`
Failure: missing/invalid checkpoint or unsupported input shape

### Predict DT + Gradients (FiberAnalyzer.predict_dt)
Consumes: `image`
Produces: `dt`, `dt_u8`, `gx`, `gy`, `laplacian`
Assumes: `dt` represents distances in 0..255 range
Steps:
1. Run DT inference, smooth, compute gradients and Laplacian
2. Cache `dt`/grads and reset downstream state
Params: `smooth_sigma=1.5`, `grad_sigma=1.0`, `max_angle_deg`
Failure: image is None or model inference error

### Centerline Extraction (FiberAnalyzer.compute_centerlines)
Consumes: `dt_u8`, `dt`, (`gx`,`gy` if strict)
Produces: `fragments`, `markers`, `centerlines_rgb`
Assumes: ridge detector expects uint8 image; `dt` aligned with `dt_u8`
Steps:
1. Detect ridges + junctions, split/trim polylines
2. Sliding-window filter by intensity/angle (+ strict gradient checks)
3. Sample `radii` from `dt`, render `markers`
Params: `sigma`, `low`, `high`, `trim`, `win_len`, `std_thresh`, `max_angle_deg`, `strict`
Failure: empty ridges ⇒ no markers/segments

### Fiber Clustering (stitch_and_cluster)
Consumes: `fragments`
Produces: `fragments` with updated `parent_fiber_id`, connections
Assumes: endpoints have direction + radius; distances in px
Steps:
1. Propose endpoint pairs by proximity and facing cost
2. Mutual best-match connect, union-find to set parent ids
Params: `d_max`, `threshold`, `max_neighbors`, `length_weight`, `radius_weight`
Failure: clustering errors ⇒ no connections, ids unchanged

### Watershed Segmentation (FiberAnalyzer.watershed_dt_segmentation)
Consumes: `dt_u8`, `markers`
Produces: `labels`
Assumes: markers >0 and threshold in [0..255]
Steps:
1. Build fiber mask from `dt_u8 >= thr`
2. Watershed on `1 - dt_norm` within mask
Params: `thr_u8`, `connectivity=8`
Failure: empty fiber mask ⇒ all-zero labels

### Strict Ridge Segmentation (FiberAnalyzer.segment_strict_ridge)
Consumes: `fragments` (+ `radii`), `dt_u8`
Produces: `labels`
Assumes: radii correspond to DT in pixels
Steps:
1. Keep runs where `radii >= radius_thresh_px`
2. Draw circles along runs (radius + extra), label by parent id
Params: `radius_thresh_px`, `dilate_extra_px`, `min_run_points`
Failure: no valid runs ⇒ all-zero labels

### Fiber Analysis (FiberAnalyzer.analyze)
Consumes: `fragments` or `centerline_segments`, `dt`, `gx`, `gy`
Produces: `fiber_metrics`
Assumes: gradients align with `dt`; scale_factor >0
Steps:
1. Resample centerlines, compute normals from `dt`/grads
2. Sample radius and angle per point, aggregate lists
Params: `spacing_px`
Failure: missing `dt`/grads ⇒ empty metrics

### Pore Sampling (FiberAnalyzer.void_mask_and_pore_dt / FiberAnalyzer.sample_inscribed_circles)
Consumes: `dt_u8`
Produces: `void_mask`, `pore_dt`, `circles`
Assumes: pores are complement of fiber mask
Steps:
1. Compute void mask from `dt_u8 < thr`, DT inside voids
2. Pick local maxima and keep non-overlapping inscribed circles
Params: `thr_u8`, `min_radius_px`, `min_peak_px`, `peak_footprint`
Failure: empty void mask ⇒ no circles

3) Relations Map
- DT Inference -> Predict DT + Gradients [sequential: `dt`]
- Predict DT + Gradients -> Centerline Extraction [sequential: `dt_u8`, `dt`, `gx`, `gy`]
- Centerline Extraction => Fiber Clustering [refinement: `parent_fiber_id` + connections]
- Centerline Extraction -> Watershed Segmentation [sequential: `markers`, `dt_u8`]
- Centerline Extraction -> Strict Ridge Segmentation [sequential: `fragments`, `dt_u8`]
- Watershed Segmentation || Fiber Analysis [parallel: `dt`/`grads` shared]
- Strict Ridge Segmentation replaces Watershed Segmentation when `radius_thresh_px` is set

4) Pipeline DAG (dataflow)
Image Input --(`image`)--> DT Inference  
DT Inference --(`dt`)--> Predict DT + Gradients  
Predict DT + Gradients --(`dt_u8`, `dt`, `gx`, `gy`)--> Centerline Extraction  
Centerline Extraction --(`fragments`)--> Fiber Clustering  
Centerline Extraction --(`markers`, `dt_u8`)--> Watershed Segmentation  
Centerline Extraction --(`fragments`, `dt_u8`)--> Strict Ridge Segmentation  
Predict DT + Gradients --(`dt`, `gx`, `gy`, `fragments`)--> Fiber Analysis  
Predict DT + Gradients --(`dt_u8`)--> Pore Sampling  

Mode switch:
Watershed Segmentation OR Strict Ridge Segmentation (if `radius_thresh_px` set)

Cache: `dt`/`gx`/`gy` invalidated by `predict_dt`, checkpoint change, or device change  
Cache: `markers` invalidated by `enable_fiber_clustering` or re-run of centerline extraction

5) Minimal End-to-End Summary (≤ 6 lines)
- DT Inference outputs `dt`
- Predict DT + Gradients outputs `dt_u8`, `gx`, `gy`, `laplacian`
- Centerline Extraction outputs `fragments` + `markers`
- Watershed Segmentation outputs `labels` (or Strict Ridge Segmentation in ridge mode)
- Fiber Analysis outputs `fiber_metrics`
- Pore Sampling outputs `void_mask`, `pore_dt`, `circles`