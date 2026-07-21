# Fiberpose

**Fiberpose** is a research software package for segmentation and quantitative morphometry of fibrous microstructures. Instead of relying on threshold-based binarization, Fiberpose predicts a neural distance-transform field from microscopy images and uses it as a geometric representation for centerline extraction, instance segmentation, diameter measurement, orientation analysis, and pore statistics.

It is designed for fibrous materials where conventional thresholding becomes fragile: dense networks, weak contrast, uneven illumination, touching fibers, and overlapping structures.

> Status: research-preview software under active development. Validate outputs on your own data before using them for publication or production decisions.

---

## Key features

- **Neural distance-transform prediction** from raw microscopy images.
- **Fiber centerline extraction** using ridge detection on the predicted distance-transform field.
- **Instance segmentation** of fiber fragments using marker-controlled watershed or radial expansion modes.
- **Morphometric analysis** of fiber diameter, orientation, orientation index, porosity, and pore-size distributions.
- **Interactive desktop GUI** for loading images, inspecting intermediate maps, adjusting parameters, visualizing overlays, and exporting results.
- **Batch-oriented CLI** for reproducible processing of folders and image stacks.
- **Publication-oriented exports**: distance-transform maps, labels, overlays, diameter/orientation overlays, measurement tables, summaries, and manifests.

---

## Method overview

Fiberpose follows a simple geometric pipeline:

1. A neural network predicts a continuous distance-transform field from the input image.
2. Centerlines are extracted as ridges of this field.
3. Fiber instances are separated using centerline markers and distance-transform geometry.
4. Measurements are sampled along centerlines:
   - local radius is one pixel plus the mean distance-transform value sampled one pixel away on each side,
   - local orientation is estimated from the distance field gradients,
   - pore sizes are estimated from the distance transform of the void space.

This makes the distance-transform field the central representation connecting neural inference, segmentation, and quantitative analysis.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/vikba/Fiberpose.git
cd Fiberpose
```

### 2. Create a Python environment

Conda is recommended.

```bash
conda create -n fiberpose python=3.11 -y
conda activate fiberpose
```

### 3. Install runtime dependencies

For normal GUI/CLI use:

```bash
pip install -r requirements.txt
```

For training or development workflows:

```bash
pip install -r requirements-train.txt
```

### GPU note

Fiberpose runtime inference uses ONNX Runtime. GPU acceleration is available only if your installed ONNX Runtime build exposes `CUDAExecutionProvider`. If CUDA is unavailable, Fiberpose automatically falls back to CPU execution.

For training, install the PyTorch build that matches your CUDA version using the official PyTorch installation instructions, then install the remaining training dependencies.

---

## Pretrained model

By default, Fiberpose expects an ONNX checkpoint at:

```text
checkpoints/model_fiber_2_no_bg.onnx
```

If the model file is not included in the repository, download it from the project release page or place your own compatible ONNX model in the `checkpoints/` directory. You can also select a model in the GUI or specify it through `resources/config.json`.

---

## Running Fiberpose

`run.py` is the main launcher for both GUI and CLI modes.

### GUI mode

```bash
python run.py
```

or explicitly:

```bash
python run.py --gui
```

The GUI supports:

- single image, folder, and stack loading;
- raw image, distance-transform, Laplacian, centerline, segmentation, overlay, pore, diameter, and orientation views;
- model selection and CPU/GPU toggle;
- scalebar calibration;
- DT profile plotting;
- batch processing;
- export of masks, overlays, and CSV tables.

### CLI mode

Run segmentation on an image, folder, or stack:

```bash
python run.py --cli --input /path/to/image_or_folder_or_stack
```

Run the full measurement pipeline:

```bash
python run.py --cli --input /path/to/image_or_folder_or_stack --measurements
```

Write outputs to a selected directory:

```bash
python run.py --cli --input /path/to/image_or_folder_or_stack --output /path/to/output
```

Use a custom config:

```bash
python run.py --cli --input /path/to/image_or_folder_or_stack --config resources/config.json
```

Export only measurement CSV files:

```bash
python run.py --cli --input /path/to/image_or_folder_or_stack --measurements --save-kind csv
```

### CLI arguments

| Argument | Description |
|---|---|
| `--gui` | Force GUI mode. |
| `--cli` | Run command-line processing mode. |
| `--input <path>` | Input image, folder, or stack. Required in CLI mode. |
| `--config <path>` | Path to a JSON configuration file. Default: `resources/config.json`. |
| `--output <dir>` | Output directory for exported files. |
| `--measurements` | Run full measurement pipeline. Without this flag, CLI runs segmentation-only. |
| `--invert` | Invert image intensities before inference. Useful when fibers are dark on a bright background. |
| `--save-kind results\|csv` | Export full result artifacts or only CSV measurement tables. Default: `results`. |

### CLI exit codes

| Code | Meaning |
|---|---|
| `0` | Processing finished without failures. |
| `1` | Processing finished, but one or more frames failed during processing or export. |
| `2` | Invalid arguments, missing input/config/model, or startup/runtime error. |

---

## Supported inputs

Fiberpose can process common microscopy image formats, including:

```text
.tif, .tiff, .png, .jpg, .jpeg, .bmp, .webp
```

Folders are processed as batches. Multi-page TIFF files and image stacks are treated as stack inputs.

---

## Outputs

Depending on the selected export options, Fiberpose can save:

| Output | Description |
|---|---|
| `*_dt` | Predicted distance-transform map. |
| `*_labels` | Instance label visualization. |
| `*_overlay` | Segmentation overlay on the original image. |
| `*_diameters` | Diameter measurement overlay. |
| `*_orientations` | Orientation measurement overlay. |
| `*_outline` | Segmentation outline image. |
| `*_samples.csv` | Per-sample measurement table. |
| `*_summary.csv` | Batch-level summary table. |
| `manifest.json` | Export manifest with written files, skipped outputs, and errors. |

CSV exports include fiber diameter statistics, orientation metrics, porosity, and sample counts when the full measurement pipeline is enabled.

---

## Configuration

Default parameters are stored in:

```text
resources/config.json
```

The configuration controls model path, device preference, image scaling, ridge detection parameters, segmentation mode, centerline filtering, measurement spacing, and export options.

You can edit configuration values in the GUI through the configuration editor or pass a config file directly to the CLI:

```bash
python run.py --cli --input /path/to/data --config /path/to/config.json
```

---

## Project structure

Typical repository layout:

```text
Fiberpose/
├── run.py                     # Main GUI/CLI launcher
├── requirements.txt           # Runtime dependencies
├── requirements-train.txt     # Training/development dependencies
├── resources/
│   ├── config.json            # Default configuration
│   └── fiber_tool.ui          # Qt GUI layout
├── checkpoints/
│   └── model_fiber_2_no_bg.onnx  # Default ONNX model checkpoint
└── sources/
    ├── ui_main.py             # Main PyQt GUI
    ├── pipeline.py            # Pipeline orchestration
    ├── fiber_analyzer_2D.py   # 2D prediction, segmentation, and measurement logic
    ├── ridge_detector.py      # Ridge/centerline detector
    ├── ml.py                  # ONNX Runtime inference wrapper
    ├── domain.py              # Data models and configuration
    ├── io_export.py           # Export services
    ├── ui_support.py          # GUI helpers
    └── utils.py               # Numeric and image helper functions
```

---

## Training

Training is currently intended for developers and advanced users. Install the training requirements first:

```bash
pip install -r requirements-train.txt
```

A typical training dataset should contain paired microscopy images and corresponding distance-transform targets. The current Fiberpose model family is based on a residual U-Net-style architecture trained to predict distance-transform geometry rather than binary masks.

Training scripts and dataset-preparation utilities may change while the project is under active development.

---

## Troubleshooting

### Model file not found

Check that the default model exists:

```text
checkpoints/model_fiber_2_no_bg.onnx
```

or update the model path in `resources/config.json`.

### GPU is not used

Confirm that ONNX Runtime sees CUDA:

```python
import onnxruntime as ort
print(ort.get_available_providers())
```

If `CUDAExecutionProvider` is missing, install the correct ONNX Runtime GPU package for your platform.

### Segmentation is inverted or poor

Try the `--invert` CLI flag or the invert option in the GUI if your fibers are dark on a bright background.

### Measurements are missing

Run the full analysis pipeline, not segmentation-only:

```bash
python run.py --cli --input /path/to/data --measurements
```

---

## Citation

If you use Fiberpose in scientific work, please cite the associated manuscript/preprint when available.

```bibtex
@software{fiberpose,
  title  = {Fiberpose: A deep learning framework for segmentation and quantitative analysis of fibrous structures},
  author = {Balashov, Viktor},
  year   = {2026},
  url    = {https://github.com/vikba/Fiberpose}
}
```

---

## License

Fiberpose is free software licensed under the GNU General Public License,
version 3 only (`GPL-3.0-only`). See `LICENSE` for details.

---

## Disclaimer

Fiberpose is research software for microscopy image analysis. It is not a medical device and should not be used for clinical diagnosis or treatment decisions.
