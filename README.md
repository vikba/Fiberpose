# Fiberpose

Fiberpose is a specialized deep learning tool designed to predict the **Distance Transform (DT)** from original fiber images using a **ResUNet** architecture. It performs high-precision segmentation via a two-mask watershed approach and calculates fiber thickness using tangent-normal and DiameterJ-style metrics.

---

## 🚀 Features

* **ResUNet Backbone:** Leverages residual connections for high-fidelity distance transform prediction.
* **Dual-Mask Watershed:** Optimized segmentation to handle overlapping or touching fibers.
* **Advanced Metrology:** Includes tangent-normal measurement and metrics inspired by DiameterJ for accurate diameter estimation.

---

## 🛠️ Installation

Ensure you have [Conda](https://docs.conda.io/en/latest/miniconda.html) installed on your system before proceeding.

### 1. Clone the Repository
```bash
git clone https://github.com/vikba/Fiberpose.git
cd Fiberpose
```

### 2. Setup environment
```bash
# Create the environment with Python 3.11
conda create -n fiberpose python=3.11 -y

# Activate the environment
conda activate fiberpose
```

### 3. Install dependencies by use case
```bash
# CPU inference/runtime dependencies (end-user app)
pip install -r requirements.txt

# Training dependencies (adds PyTorch + tooling on top of runtime deps)
pip install -r requirements-train.txt
```

### 4. GPU training note
`requirements-train.txt` keeps PyTorch package names generic. For CUDA-specific wheels, install the matching `torch`/`torchvision` build for your platform using the official PyTorch instructions, then install the rest of the training stack.

---

## ▶️ Running Fiberpose

`run.py` is now the single launcher for both GUI and CLI modes.

### GUI (default)
```bash
python run.py
```

You can also force GUI mode explicitly:
```bash
python run.py --gui
```

### CLI (batch-oriented)
CLI mode is enabled with `--cli` and requires `--input`.

```bash
python run.py --cli --input /path/to/folder_or_stack
```

Compute full measurements instead of segmentation-only:
```bash
python run.py --cli --input /path/to/folder_or_stack --measurements
```

Write outputs to a specific directory:
```bash
python run.py --cli --input /path/to/folder_or_stack --output /path/to/out
```

Use a custom config and CSV-focused export:
```bash
python run.py --cli --input /path/to/folder_or_stack --config resources/config.json --save-kind csv
```

### CLI arguments
- `--cli`: run command-line processing mode.
- `--input <path>`: input source path (required in CLI mode).
- `--config <path>`: config JSON path (defaults to `resources/config.json`).
- `--output <dir>`: output directory for exported artifacts.
- `--measurements`: run full measurement pipeline (default is segmentation-only).
- `--invert`: invert input intensities before inference.
- `--save-kind results|csv`: export preset kind (`results` default).

### Exit behavior (CLI)
- `0`: run finished without failures.
- `1`: run completed but had processing/export failures.
- `2`: invalid arguments or startup/runtime dependency/config/input errors.

