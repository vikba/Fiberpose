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

