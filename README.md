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
git clone [https://github.com/vikba/Fiberpose.git](https://github.com/vikba/Fiberpose.git)
cd Fiberpose

### 2. Setup environment
```bash

# Create the environment with Python 3.11
conda create -n fiberpose python=3.11 -y

# Activate the environment
conda activate fiberpose

# Install dependencies
pip install -r requirements.txt
