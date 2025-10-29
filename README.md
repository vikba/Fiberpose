# Fiberpose — CPnet DT + Watershed + DiameterJ

Predict **distance transform (DT)** from the original image using **CPnet**, then segment with a two-mask watershed and measure fiber thickness with tangent-normal and DiameterJ-style metrics.

## Install

```bash
conda create -n fiberpose python=3.11 -y
conda activate fiberpose
pip install -r requirements.txt
