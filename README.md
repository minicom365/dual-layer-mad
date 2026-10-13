# DualLayerMAD

> **Robust Noise Baseline Estimation Immune to Step-Edges and Trajectory Turning Points**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

[English](README.md) | [한국어](README_KO.md)

---

## 📌 Overview

Conventional moving standard deviation and single-layer Median Absolute Deviation (MAD) estimators severely overestimate the noise floor whenever a signal undergoes sudden step transitions (e.g. tracking re-acquisitions, sensor glitches, state changes).

**DualLayerMAD** provides a mathematically robust two-tier noise estimation pipeline:
1. **Point-Symmetry Warmup**: Reflects virtual negative-time samples across $(0, v_0)$ to eliminate initialization lag at $t=0$.
2. **Level 1 Rejection**: Filters out large step-edges and transient spikes exceeding $2.5 \times \text{MAD}_1$.
3. **Level 2 Pure MAD**: Evaluates the uncontaminated noise residual on surviving inliers:
   $$\sigma = 1.4826 \times \text{MAD}_{\text{pure}}$$

---

## 🚀 Installation

```bash
pip install dual-layer-mad
```

---

## 💻 Quick Start

### 1. Real-time Streaming (`DualLayerMAD`)

```python
from dual_layer_mad import DualLayerMAD

estimator = DualLayerMAD(window_size=32)

for sample in sensor_stream:
    estimator.update(sample)
    noise_sigma = estimator.get_std()
    noise_variance = estimator.get_variance()
```

### 2. Batch Signal Processing (`get_dual_mad_noise`)

```python
import numpy as np
from dual_layer_mad import get_dual_mad_noise

signal = np.array([...])
noise_curve = get_dual_mad_noise(signal, window_size=50, chunk_size=10)
```

---

## 📄 License

MIT License. Designed by minicom365.
