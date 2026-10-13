# DualLayerMAD (한국어 가이드)

> **스텝 에지 및 이상치에 강건한 2중 MAD 기반 노이즈 배경 수준 측정 알고리즘**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

[English](README.md) | [한국어](README_KO.md)

---

## 📌 개요 및 고안 배경

신호 처리, 모션 트래킹, 칼만 필터, 센서 융합 시스템에서 배경 노이즈 분산($\sigma^2$)을 실시간으로 추정할 때 기존의 이동 표준편차(Moving Standard Deviation)나 단순 1차 MAD(Median Absolute Deviation)는 다음과 같은 치명적인 결함이 있습니다:
- 데이터가 급격히 꺾이거나(급가속/스텝 점프) 센서 튀김이 발생할 때, **실제 노이즈가 아닌 신호의 기하학적 에지를 노이즈로 착각하여 분산이 수십~수백 배 폭증**합니다.

**DualLayerMAD**는 minicom365가 고안한 2단계 계층적 MAD 필터링 알고리즘으로 다음 3가지 핵심 기법을 결합하여 이 문제를 완벽히 해결합니다:
1. **점대칭 반사 웜업 (Point-Symmetry Warmup)**: $t=0$ 초기화 시점 데이터 부족 문제를 음의 시간 축 가상 반사 샘플($v_{\text{neg}} = 2v_0 - v$)로 채워 지연 없이 즉각 수렴.
2. **1단계 이상치/스텝 에지 컷오프**: 1차 MAD의 $2.5 \times \text{MAD}_1$ 임계값을 초과하는 급격한 점프 성분을 에지로 판정하여 1차 제거.
3. **2단계 순수 노이즈 잔차 추정**: 에지가 제거된 순수 인라이어(Inlier) 잔차에 대해 2차 MAD를 재연산하여 순수 가우시안 정규분포 분산($\sigma = 1.4826 \times \text{MAD}_{\text{pure}}$) 도출.

---

## 🚀 설치 방법

```bash
pip install dual-layer-mad
```

---

## 💻 사용 예제

### 1. 실시간 스트리밍 모드 (`DualLayerMAD`)
```python
from dual_layer_mad import DualLayerMAD

# 윈도우 크기 32 샘플로 추정기 생성
estimator = DualLayerMAD(window_size=32)

# 센서 스트림 실시간 주입
for new_sample in sensor_stream:
    estimator.update(new_sample)
    
    # 노이즈 표준편차(sigma) 및 분산(sigma^2) 획득
    noise_sigma = estimator.get_std()
    noise_var = estimator.get_variance()
```

### 2. 1D 배열 배치 처리 모드 (`get_dual_mad_noise`)
```python
import numpy as np
from dual_layer_mad import get_dual_mad_noise

# 1D 신호 배열 준비
signal = np.array([...])

# 1차 차분 신호 기반 스텝 에지 면역 노이즈 곡선 도출
noise_curve = get_dual_mad_noise(signal, window_size=50, chunk_size=10)
```

---

## 📄 라이선스

MIT License. Designed by minicom365.
