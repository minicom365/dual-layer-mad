# [EN] DualLayerMAD: Robust Noise Baseline Estimation Immune to Step-Edges and Outliers
# [KO] DualLayerMAD: 스텝 에지 및 이상치에 강건한 노이즈 배경 수준 측정 알고리즘
# Author: minicom365
# License: MIT

import math
from typing import List, Union, Sequence, Optional
import numpy as np

class DualLayerMAD:
    """
    [EN] Real-time streaming Dual-Layer Median Absolute Deviation (MAD) noise variance estimator.
    Immune to sudden step-edges and trajectory turning points via point-symmetry warmup and two-tier filtering.
    
    [KO] 실시간 스트리밍 2중 중앙값 절대편차(MAD) 기반 노이즈 분산/표준편차 추정기.
    점대칭 반사 웜업(Point-Symmetry Warmup)과 2단계 이상치 필터링을 통해 급격한 스텝 에지나 데이터 꺾임에 강건합니다.
    """
    __slots__ = ('window_size', 'history')

    def __init__(self, window_size: int = 32):
        if window_size < 4:
            raise ValueError("window_size must be at least 4.")
        self.window_size = int(window_size)
        self.history: List[float] = []

    def reset(self) -> None:
        """Reset the internal history buffer."""
        self.history.clear()

    def update(self, value: float) -> None:
        """
        [EN] Push a new scalar observation into the rolling window.
        [KO] 새로운 스칼라 관측값을 롤링 윈도우에 추가.
        """
        self.history.append(float(value))
        if len(self.history) > self.window_size:
            self.history.pop(0)

    def get_std(self) -> float:
        """
        [EN] Estimate robust standard deviation (sigma) of the noise background.
        [KO] 노이즈 배경의 강건한 표준편차(sigma = 1.4826 * pure_mad) 추정.
        """
        if not self.history:
            return 0.0

        # Point-symmetry warmup at t=0
        # (fill negative time window across (0, v0))
        if len(self.history) < self.window_size:
            v0 = self.history[0]
            # Reflected virtual samples in negative time: v_neg = 2*v0 - v
            virtual = [2.0 * v0 - x for x in reversed(self.history[1:])]
            sample_set = (virtual + self.history)[-self.window_size:]
        else:
            sample_set = self.history

        n = len(sample_set)
        if n < 2:
            return 0.0

        sorted_history = sorted(sample_set)
        med = sorted_history[n // 2]
        abs_devs = [abs(x - med) for x in sample_set]
        sorted_abs_devs = sorted(abs_devs)
        mad1 = sorted_abs_devs[n // 2]
        if mad1 == 0:
            return 0.0

        # Level 1: Step-edge & outlier rejection threshold
        threshold = 2.5 * mad1
        candidates = [
            x for x, dev in zip(sample_set, abs_devs) if dev <= threshold
        ]

        c_len = len(candidates)
        if c_len < 2:
            return mad1 * 1.4826

        # Level 2: Pure noise deviation estimation on filtered inliers
        sorted_candidates = sorted(candidates)
        pure_med = sorted_candidates[c_len // 2]

        pure_abs_devs = sorted([abs(x - pure_med) for x in candidates])
        pure_mad = pure_abs_devs[c_len // 2]

        return pure_mad * 1.4826

    def get_variance(self) -> float:
        """
        [EN] Estimate robust noise variance (sigma^2).
        [KO] 노이즈 배경의 강건한 분산(sigma^2) 추정.
        """
        s = self.get_std()
        return s * s

def get_dual_mad_noise(
    signal: Union[np.ndarray, Sequence[float]],
    window_size: int = 50,
    chunk_size: int = 10
) -> np.ndarray:
    """
    [EN] Batch Dual-Layer MAD noise curve estimator for 1D signals (Step-Edge Immune).
    Processes signal differences in optimized chunks to achieve high throughput.

    [KO] 1D 신호를 위한 배치 2중 MAD 노이즈 곡선 추정기 (스텝 에지 면역).
    신호의 1차 차분에 대해 2단계 MAD 필터링을 수행하여 이상치와 데이터 꺾임에 강건한 노이즈 레벨을 도출합니다.
    """
    arr = np.asarray(signal, dtype=np.float64)
    n = len(arr)
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    if n == 1:
        return np.zeros(1, dtype=np.float64)

    diff = np.abs(np.diff(arr, prepend=arr[0]))
    noise_curve = np.zeros(n, dtype=np.float64)

    for i in range(0, n, chunk_size):
        start = max(0, i - window_size)
        end = min(n, i + chunk_size)
        window = diff[start:end]

        if len(window) < 2:
            continue

        # Level 1: Step-edge removal (reject points far from median)
        med = np.median(window)
        mad = np.median(np.abs(window - med))
        mask = np.abs(window - med) < (2.5 * mad + 1e-9)
        candidates = window[mask]

        # Level 2: Estimate pure noise on surviving inlier residuals
        if len(candidates) > 2:
            pure_mad = np.median(np.abs(candidates - np.median(candidates)))
            val = pure_mad * 1.4826
            noise_curve[i:i + chunk_size] = val
        else:
            noise_curve[i:i + chunk_size] = mad * 1.4826

    return noise_curve
