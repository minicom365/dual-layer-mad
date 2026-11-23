# [EN] DualLayerMAD: Robust Noise Baseline Estimation Immune to Step-Edges and Outliers
# [KO] DualLayerMAD: 스텝 에지 및 이상치에 강건한 노이즈 배경 수준 측정 알고리즘
# [Arch] Implements 1D Dual-Layer MAD with non-zero MAD1 fallback and 4-step conditional truncated-normal fixed-point compensation (excluding static 1.4826/1.6558 scales and pre-MAD folded differences).
# Author: minicom365
# License: MIT

import math
from typing import List, Union, Sequence
import numpy as np

_SQRT2 = 1.4142135623730951
_INV_SQRT2PI = 0.3989422804014327


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / _SQRT2))


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) * _INV_SQRT2PI


def _norm_ppf(p: float) -> float:
    z = 0.0
    for _ in range(6):
        err = _norm_cdf(z) - p
        pdf = _norm_pdf(z)
        z -= err / (pdf + 0.5 * z * err)
    return z


_Z75 = _norm_ppf(0.75)
_C_GAUSS = 1.0 / _Z75
_C0_TAU25 = 2.5 * _Z75
_Q0_TAU25 = _norm_ppf(0.25 + 0.5 * _norm_cdf(_C0_TAU25))


def _median_sorted(s: List[float]) -> float:
    m = len(s)
    mid = m >> 1
    if m & 1:
        return s[mid]
    return 0.5 * (s[mid - 1] + s[mid])


def _truncated_mad_fixed_point(mad2: float, r1: float) -> float:
    sig = mad2 / _Q0_TAU25
    r1_floor = 1e-12 * r1
    for _ in range(4):
        c_eff = max(_C0_TAU25, r1 / max(sig, r1_floor))
        sig = mad2 / _norm_ppf(0.25 + 0.5 * _norm_cdf(c_eff))
    return sig


def _dual_mad_std_from_list(sample_set: List[float]) -> float:
    n = len(sample_set)
    if n < 2:
        return 0.0

    sorted_history = sorted(sample_set)
    rng = sorted_history[-1] - sorted_history[0]
    if rng <= 0.0:
        return 0.0

    tol = 1e-12 * rng
    min_nz = 2 if n < 16 else max(3, n // 20)
    med = _median_sorted(sorted_history)
    abs_devs = [abs(x - med) for x in sample_set]
    sorted_abs_devs = sorted(abs_devs)
    mad1 = _median_sorted(sorted_abs_devs)

    if mad1 <= tol:
        nz_devs = [d for d in sorted_abs_devs if d > tol]
        if len(nz_devs) < min_nz:
            return 0.0
        eff_mad1 = _median_sorted(nz_devs)
        r1 = 2.5 * eff_mad1
        nz_inliers = [d for d in nz_devs if d <= r1]
        if len(nz_inliers) < 2:
            return eff_mad1 * _C_GAUSS
        return _truncated_mad_fixed_point(_median_sorted(nz_inliers), r1)

    # Level 1: Step-edge & outlier rejection threshold
    r1 = 2.5 * mad1
    candidates = [
        x for x, dev in zip(sample_set, abs_devs) if dev <= r1
    ]

    c_len = len(candidates)
    if c_len < 2:
        return mad1 * _C_GAUSS

    # Level 2: Pure noise deviation estimation on filtered inliers
    sorted_candidates = sorted(candidates)
    pure_med = _median_sorted(sorted_candidates)

    pure_abs_devs = sorted(abs(x - pure_med) for x in candidates)
    pure_mad = _median_sorted(pure_abs_devs)
    if pure_mad <= tol:
        nz_pure = [d for d in pure_abs_devs if d > tol]
        if len(nz_pure) < min_nz:
            return 0.0
        pure_mad = _median_sorted(nz_pure)

    return _truncated_mad_fixed_point(pure_mad, r1)


def _dual_mad_std_from_array(window: np.ndarray) -> float:
    n = window.size
    if n < 2:
        return 0.0

    rng = float(np.max(window) - np.min(window))
    if rng <= 0.0:
        return 0.0

    tol = 1e-12 * rng
    min_nz = 2 if n < 16 else max(3, n // 20)
    med = float(np.median(window))
    abs_devs = np.abs(window - med)
    mad1 = float(np.median(abs_devs))

    if mad1 <= tol:
        nz_devs = abs_devs[abs_devs > tol]
        if nz_devs.size < min_nz:
            return 0.0
        eff_mad1 = float(np.median(nz_devs))
        r1 = 2.5 * eff_mad1
        nz_inliers = nz_devs[nz_devs <= r1]
        if nz_inliers.size < 2:
            return eff_mad1 * _C_GAUSS
        return _truncated_mad_fixed_point(float(np.median(nz_inliers)), r1)

    r1 = 2.5 * mad1
    candidates = window[abs_devs <= r1]
    if candidates.size < 2:
        return mad1 * _C_GAUSS

    pure_med = float(np.median(candidates))
    pure_devs = np.abs(candidates - pure_med)
    pure_mad = float(np.median(pure_devs))
    if pure_mad <= tol:
        nz_pure = pure_devs[pure_devs > tol]
        if nz_pure.size < min_nz:
            return 0.0
        pure_mad = float(np.median(nz_pure))

    return _truncated_mad_fixed_point(pure_mad, r1)


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
        [KO] 노이즈 배경의 강건한 절단 정규분포 보정 표준편차(sigma) 추정.
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

        return _dual_mad_std_from_list(sample_set)

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
    Processes signed 1st-order differences in optimized chunks with sqrt(2) norm and truncation compensation.

    [KO] 1D 신호를 위한 배치 2중 MAD 노이즈 곡선 추정기 (스텝 에지 면역).
    부호 있는 1차 차분에 대해 2단계 MAD 필터링과 sqrt(2) 정규화 및 절단 정규분포 고정점 보정을 수행합니다.
    """
    arr = np.asarray(signal, dtype=np.float64)
    n = len(arr)
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    if n == 1:
        return np.zeros(1, dtype=np.float64)

    diff = np.diff(arr, prepend=arr[0])
    noise_curve = np.zeros(n, dtype=np.float64)

    for i in range(0, n, chunk_size):
        start = max(0, i - window_size)
        end = min(n, i + chunk_size)
        window = diff[start:end]

        if window.size < 2:
            continue

        noise_curve[i:i + chunk_size] = _dual_mad_std_from_array(window) / _SQRT2

    return noise_curve
