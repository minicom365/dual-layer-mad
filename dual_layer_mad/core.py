# [EN] DualLayerMAD: Robust Noise Baseline Estimation Immune to Step-Edges and Outliers
# [KO] DualLayerMAD: 스텝 에지 및 이상치에 강건한 노이즈 배경 수준 측정 알고리즘
# [Arch] Implements 1D Dual-Layer MAD with Twin-Adjacent-Shell excess point-mass debiasing and 4-step mixture-quantile fixed-point compensation, excluding naive tie-count>=2 filtering and uncompensated zero-inflated MAD2.
# Author: minicom365
# License: MIT

import bisect
import math
from typing import List, Tuple, Union, Sequence
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


def _median_sorted_range(s: List[float], offset: int, count: int) -> float:
    mid = offset + (count >> 1)
    if count & 1:
        return s[mid]
    return 0.5 * (s[mid - 1] + s[mid])


def _median_sorted(s: List[float]) -> float:
    return _median_sorted_range(s, 0, len(s))


def _median_sorted_np(s: np.ndarray) -> float:
    m = int(s.size)
    mid = m >> 1
    if m & 1:
        return float(s[mid])
    return 0.5 * (float(s[mid - 1]) + float(s[mid]))


def _excess_point_mass_list(sorted_devs: List[float], tol: float) -> Tuple[int, float]:
    n = len(sorted_devs)
    first_gt = bisect.bisect_right(sorted_devs, tol)
    if first_gt < 2 or first_gt >= n:
        return first_gt, 0.0
    nz_cnt = n - first_gt
    ref_off = min(nz_cnt - 1, min(3, nz_cnt // 150))
    d_ref = sorted_devs[first_gt + ref_off]
    n1 = bisect.bisect_right(sorted_devs, 1.5 * d_ref, first_gt, n) - first_gt
    if 2 * n1 >= first_gt:
        return first_gt, 0.0
    return first_gt, float(first_gt) / float(n)


def _excess_point_mass_array(sorted_devs: np.ndarray, tol: float) -> Tuple[int, float]:
    n = int(sorted_devs.size)
    first_gt = int(np.searchsorted(sorted_devs, tol, side="right"))
    if first_gt < 2 or first_gt >= n:
        return first_gt, 0.0
    nz_cnt = n - first_gt
    ref_off = min(nz_cnt - 1, min(3, nz_cnt // 150))
    d_ref = float(sorted_devs[first_gt + ref_off])
    n1 = int(np.searchsorted(sorted_devs, 1.5 * d_ref, side="right")) - first_gt
    if 2 * n1 >= first_gt:
        return first_gt, 0.0
    return first_gt, float(first_gt) / float(n)


def _debias_mad1_point_mass(mad1: float, p0_hat: float) -> float:
    if p0_hat <= 0.0:
        return mad1
    p_eff = p0_hat if p0_hat < 0.49 else 0.49
    z_zi = _norm_ppf(1.0 - 0.25 / (1.0 - p_eff))
    return mad1 * (_Z75 / z_zi)


def _truncated_mad_fixed_point(mad2: float, r1: float) -> float:
    sig = mad2 / _Q0_TAU25
    r1_floor = 1e-12 * max(r1, 1.0)
    for _ in range(4):
        c_eff = max(_C0_TAU25, r1 / max(sig, r1_floor))
        sig = mad2 / _norm_ppf(0.25 + 0.5 * _norm_cdf(c_eff))
    return sig


def _mixture_mad_fixed_point(mad2: float, r1: float, p0_in: float) -> float:
    if p0_in <= 0.0:
        return _truncated_mad_fixed_point(mad2, r1)
    w_in = (1.0 - 2.0 * p0_in) / (4.0 * (1.0 - p0_in))
    sig = mad2 / _norm_ppf(0.5 + w_in * (2.0 * _norm_cdf(_C0_TAU25) - 1.0))
    r1_floor = 1e-12 * max(r1, 1.0)
    for _ in range(4):
        c_eff = max(_C0_TAU25, r1 / max(sig, r1_floor))
        sig = mad2 / _norm_ppf(0.5 + w_in * (2.0 * _norm_cdf(c_eff) - 1.0))
    return sig


def _dual_mad_std_from_list(sample_set: List[float]) -> float:
    n = len(sample_set)
    if n < 2:
        return 0.0

    sorted_history = sorted(sample_set)
    rng = sorted_history[-1] - sorted_history[0]
    if math.isnan(rng):
        return float("nan")
    if rng <= 0.0:
        return 0.0

    tol = 1e-12 * rng
    min_nz = 2 if n < 16 else max(3, n // 20)
    med = _median_sorted(sorted_history)
    abs_devs = [abs(x - med) for x in sample_set]
    sorted_abs_devs = sorted(abs_devs)
    mad1 = _median_sorted(sorted_abs_devs)
    first_gt, p0_hat = _excess_point_mass_list(sorted_abs_devs, tol)

    if mad1 <= tol or (first_gt * 2 >= n) or p0_hat >= 0.49:
        nz_count = n - first_gt
        if nz_count < min_nz:
            return 0.0
        eff_mad1 = _median_sorted_range(sorted_abs_devs, first_gt, nz_count)
        r1 = 2.5 * eff_mad1
        nzi_end = bisect.bisect_right(sorted_abs_devs, r1, first_gt, n)
        nzi_count = nzi_end - first_gt
        if nzi_count < 2:
            return eff_mad1 * _C_GAUSS
        return _truncated_mad_fixed_point(_median_sorted_range(sorted_abs_devs, first_gt, nzi_count), r1)

    # Level 1: Zero-inflation-debiased step-edge & outlier rejection threshold (50% breakdown preserved)
    mad1_eff = _debias_mad1_point_mass(mad1, p0_hat)
    r1 = 2.5 * mad1_eff
    candidates = [
        x for x, dev in zip(sample_set, abs_devs) if dev <= r1
    ]

    c_len = len(candidates)
    if c_len < 2:
        return mad1_eff * _C_GAUSS

    # Level 2: Pure noise deviation estimation on filtered inliers
    sorted_candidates = sorted(candidates)
    pure_med = _median_sorted(sorted_candidates)

    pure_abs_devs = sorted(abs(x - pure_med) for x in candidates)
    pure_mad = _median_sorted(pure_abs_devs)
    first_gt2, p0_in = _excess_point_mass_list(pure_abs_devs, tol)

    if pure_mad <= tol or (first_gt2 * 2 >= c_len) or p0_in >= 0.35:
        nz_pure_count = c_len - first_gt2
        if nz_pure_count < min_nz:
            return 0.0
        pure_mad_nz = _median_sorted_range(pure_abs_devs, first_gt2, nz_pure_count)
        return _truncated_mad_fixed_point(pure_mad_nz, r1)

    return _mixture_mad_fixed_point(pure_mad, r1, p0_in)


def _dual_mad_std_from_array(window: np.ndarray) -> float:
    n = int(window.size)
    if n < 2:
        return 0.0
    if np.isnan(window).any():
        return float("nan")

    rng = float(np.max(window) - np.min(window))
    if math.isnan(rng):
        return float("nan")
    if rng <= 0.0:
        return 0.0

    tol = 1e-12 * rng
    min_nz = 2 if n < 16 else max(3, n // 20)
    med = float(np.median(window))
    abs_devs = np.abs(window - med)
    sorted_abs = np.sort(abs_devs)
    mad1 = _median_sorted_np(sorted_abs)
    first_gt, p0_hat = _excess_point_mass_array(sorted_abs, tol)

    if mad1 <= tol or (first_gt * 2 >= n) or p0_hat >= 0.49:
        nz_count = n - first_gt
        if nz_count < min_nz:
            return 0.0
        eff_mad1 = _median_sorted_np(sorted_abs[first_gt:])
        r1 = 2.5 * eff_mad1
        nzi_end = int(np.searchsorted(sorted_abs, r1, side="right"))
        if nzi_end - first_gt < 2:
            return eff_mad1 * _C_GAUSS
        return _truncated_mad_fixed_point(_median_sorted_np(sorted_abs[first_gt:nzi_end]), r1)

    mad1_eff = _debias_mad1_point_mass(mad1, p0_hat)
    r1 = 2.5 * mad1_eff
    candidates = window[abs_devs <= r1]
    c_len = int(candidates.size)
    if c_len < 2:
        return mad1_eff * _C_GAUSS

    pure_med = float(np.median(candidates))
    sorted_pure = np.sort(np.abs(candidates - pure_med))
    pure_mad = _median_sorted_np(sorted_pure)
    first_gt2, p0_in = _excess_point_mass_array(sorted_pure, tol)

    if pure_mad <= tol or (first_gt2 * 2 >= c_len) or p0_in >= 0.35:
        if c_len - first_gt2 < min_nz:
            return 0.0
        return _truncated_mad_fixed_point(_median_sorted_np(sorted_pure[first_gt2:]), r1)

    return _mixture_mad_fixed_point(pure_mad, r1, p0_in)


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
