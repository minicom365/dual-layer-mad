# [EN] Unit tests for DualLayerMAD
# [KO] DualLayerMAD 단위 테스트 및 스텝 에지 면역성 검증

import math
import numpy as np
import pytest
from dual_layer_mad import DualLayerMAD, get_dual_mad_noise

def test_gaussian_noise_estimation():
    np.random.seed(42)
    true_sigma = 2.5
    noise = np.random.normal(0, true_sigma, 500)

    estimator = DualLayerMAD(window_size=64)
    for x in noise:
        estimator.update(x)

    est_sigma = estimator.get_std()
    # Should estimate true_sigma within 15% tolerance on 64-sample rolling window
    assert abs(est_sigma - true_sigma) / true_sigma < 0.15, f"Expected ~{true_sigma}, got {est_sigma}"


def test_unbiased_gaussian_large_sample_and_outliers():
    """Verify <5% relative error on n=1000 Gaussian (streaming & 1st-diff batch), eliminating -10.46% truncation bias."""
    rng = np.random.default_rng(12345)
    true_sigma = 3.0
    n = 1000
    clean = rng.normal(0.0, true_sigma, n)

    est = DualLayerMAD(window_size=n)
    for x in clean:
        est.update(float(x))
    sigma_stream = est.get_std()
    assert abs(sigma_stream - true_sigma) / true_sigma < 0.05, f"Unbiased streaming expected ~{true_sigma}, got {sigma_stream}"

    # Also verify under 10% extreme outliers that 4-step fixed point does not overshoot
    contaminated = clean.copy()
    contaminated[:100] += 250.0
    est_out = DualLayerMAD(window_size=n)
    for x in contaminated:
        est_out.update(float(x))
    sigma_out = est_out.get_std()
    assert abs(sigma_out - true_sigma) / true_sigma < 0.05, f"Outlier-contaminated expected ~{true_sigma}, got {sigma_out}"

    # Verify signed 1st-diff + sqrt(2) norm on sloped Gaussian signal
    ramp_noisy = 1.5 * np.arange(n, dtype=np.float64) + clean
    curve = get_dual_mad_noise(ramp_noisy, window_size=500, chunk_size=100)
    sigma_batch = float(np.median(curve[200:]))
    assert abs(sigma_batch - true_sigma) / true_sigma < 0.05, f"Unbiased batch 1st-diff expected ~{true_sigma}, got {sigma_batch}"


def test_quantized_sparse_mad1_zero_fallback():
    """Verify non-zero fallback when >50% of samples equal the median (mad1 == 0) on quantized/sparse signals."""
    rng = np.random.default_rng(7)
    # 65% exact zeros + 33% active Gaussian jitter (sigma=2.0) + 2% extreme outliers
    n = 400
    sparse = np.zeros(n, dtype=np.float64)
    active_idx = rng.choice(n, size=140, replace=False)
    sparse[active_idx[:132]] = rng.normal(0.0, 2.0, size=132)
    sparse[active_idx[132:]] = 200.0

    est = DualLayerMAD(window_size=n)
    for x in sparse:
        est.update(float(x))
    s_sparse = est.get_std()
    assert 1.5 < s_sparse < 2.5, f"Sparse mad1==0 fallback failed: got {s_sparse}"

    # Quantized integer signal where mad1 == 0
    quantized = np.round(rng.normal(0.0, 0.65, size=300))
    est_q = DualLayerMAD(window_size=300)
    for x in quantized:
        est_q.update(float(x))
    assert est_q.get_std() > 0.5, f"Quantized signal collapsed to {est_q.get_std()}"

    # Batch get_dual_mad_noise on quantized staircase signal where >50% of 1st-diffs are zero (mad1 == 0)
    quant_sig = np.round(np.linspace(0.0, 20.0, n) + rng.normal(0.0, 0.55, size=n))
    quant_sig[150] += 80.0
    batch_curve = get_dual_mad_noise(quant_sig, window_size=200, chunk_size=50)
    assert 0.4 < float(np.median(batch_curve)) < 2.0, f"Batch mad1==0 fallback failed: {np.median(batch_curve)}"


def test_step_edge_immunity():
    """Verify that a massive step-edge (jump of +100) does not blow up the noise estimate"""
    np.random.seed(42)
    true_sigma = 1.0
    
    # Baseline noise around 0.0
    signal_part1 = np.random.normal(0.0, true_sigma, 50)
    # Sudden step edge to 100.0 (e.g. tracking jump)
    signal_part2 = np.random.normal(100.0, true_sigma, 50)
    combined = np.concatenate([signal_part1, signal_part2])

    estimator = DualLayerMAD(window_size=32)
    stds = []
    for x in combined:
        estimator.update(x)
        stds.append(estimator.get_std())

    # Right at the step transition (frame 50), the estimate must remain stable and not blow up to 50+
    max_transient_std = max(stds[48:55])
    assert max_transient_std < 4.0 * true_sigma, f"Step-edge blew up noise estimation to {max_transient_std}"

def test_batch_dual_mad_noise():
    np.random.seed(42)
    n = 200
    signal = np.sin(np.linspace(0, 10, n)) + np.random.normal(0, 0.5, n)
    # Inject large outliers
    signal[50] += 50.0
    signal[120] -= 40.0

    noise_curve = get_dual_mad_noise(signal, window_size=40, chunk_size=10)
    assert len(noise_curve) == n
    # Outlier points should not cause noise curve to explode
    assert np.all(noise_curve < 3.0), f"Noise curve exceeded bound: max {np.max(noise_curve)}"
    assert np.mean(noise_curve) > 0.1

if __name__ == "__main__":
    test_gaussian_noise_estimation()
    test_unbiased_gaussian_large_sample_and_outliers()
    test_quantized_sparse_mad1_zero_fallback()
    test_step_edge_immunity()
    test_batch_dual_mad_noise()
    print("[+] All DualLayerMAD unit tests passed successfully!")
