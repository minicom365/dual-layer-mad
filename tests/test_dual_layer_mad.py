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


def test_zero_inflation_sweep_quantization_and_breakdown():
    """Verify p0=0.00..0.60 zero-inflation continuity (odd/even N), uint8 safety, and p0=0.40+31% outlier breakdown."""
    rng = np.random.default_rng(20261124)
    sigma = 2.0
    for n in [1000, 1001]:
        for p0 in [0.00, 0.10, 0.25, 0.30, 0.34, 0.40, 0.48, 0.50, 0.55, 0.60]:
            k0 = int(round(p0 * n))
            vals_list, vals_arr = [], []
            for _ in range(20):
                x = rng.normal(0.0, sigma, size=n)
                if k0 > 0:
                    x[rng.choice(n, size=k0, replace=False)] = 0.0
                est = DualLayerMAD(window_size=n)
                est.history = x.tolist()
                vals_list.append(est.get_std())
                # Batch 1st-diff on cumulative walk of zero-inflated increments
                walk = np.cumsum(x) * math.sqrt(2.0)
                curve = get_dual_mad_noise(walk, window_size=n, chunk_size=n)
                vals_arr.append(float(curve[-1]))

            m_list = float(np.mean(vals_list))
            m_arr = float(np.mean(vals_arr))
            assert 0.90 * sigma <= m_list <= 1.12 * sigma, f"List N={n}, p0={p0}: got {m_list}"
            assert 0.90 * sigma <= m_arr <= 1.12 * sigma, f"Array N={n}, p0={p0}: got {m_arr}"

    # uint8 quantized Gaussian (sigma=2.0, 5.0) without zero-inflation must not jump
    for true_sig, lo, hi in [(2.0, 1.50, 2.30), (5.0, 4.70, 5.30)]:
        q = np.clip(np.round(128.0 + rng.normal(0.0, true_sig, size=1000)), 0.0, 255.0)
        est_q = DualLayerMAD(window_size=1000)
        est_q.history = q.tolist()
        s_q = est_q.get_std()
        assert lo <= s_q <= hi, f"uint8 sigma={true_sig} jumped to {s_q}"

    # p0=0.40 exact zeros + 31% gross outliers (+1000) preserves 50% breakdown
    bd_vals = []
    for _ in range(20):
        x_bd = np.concatenate([np.zeros(400), rng.normal(0.0, 1.0, size=290), np.full(310, 1000.0)])
        rng.shuffle(x_bd)
        est_bd = DualLayerMAD(window_size=1000)
        est_bd.history = x_bd.tolist()
        bd_vals.append(est_bd.get_std())
    assert 0.85 <= float(np.mean(bd_vals)) <= 1.20, f"Breakdown failed: {np.mean(bd_vals)}"


if __name__ == "__main__":
    test_gaussian_noise_estimation()
    test_unbiased_gaussian_large_sample_and_outliers()
    test_quantized_sparse_mad1_zero_fallback()
    test_step_edge_immunity()
    test_batch_dual_mad_noise()
    test_zero_inflation_sweep_quantization_and_breakdown()
    print("[+] All DualLayerMAD unit tests passed successfully!")
