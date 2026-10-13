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
    # Should estimate true_sigma within 15% tolerance
    assert abs(est_sigma - true_sigma) / true_sigma < 0.15, f"Expected ~{true_sigma}, got {est_sigma}"

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
    test_step_edge_immunity()
    test_batch_dual_mad_noise()
    print("[+] All DualLayerMAD unit tests passed successfully!")
