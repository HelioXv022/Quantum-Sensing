import numpy as np
import pytest

from qsensing import (
    analyze_case,
    detection_time,
    effective_signal,
    generate_cases,
    success_probability,
    trace_distance,
)


def test_coherent_signals_add_linearly():
    n = 25
    assert effective_signal(np.full(n, 0.5), np.full(n, 1.3)) == pytest.approx(n * 0.5)


def test_opposite_phases_cancel():
    amplitudes = np.ones(4)
    phases = np.array([0.0, np.pi, 0.5 * np.pi, -0.5 * np.pi])
    assert effective_signal(amplitudes, phases) == pytest.approx(0.0, abs=1e-12)


def test_trace_distance_reaches_one_at_half_period():
    b_eff = 3.0
    t_half = np.pi / (2.0 * b_eff)  # phase_factor 2: Theta = 2 B_eff T = pi
    assert trace_distance(t_half, b_eff) == pytest.approx(1.0)
    assert trace_distance(0.0, b_eff) == pytest.approx(0.0)


def test_detection_time_hits_target():
    b_eff, target = 2.5, 0.8
    tau = detection_time(b_eff, target)
    assert trace_distance(tau, b_eff) == pytest.approx(target)
    assert success_probability(trace_distance(tau, b_eff)) == pytest.approx(0.9)


def test_detection_time_is_inverse_in_signal():
    assert detection_time(1.0, 0.5) == pytest.approx(10.0 * detection_time(10.0, 0.5))


def test_detection_time_edge_cases():
    assert detection_time(0.0, 0.5) == np.inf
    with pytest.raises(ValueError):
        detection_time(1.0, 1.5)


def test_generate_cases_is_reproducible():
    a = generate_cases(36, np.random.default_rng(7))
    b = generate_cases(36, np.random.default_rng(7))
    for case_a, case_b in zip(a, b):
        np.testing.assert_array_equal(case_a.phases, case_b.phases)
        np.testing.assert_array_equal(case_a.amplitudes, case_b.amplitudes)


def test_homogeneous_case_is_fully_coherent():
    homogeneous = generate_cases(16, np.random.default_rng(0))[0]
    row = analyze_case(homogeneous, d_target=0.8)
    assert row["coherence_fraction"] == pytest.approx(1.0)
