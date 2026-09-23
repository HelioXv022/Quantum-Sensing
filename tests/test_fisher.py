"""Fisher-information checks; skipped when squint is not installed (it needs Python 3.11+)."""

import numpy as np
import pytest

pytest.importorskip("squint")

from qsensing.disorder import disorder_average, predictions  # noqa: E402
from qsensing.fisher import fisher_matrices, signal_fisher  # noqa: E402

N = 4


def test_product_state_qfim_is_identity():
    qfim, _ = fisher_matrices(N, "product")
    np.testing.assert_allclose(qfim, np.eye(N), atol=1e-9)


def test_ghz_qfim_is_all_ones():
    # a GHZ probe only sees the sum of the local phases
    qfim, _ = fisher_matrices(N, "ghz")
    np.testing.assert_allclose(qfim, np.ones((N, N)), atol=1e-9)


def test_flipped_ghz_qfim_is_sign_outer_product():
    flips = (1, 2)
    signs = np.array([1, -1, -1, 1])
    qfim, _ = fisher_matrices(N, "ghz", flips=flips)
    np.testing.assert_allclose(qfim, np.outer(signs, signs), atol=1e-9)


@pytest.mark.parametrize("probe", ["product", "ghz"])
def test_x_basis_readout_saturates_qfi(probe):
    qfim, cfim = fisher_matrices(N, probe, phase=0.3)
    np.testing.assert_allclose(cfim, qfim, atol=1e-8)


def test_uniform_couplings_give_sql_and_heisenberg_limit():
    product, _ = fisher_matrices(N, "product")
    ghz, _ = fisher_matrices(N, "ghz")
    g = np.ones(N)
    assert signal_fisher(product, g) == pytest.approx(N)
    assert signal_fisher(ghz, g) == pytest.approx(N**2)


def test_disorder_averages_match_analytic_values():
    n = 6
    product, _ = fisher_matrices(n, "product")
    ghz, _ = fisher_matrices(n, "ghz")
    phases = np.random.default_rng(0).uniform(-np.pi, np.pi, (20000, n))
    means = {k: v.mean() for k, v in disorder_average(product, ghz, phases).items()}
    expected = predictions(np.array([n]))
    for key, value in means.items():
        assert value == pytest.approx(expected[key][0], rel=0.03), key
