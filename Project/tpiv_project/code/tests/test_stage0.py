# Tests for stage 0. Run from the code directory:  python -m pytest tests -q
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stage0 import (autocorrelation_time, estimate_energy, exact_energy, grad_logpsi_gaussian,  # noqa: E402
                    harmonic_potential, laplacian_logpsi_gaussian, local_energy, logpsi_gaussian,
                    optimize_gaussian, vmc_gaussian)


def eloc_at(alpha, positions):
    # local energy at given 2D positions, one particle
    params = np.array([alpha])
    x = np.array(positions, dtype=float).reshape(-1, 1, 2)
    return local_energy(lambda x: logpsi_gaussian(params, x), lambda x: grad_logpsi_gaussian(params, x),
                        lambda x: laplacian_logpsi_gaussian(params, x), harmonic_potential, x)


def test_local_energy_hand_selected_positions():
    # E_L = d alpha + (1/2 - 2 alpha^2) r^2 with d = 2
    assert eloc_at(0.3, [0, 0])[0] == pytest.approx(0.6)
    assert eloc_at(0.3, [1, 0])[0] == pytest.approx(0.6 + 0.32)
    assert eloc_at(0.8, [1, 1])[0] == pytest.approx(1.6 - 0.78*2)
    assert eloc_at(0.5, [0.3, -1.2])[0] == pytest.approx(1.0)


def test_local_energy_matches_automatic_differentiation():
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    jnp = jax.numpy
    alpha = 0.3
    logpsi = lambda r: -alpha*jnp.sum(r**2)

    def eloc_autodiff(r):
        grad = jax.grad(logpsi)(r)
        laplacian = jnp.trace(jax.hessian(logpsi)(r))
        return -0.5*(laplacian + grad @ grad) + 0.5*jnp.sum(r**2)

    positions = np.random.default_rng(0).normal(size=(10, 2))
    expected = [float(eloc_autodiff(jnp.array(r))) for r in positions]
    np.testing.assert_allclose(eloc_at(alpha, positions), expected, atol=1e-12)


def test_exact_state_has_zero_variance():
    # note: this checks the local energy, not the sampler (E_L = 1 wherever we sample)
    x, eloc, _ = vmc_gaussian(0.5, n_samples=2000)
    assert np.max(np.abs(eloc - 1)) < 1e-12


def test_sampler_reproduces_r2():
    # |psi|^2 ~ exp(-2 alpha r^2), so <r^2> = d / (4 alpha)
    x, _, _ = vmc_gaussian(0.3)
    mean, err = estimate_energy(np.sum(x**2, axis=(1, 2)))
    assert abs(mean - 2/(4*0.3)) < 4*err


@pytest.mark.parametrize("alpha", [0.3, 0.8])
def test_energy_matches_analytic_curve(alpha):
    _, eloc, _ = vmc_gaussian(alpha)
    energy, err = estimate_energy(eloc)
    assert abs(energy - exact_energy(alpha)) < 4*err


def test_autocorrelation_time_of_independent_samples():
    assert autocorrelation_time(np.random.default_rng(1).normal(size=10000)) == pytest.approx(1, abs=0.1)


def test_optimizer_finds_exact_alpha():
    params, _ = optimize_gaussian(alpha0=0.2)
    assert abs(params[0] - 0.5) < 1e-3
