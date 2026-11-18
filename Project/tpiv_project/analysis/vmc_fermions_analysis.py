# Analysis of the noninteracting-fermion VMC (stage 1): sanity checks, energy runs, correlation monitoring,
# radial number density, same-spin pair distribution and exchange hole.
# The VMC functions are in code/vmc_fermions.py.  Run from anywhere:  python analysis/vmc_fermions_analysis.py
import sys
from pathlib import Path

import numpy as onp
import matplotlib.pyplot as plt
from jax import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from vmc_fermions import (D, rng, logpsi, occupied_orbitals, exact_energy, local_energy, vmc_fermions,  # noqa: E402
                          autocorrelation, autocorrelation_time, radial_number_density, exact_radial_density,
                          same_spin_pair_distances, independent_reference_samples)


# ---------------------------------------------------------------------------
# Sanity checks: shell filling and antisymmetry
# ---------------------------------------------------------------------------

for nu, nd in [(1,1), (2,2), (3,3), (6,6)]:
    print(f"N_up={nu}, N_down={nd} | orbitals={occupied_orbitals(nu)} | E0={exact_energy(nu,nd)}")
assert [exact_energy(n,n) for n in (1,2,3,6)] == [2,6,10,28]

# Same-spin swap: one minus sign, no change in |psi|².
X_test = rng.normal(size=(20,6,2))
X_swapped = X_test.copy()
X_swapped[:,[0,1],:] = X_test[:,[1,0],:]
l1 = logpsi(X_test, 3)
l2 = logpsi(X_swapped, 3)
onp.testing.assert_allclose(l2, l1, atol=1e-12)
print("Same-spin exchange: sign flips and |psi|² is unchanged.")


# ---------------------------------------------------------------------------
# VMC runs: energy and correlation monitoring
# ---------------------------------------------------------------------------

sweep_step = 1   # Metropolis steps between recorded samples: set it to about the tau printed below
runs = {}
autocorr = {}
for nu,nd in [(1,1),(2,2),(3,1)]:
# for nu,nd in [(5,5),(2,8),(4,6)]:
    X,eloc,e,err,accept = vmc_fermions(logpsi, n_up = nu, n_particles=nu + nd, n_samples=1000, n_discard=4, sweep_step=sweep_step)
    runs[(nu,nd)] = X
    expected = exact_energy(nu,nd)
    print(f'{nu}+{nd}: VMC E={e:.12f} ± {err:.2g} | exact={expected} | '
          f'acceptance={accept:.3f} | max local deviation={np.max(np.abs(eloc-expected)):.2g}')
    assert onp.abs(e-expected)<1e-10

    V = 0.5 * onp.sum(X**2, axis=(2,3))   # potential energy, shape (n_samples, n_chains)
    autocorr[(nu,nd)] = autocorrelation(V)
    tau = autocorrelation_time(autocorr[(nu,nd)])
    print(f'{nu}+{nd}: autocorrelation time tau = {tau:.1f} recorded samples '
          f'(sweep_step={sweep_step}) -> effective samples per chain ~ {len(X)/tau:.0f}')

fig,ax = plt.subplots(figsize=(6.6,3.7))
for (nu,nd),X in runs.items():
    # X has shape (n_samples, n_chains, N, D): local energy of the first 250 sweeps, chain 0
    e=local_energy(X[:250],nu)[:,0]
    ax.plot(np.arange(min(250,len(e))),e[:250],label=f'{nu}+{nd}: E={exact_energy(nu,nd)}',lw=1.4)
ax.set(xlabel='Recorded sweep',ylabel=r'Local energy $E_L$',title='Exact-eigenstate VMC: zero-variance benchmark')
ax.legend(frameon=False)
plt.show()

# autocorrelation of V: it should drop to ~0 within a few recorded samples if sweep_step is large enough
fig,ax = plt.subplots(figsize=(6.6,3.7))
for (nu,nd),C in autocorr.items():
    ax.plot(onp.arange(len(C)), C, label=f'{nu}+{nd}')
ax.axhline(0, color='gray', lw=0.8)
ax.set(xlabel='Lag t (recorded samples)', ylabel=r'$C(t)$ of $V=\frac{1}{2}\sum_i r_i^2$',
       title=f'Autocorrelation (sweep_step={sweep_step})')
ax.legend(frameon=False)
plt.show()


# ---------------------------------------------------------------------------
# Radial number density
# ---------------------------------------------------------------------------

# check against the closed form for 3+3: n(r) = 2/pi exp(-r^2) (1 + 2 r^2)
r_test = onp.linspace(0, 3, 30)
onp.testing.assert_allclose(exact_radial_density(r_test, 3, 3),
                            (2/onp.pi)*onp.exp(-r_test**2)*(1 + 2*r_test**2), rtol=1e-5, atol=1e-7)

bins = onp.linspace(0, 5.5, 60)
r_fine = onp.linspace(0, 5.5, 500)
fig, axes = plt.subplots(1, len(runs), figsize=(13,3.6), sharey=True, squeeze=False)
axes = axes[0]
for ax, ((nu,nd), X) in zip(axes, runs.items()):
    # all chains together: (n_samples, n_chains, N, D) -> (n_samples*n_chains, N, D)
    X_flat = X.reshape(-1, nu + nd, D)
    r, density, edges = radial_number_density(X_flat, bins)
    ax.plot(r_fine, exact_radial_density(r_fine, nu, nd), label='Exact orbital density')
    ax.plot(r, density, '.', markersize=4, label='VMC annular density')
    integrated = onp.sum(density * onp.pi * (edges[1:]**2 - edges[:-1]**2))
    ax.set(xlabel=r'$r$', title=f'{nu}+{nd}: annular integral = {integrated:.3f}')
    assert abs(integrated - (nu + nd)) < 0.05
axes[0].set_ylabel(r'$n(r)$ (particles per unit area)')
axes[0].legend(frameon=False, fontsize=9)
fig.tight_layout()
plt.show()


# ---------------------------------------------------------------------------
# Same-spin pair distribution and exchange hole
# ---------------------------------------------------------------------------

# Exchange hole: in 2D even independent particles have P(d) ~ d at small d (phase space 2 pi d dd),
# so we look at the area-normalised pair density P(d)/(2 pi d): for same-spin fermions it goes to 0 (as d^2),
# for the independent reference it stays finite
fig, (ax_pair, ax_area) = plt.subplots(1, 2, figsize=(11,4))
for k, ((nu,nd), X) in enumerate(runs.items()):
    C = nu*(nu-1)//2 + nd*(nd-1)//2
    if C == 0:
        continue
    X_flat = X.reshape(-1, nu + nd, D)
    d_real = same_spin_pair_distances(X_flat, nu).ravel()
    d_ref = same_spin_pair_distances(independent_reference_samples(X_flat), nu).ravel()
    assert d_real.size == len(X_flat) * C

    edges = onp.linspace(0, max(d_real.max(), d_ref.max()) + 1e-6, 62)
    centers = 0.5 * (edges[1:] + edges[:-1])
    widths = onp.diff(edges)
    P = onp.histogram(d_real, bins=edges)[0] / (len(X_flat) * C * widths)
    P_ref = onp.histogram(d_ref, bins=edges)[0] / (len(X_flat) * C * widths)
    assert onp.isclose(onp.sum(P * widths), 1.0)
    assert onp.isclose(onp.sum(P_ref * widths), 1.0)

    # fraction of same-spin pairs closer than d_small: suppressed for fermions
    d_small = 0.3
    frac, frac_ref = onp.mean(d_real < d_small), onp.mean(d_ref < d_small)
    print(f'{nu}+{nd}: same-spin pairs with d < {d_small}: fermions {frac:.4f} | independent reference {frac_ref:.4f}')
    assert frac < frac_ref

    ax_pair.plot(centers, P, color=f'C{k}', label=f'{nu}+{nd} fermions')
    ax_pair.plot(centers, P_ref, color=f'C{k}', ls='--', label=f'{nu}+{nd} reference')
    ax_area.plot(centers, P / (2*onp.pi*centers), color=f'C{k}', label=f'{nu}+{nd} fermions')
    ax_area.plot(centers, P_ref / (2*onp.pi*centers), color=f'C{k}', ls='--', label=f'{nu}+{nd} reference')
ax_pair.set(xlabel=r'Pair separation $d$', ylabel=r'$P_{\mathrm{same}}(d)$', title='Pair-distance PDF (integral = 1)')
ax_area.set(xlabel=r'Pair separation $d$', ylabel=r'$P_{\mathrm{same}}(d)/(2\pi d)$',
            title='Area-normalised pair density: exchange hole', xlim=(0, 2.1))
ax_pair.legend(frameon=False, fontsize=8)
ax_area.legend(frameon=False, fontsize=8)
fig.tight_layout()
plt.show()
