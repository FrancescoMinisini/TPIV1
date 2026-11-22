import numpy as onp
import matplotlib.pyplot as plt
import typing
from math import factorial
from functools import partial
# from scipy.special import eval_hermite
from jax import hessian, grad, vmap
from jax import numpy as np
from numpy.polynomial.hermite import herm2poly
from tqdm import tqdm
import jax
# double precision: in float32 the energy is only accurate to ~5e-7, too coarse for the 1e-10 check
jax.config.update("jax_enable_x64", True)


rng = onp.random.default_rng(1)
# number of particles
N = 8
n_up = 2
n_down = N - n_up
n_chains = 10
n_samples = 1000
D = 2

def hermite1d(x, n):
    c = [0]*n + [1]
    # c = [0 for i in range(n)]
    # c+=[1]
    
    #creates a polynomial in the hermite basis
    coeff = herm2poly(c)
    #converts from hermite basis to polynomial basis
    H = np.polyval(coeff[::-1], x)
    return H * np.exp(-0.5*x**2) / (
        np.pi**0.25 * np.sqrt(2.0**n * factorial(n))
    )

def hermite2d(nx, ny, x, y):
    return hermite1d(x, nx) * hermite1d(y, ny)

#as noted in notes we do (nx,ny): (s,0),(s-1,1)...(0,s)
def occupied_orbitals(n):
    labels = []
    s = 0
    while(len(labels)<n):
        for i in range(s+1):
            labels.append((s-i, i))
            if len(labels) == n:
                return labels
        s +=1 
    return labels
        

#Z is the mc samples of dimention (n_chians, N, D: x,y)
# we have to go through the Mij = phi_j ( z_i) where j is the j-th element in the labels and i is the i_th element in the coordinates
def orbital_values(Z,labels):
    N = Z.shape[1]
    x,y = Z[:,:,0], Z[:,:,1]
    M = np.stack([hermite2d(nx,ny, x, y) for nx, ny in labels],axis=-1)
    return M
    
def build_slaters(Z, n_up):
    N = Z.shape[1]
    M_up = orbital_values(Z[:,:n_up:,:], occupied_orbitals(n_up))
    M_down = orbital_values(Z[:,n_up:,:], occupied_orbitals(N-n_up))
    return M_up, M_down
    
# jit: called at every Metropolis step, compiled it is ~10x faster than op-by-op execution
@partial(jax.jit, static_argnums=1)
def logpsi(Z, n_up):
    M_up, M_down = build_slaters(Z, n_up)
    s_up, D_up = np.linalg.slogdet(M_up)
    s_down, D_down = np.linalg.slogdet(M_down)
    return D_up + D_down

def exact_energy(n_up, n_down):
    return sum(nx+ny+1 for nx,ny in occupied_orbitals(n_up)) + \
           sum(nx+ny+1 for nx,ny in occupied_orbitals(n_down))

# for nu, nd in [(1,1), (2,2), (3,3), (6,6)]:
#     print(f"N_up={nu}, N_down={nd} | orbitals={occupied_orbitals(nu)} | E0={exact_energy(nu,nd)}")
# assert [exact_energy(n,n) for n in (1,2,3,6)] == [2,6,10,28]

# Same-spin swap: one minus sign, no change in |psi|².
X_test = rng.normal(size=(20,6,2))
X_swapped = X_test.copy()
X_swapped[:,[0,1],:] = X_test[:,[1,0],:]
l1 = logpsi(X_test, 3)
l2 = logpsi(X_swapped, 3)
onp.testing.assert_allclose(l2, l1, atol=1e-12)
print("Same-spin exchange: sign flips and |psi|² is unchanged.")

    
def metropolis_step(logpsi, Z, logpsi_Z, step_size, n_up, rng=rng):
    Z_prime = Z.copy()
    # Z has shape (n_chains, N, D)
    Z_prime += rng.uniform(-step_size, step_size, size=Z.shape)
    logpsi_Z_prime = logpsi(Z_prime, n_up)
    R = np.exp(np.minimum(0., 2*(logpsi_Z_prime-logpsi_Z)))
    accept = R > np.asarray(rng.random(Z.shape[0]))
    Z_new = np.where(accept[:, None, None], Z_prime, Z)
    
    logpsi_new = np.where(
        accept, logpsi_Z_prime, logpsi_Z
    )

    return Z_new, logpsi_new, accept

# E = -.5 sum(lapl L + grad L^2) + .5*(x^2 +y^2)
def local_energy_single(Z, n_up):
    return kinetic(Z, n_up) + potential(Z)

# vmap works like an iterator over more complex expressions: it batchifies them
# here it maps over the chains: Z.shape = (n_chains, N, D) -> output.shape = (n_chains,)
# jit: local_energy is now called once per recorded sample, without jit jax would re-trace grad and hessian at every call
@partial(jax.jit, static_argnums=1)
def local_energy(Z, n_up):
    f = lambda z: local_energy_single(z, n_up)
    return vmap(f)(Z)


def kinetic(z, n_up):
    f = lambda x: logpsi(x[None, ...], n_up)[0]
    gradL = grad(f)(z)
    hess = hessian(f)(z)
    lapl = np.trace(hess.reshape(z.size, z.size))
    return -0.5 * (lapl + np.sum(gradL**2))

def potential(z):
    return 0.5 * np.sum(z**2)

# sweep_step = number of Metropolis steps between two recorded samples (thinning, to reduce correlation)
# No configuration is stored: at every recorded sample we measure E_L and V of each chain and update running averages.
# Returns
#   energy, error   : running estimate after every recorded sample, shape (n_samples,): energy[i] is the mean of E_L
#                     over the first i+1 samples, error[i] its standard error from the spread of the n_chains
#                     independent chain means. The final results are energy[-1], error[-1]
#   sigma           : running standard deviation of E_L inside a chain, shape (n_samples,), final value sigma[-1]
#                     (0 for an exact eigenstate: zero-variance benchmark; sigma[0] is nan, one sample has no spread)
#   acceptance_rate : fraction of accepted moves after thermalisation
#   V               : potential energy of every recorded sample, shape (n_samples, n_chains); one number per sample
#                     (N*D times smaller than the configurations), kept for the autocorrelation and the running mean
def sample_mcmc(logpsi, n_particles, n_samples, n_discard, step_size, n_up, n_chains, Z0=None, rng=rng, sweep_step=1):
    # Initial state of every chain
    if Z0 is None:
        Z = np.asarray(rng.normal(size=(n_chains, n_particles, D)))
    else:
        Z = np.asarray(Z0)
        assert Z.shape == (n_chains, n_particles, D)
    # Current log wavefunction for each chain
    logpsi_Z = logpsi(Z, n_up)

    # Thermalisation: move the chains, measure nothing. The acceptance is not counted here either,
    # it should describe the sampler in equilibrium, not the walk away from the random starting point
    for _ in tqdm(range(n_discard * sweep_step), desc='thermalisation'):
        Z, logpsi_Z, _ = metropolis_step(logpsi, Z, logpsi_Z, step_size, n_up, rng)

    # Running mean and sum of squared deviations of E_L for each chain (Welford's update). It is numerically stable,
    # unlike sum(E^2) - sum(E)^2, which cancels catastrophically when E_L is almost constant as it is here
    E_mean = onp.zeros(n_chains)
    E_M2 = onp.zeros(n_chains)
    V = onp.empty((n_samples, n_chains))
    n_accepted = onp.zeros(n_chains)
    # running estimates, one value per recorded sample (n_samples numbers each, no n_chains or N*D axis)
    energy = onp.empty(n_samples)
    error = onp.empty(n_samples)
    sigma = onp.empty(n_samples)

    for i in tqdm(range(n_samples), desc='sampling'):
        for _ in range(sweep_step):
            Z, logpsi_Z, accept = metropolis_step(logpsi, Z, logpsi_Z, step_size, n_up, rng)
            n_accepted += accept
        eloc = onp.asarray(local_energy(Z, n_up))      # shape (n_chains,)
        delta = eloc - E_mean
        E_mean += delta / (i + 1)
        E_M2 += delta * (eloc - E_mean)
        V[i] = vmap(potential)(Z)

        # E_mean holds the mean energy of each chain over the first i+1 samples. The chains are independent,
        # so the spread of their means gives an honest error bar even though the samples inside one chain are correlated
        energy[i] = E_mean.mean()
        error[i] = E_mean.std(ddof=1) / onp.sqrt(n_chains)
        sigma[i] = onp.sqrt(onp.mean(E_M2) / i) if i > 0 else onp.nan

    acceptance_rate = n_accepted.mean() / (n_samples * sweep_step)
    return energy, error, sigma, acceptance_rate, V

# Correlation monitoring
# E_L is constant for these exact eigenstates (its autocorrelation is 0/0), so we monitor the potential
# energy V = 1/2 sum_i r_i^2 of every recorded sample, obs shape (n_samples, n_chains)
# C(t) = <dO_s dO_{s+t}> / <dO^2>, dO = O - chain mean, averaged over s and over chains
def autocorrelation(obs, max_lag=100):
    dO = obs - obs.mean(axis=0)
    return onp.array([onp.mean(dO[:len(dO)-t] * dO[t:]) for t in range(min(max_lag, len(dO)//2))]) / onp.mean(dO**2)

# integrated autocorrelation time tau = 1 + 2 sum_t C(t), summed until C(t) first drops below 0:
# samples ~tau recorded steps apart are roughly independent, so n_samples/tau is the effective number of samples
def autocorrelation_time(C):
    cut = onp.argmax(C < 0) if onp.any(C < 0) else len(C)
    return 1 + 2 * onp.sum(C[1:cut])

sweep_step = 5   # Metropolis steps between recorded samples: set it to about the tau printed below
runs = {}        # (nu,nd) -> potential energy trace V, shape (n_samples, n_chains)
energy_runs = {} # (nu,nd) -> (energy, error, sigma) running estimates, each of shape (n_samples,)
autocorr = {}
# for nu,nd in [(1,1),(2,2),(3,1)]:
for nu,nd in [(4,4)]:
    e, err, sigma, accept, V = sample_mcmc(logpsi, n_particles=nu + nd, n_samples=5000, n_discard=400, step_size=1.1, n_up=nu, n_chains=200, sweep_step=sweep_step)
    runs[(nu,nd)] = V
    energy_runs[(nu,nd)] = (e, err, sigma)
    expected = exact_energy(nu,nd)
    print(f'{nu}+{nd}: VMC E={e[-1]:.12f} ± {err[-1]:.2g} | exact={expected} | '
          f'acceptance={accept:.3f} | std of E_L={sigma[-1]:.2g}')
    assert onp.abs(e[-1]-expected)<1e-10

    autocorr[(nu,nd)] = autocorrelation(V)
    tau = autocorrelation_time(autocorr[(nu,nd)])
    print(f'{nu}+{nd}: autocorrelation time tau = {tau:.1f} recorded samples '
          f'(sweep_step={sweep_step}) -> effective samples per chain ~ {len(V)/tau:.0f}')

# Running mean vs iteration
# E_L is constant for an exact eigenstate, so its running mean sits on E from the start and does not test the sampler.
# Virial theorem (harmonic trap): <V> = <T> = E/2, so 2<V> -> E only if the samples are really distributed as |psi|^2
fig, ax = plt.subplots(figsize=(6.6,3.7))
for k, ((nu,nd), V) in enumerate(runs.items()):
    it = onp.arange(1, len(V) + 1)
    # running mean of each chain separately; the chains are independent, so the spread between them
    # gives an honest error bar even though samples inside one chain are correlated
    run_chain = onp.cumsum(2*V, axis=0) / it[:, None]             # shape (n_samples, n_chains)
    mean = run_chain.mean(axis=1)
    err = run_chain.std(axis=1, ddof=1) / onp.sqrt(run_chain.shape[1])
    ax.plot(it, mean, color=f'C{k}', label=f'{nu}+{nd}: running mean of $2V$')
    ax.fill_between(it, mean - err, mean + err, color=f'C{k}', alpha=0.25, lw=0)
    ax.axhline(exact_energy(nu,nd), color=f'C{k}', ls='--', lw=0.8)
ax.set(xlabel='Iteration (recorded sample)', ylabel='Running mean', title='Convergence to the exact energy E (dashed)')
ax.legend(frameon=False, fontsize=8)
plt.show()

# fig,ax = plt.subplots(figsize=(6.6,3.7))
# for (nu,nd),X in runs.items():
#     # X has shape (n_samples, n_chains, N, D): local energy of the first 250 sweeps, chain 0
#     e=local_energy(X[:250],nu)[:,0]
#     ax.plot(np.arange(min(250,len(e))),e[:250],label=f'{nu}+{nd}: E={exact_energy(nu,nd)}',lw=1.4)
# ax.set(xlabel='Recorded sweep',ylabel=r'Local energy $E_L$',title='Exact-eigenstate VMC: zero-variance benchmark')
# ax.legend(frameon=False)
# plt.show()

# # autocorrelation of V: it should drop to ~0 within a few recorded samples if sweep_step is large enough
# fig,ax = plt.subplots(figsize=(6.6,3.7))
# for (nu,nd),C in autocorr.items():
#     ax.plot(onp.arange(len(C)), C, label=f'{nu}+{nd}')
# ax.axhline(0, color='gray', lw=0.8)
# ax.set(xlabel='Lag t (recorded samples)', ylabel=r'$C(t)$ of $V=\frac{1}{2}\sum_i r_i^2$',
#        title=f'Autocorrelation (sweep_step={sweep_step})')
# ax.legend(frameon=False)
# plt.show()


# # Radial number density
# # n(r) = < sum_i delta(r - r_i) >, normalised so that 2 pi int r n(r) dr = N
# # estimator: number of particles in the annulus [r_k, r_k+1) / (number of configurations * annulus area)
# def radial_number_density(samples, bins):
#     # samples has shape (n_configs, N, D)
#     radii = onp.linalg.norm(samples, axis=-1).ravel()
#     counts, edges = onp.histogram(radii, bins=bins)
#     areas = onp.pi * (edges[1:]**2 - edges[:-1]**2)
#     centers = 0.5 * (edges[1:] + edges[:-1])
#     return centers, counts / (len(samples) * areas), edges

# # exact density: sum over both spins of |phi_j(r)|^2 for the occupied orbitals, averaged over the angle
# def exact_radial_density(r, n_up, n_down, n_angles=96):
#     theta = 2*onp.pi*onp.arange(n_angles)/n_angles
#     # points on circles, shape (len(r), n_angles, 2): same layout (n_chains, N, D) that orbital_values expects
#     xy = onp.stack([onp.outer(r, onp.cos(theta)), onp.outer(r, onp.sin(theta))], axis=-1)
#     density = onp.zeros((len(r), n_angles))
#     for n in (n_up, n_down):
#         if n > 0:
#             phi = onp.asarray(orbital_values(xy, occupied_orbitals(n)))
#             density += onp.sum(phi**2, axis=-1)
#     return density.mean(axis=1)

# # check against the closed form for 3+3: n(r) = 2/pi exp(-r^2) (1 + 2 r^2)
# r_test = onp.linspace(0, 3, 30)
# onp.testing.assert_allclose(exact_radial_density(r_test, 3, 3),
#                             (2/onp.pi)*onp.exp(-r_test**2)*(1 + 2*r_test**2), rtol=1e-5, atol=1e-7)

# bins = onp.linspace(0, 5.5, 60)
# r_fine = onp.linspace(0, 5.5, 500)
# fig, axes = plt.subplots(1, len(runs), figsize=(13,3.6), sharey=True, squeeze=False)
# axes = axes[0]
# for ax, ((nu,nd), X) in zip(axes, runs.items()):
#     # all chains together: (n_samples, n_chains, N, D) -> (n_samples*n_chains, N, D)
#     X_flat = X.reshape(-1, nu + nd, D)
#     r, density, edges = radial_number_density(X_flat, bins)
#     ax.plot(r_fine, exact_radial_density(r_fine, nu, nd), label='Exact orbital density')
#     ax.plot(r, density, '.', markersize=4, label='VMC annular density')
#     integrated = onp.sum(density * onp.pi * (edges[1:]**2 - edges[:-1]**2))
#     ax.set(xlabel=r'$r$', title=f'{nu}+{nd}: annular integral = {integrated:.3f}')
#     assert abs(integrated - (nu + nd)) < 0.05
# axes[0].set_ylabel(r'$n(r)$ (particles per unit area)')
# axes[0].legend(frameon=False, fontsize=9)
# fig.tight_layout()
# plt.show()


# # Same-spin pair distribution
# # P_same(d) = 1/C < sum_sigma sum_{i<j in sigma} delta(d - |r_i - r_j|) >, with int P_same(d) dd = 1
# # C = C(n_up,2) + C(n_down,2) is the number of same-spin pairs (for 1+1, C = 0 and P_same is undefined)
# def same_spin_pair_distances(samples, n_up):
#     # samples has shape (n_configs, N, D); spin up = particles [0, n_up), spin down = [n_up, N)
#     # returns the distances, shape (n_configs, C)
#     n_configs, N, _ = samples.shape
#     pieces = []
#     for start, stop in ((0, n_up), (n_up, N)):
#         i, j = onp.triu_indices(stop - start, k=1)
#         if len(i) > 0:
#             diff = samples[:, start + i, :] - samples[:, start + j, :]
#             pieces.append(onp.linalg.norm(diff, axis=-1))
#     return onp.concatenate(pieces, axis=1) if pieces else onp.empty((n_configs, 0))

# # Reference without exchange correlations: shuffle the configurations independently for every particle.
# # Each particle keeps its own one-body distribution, but particles in the same configuration become independent.
# def independent_reference_samples(samples, rng=rng):
#     ref = samples.copy()
#     for i in range(samples.shape[1]):
#         ref[:, i, :] = samples[rng.permutation(len(samples)), i, :]
#     return ref

# # Exchange hole: in 2D even independent particles have P(d) ~ d at small d (phase space 2 pi d dd),
# # so we look at the area-normalised pair density P(d)/(2 pi d): for same-spin fermions it goes to 0 (as d^2),
# # for the independent reference it stays finite
# fig, (ax_pair, ax_area) = plt.subplots(1, 2, figsize=(11,4))
# for k, ((nu,nd), X) in enumerate(runs.items()):
#     C = nu*(nu-1)//2 + nd*(nd-1)//2
#     if C == 0:
#         continue
#     X_flat = X.reshape(-1, nu + nd, D)
#     d_real = same_spin_pair_distances(X_flat, nu).ravel()
#     d_ref = same_spin_pair_distances(independent_reference_samples(X_flat), nu).ravel()
#     assert d_real.size == len(X_flat) * C

#     edges = onp.linspace(0, max(d_real.max(), d_ref.max()) + 1e-6, 62)
#     centers = 0.5 * (edges[1:] + edges[:-1])
#     widths = onp.diff(edges)
#     P = onp.histogram(d_real, bins=edges)[0] / (len(X_flat) * C * widths)
#     P_ref = onp.histogram(d_ref, bins=edges)[0] / (len(X_flat) * C * widths)
#     assert onp.isclose(onp.sum(P * widths), 1.0)
#     assert onp.isclose(onp.sum(P_ref * widths), 1.0)

#     # fraction of same-spin pairs closer than d_small: suppressed for fermions
#     d_small = 0.3
#     frac, frac_ref = onp.mean(d_real < d_small), onp.mean(d_ref < d_small)
#     print(f'{nu}+{nd}: same-spin pairs with d < {d_small}: fermions {frac:.4f} | independent reference {frac_ref:.4f}')
#     assert frac < frac_ref

#     ax_pair.plot(centers, P, color=f'C{k}', label=f'{nu}+{nd} fermions')
#     ax_pair.plot(centers, P_ref, color=f'C{k}', ls='--', label=f'{nu}+{nd} reference')
#     ax_area.plot(centers, P / (2*onp.pi*centers), color=f'C{k}', label=f'{nu}+{nd} fermions')
#     ax_area.plot(centers, P_ref / (2*onp.pi*centers), color=f'C{k}', ls='--', label=f'{nu}+{nd} reference')
# ax_pair.set(xlabel=r'Pair separation $d$', ylabel=r'$P_{\mathrm{same}}(d)$', title='Pair-distance PDF (integral = 1)')
# ax_area.set(xlabel=r'Pair separation $d$', ylabel=r'$P_{\mathrm{same}}(d)/(2\pi d)$',
#             title='Area-normalised pair density: exchange hole', xlim=(0, 2.1))
# ax_pair.legend(frameon=False, fontsize=8)
# ax_area.legend(frameon=False, fontsize=8)
# fig.tight_layout()
# plt.show()