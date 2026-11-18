# VMC for noninteracting fermions in a 2D harmonic trap (stage 1):
# Slater determinants of harmonic-oscillator orbitals, Metropolis sampling over many chains,
# local energy with automatic differentiation, correlation and observables (density, pair distribution).
# Only definitions, nothing runs on import: the analysis is in analysis/vmc_fermions_analysis.py
import numpy as onp
from math import factorial
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


# ---------------------------------------------------------------------------
# Orbitals and Slater determinants
# ---------------------------------------------------------------------------

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

def logpsi(Z, n_up):
    M_up, M_down = build_slaters(Z, n_up)
    s_up, D_up = np.linalg.slogdet(M_up)
    s_down, D_down = np.linalg.slogdet(M_down)
    return D_up + D_down

def exact_energy(n_up, n_down):
    return sum(nx+ny+1 for nx,ny in occupied_orbitals(n_up)) + \
           sum(nx+ny+1 for nx,ny in occupied_orbitals(n_down))


# ---------------------------------------------------------------------------
# Metropolis sampling and local energy
# ---------------------------------------------------------------------------

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
# I average over samples and over chains, vmap maps a function to additional axes, if you map it 2 times it ads two additional axes
def local_energy(samples, n_up):
    f = lambda z: local_energy_single(z, n_up)
    return vmap( vmap(f) )(samples)
# output.shape = (n_samples, n_chains)

def kinetic(Z, n_up):
    f = lambda x: logpsi(x[None, ...], n_up)[0]
    gradL = grad(f)(Z)
    hess = hessian(f)(Z)
    lapl = np.trace(hess.reshape(Z.size, Z.size))
    return -0.5 * (lapl + np.sum(gradL**2))

def potential(Z):
    return 0.5 * np.sum(Z**2)

# sweep_step = number of Metropolis steps between two recorded samples (thinning, to reduce correlation)
def sample_mcmc(logpsi, n_particles, n_samples, n_discard, step_size, n_up, n_chains, Z0=None, rng=rng, sweep_step=1):
    # Initial state of every chain
    if Z0 is None:
        Z = np.asarray(rng.normal(size=(n_chains, n_particles, D)))
    else:
        Z = np.asarray(Z0)
        assert Z.shape == (n_chains, n_particles, D)
    # Current log wavefunction for each chain
    logpsi_Z = logpsi(Z, n_up)
    samples = onp.empty((n_samples, n_chains, n_particles, D))
    n_accepted = onp.zeros(n_chains)
    acceptance_rate = onp.zeros(n_chains)

    for i in tqdm(range(n_discard+n_samples)):
        for _ in range(sweep_step):
            Z, logpsi_Z, accept = metropolis_step(logpsi, Z, logpsi_Z, step_size, n_up, rng)
            if i >= n_discard:
                n_accepted += accept
        if i >= n_discard:
            samples[i- n_discard,...] =  Z
    acceptance_rate = n_accepted / (n_samples * sweep_step)

    return samples, acceptance_rate

# Do we need to count the acceptance rate + n_accepted after discarding thermalization? or after


# ---------------------------------------------------------------------------
# Energy estimates
# ---------------------------------------------------------------------------

def vmc_fermions_across_chains( logpsi, n_up, n_particles, n_samples=2500, n_discard=400, step_size=1.1, n_blocks=20, n_chains = 10, sweep_step=1 ):
    samples, acceptance_rate = sample_mcmc( logpsi, n_particles, n_samples, n_discard, step_size, n_up, n_chains= n_chains, sweep_step=sweep_step )
    eloc = local_energy(samples, n_up)

    # Mean energy for each chain
    energy = eloc.mean(axis=0)

    # Estimate standard error using block averaging
    block_size = n_samples // n_blocks
    assert block_size >= 1

    blocks = eloc[:n_blocks * block_size].reshape( n_blocks, block_size, -1 ).mean(axis=1)
    error = blocks.std(axis=0, ddof=1) / np.sqrt(n_blocks)
    return samples, eloc, energy, error, acceptance_rate


def vmc_fermions( logpsi, n_up, n_particles,  n_samples=2500, n_discard=400, step_size=1.1, n_chains = 10, sweep_step=1):
    samples, eloc, chain_energies, _, ar = (vmc_fermions_across_chains( logpsi, n_up, n_particles,n_samples, n_discard, step_size, n_chains=n_chains, sweep_step=sweep_step ))

    energy = chain_energies.mean()

    # Standard error across independent chains
    error = chain_energies.std(ddof=1) / np.sqrt( len(chain_energies) )
    acceptance_rate = ar.mean()
    return samples, eloc, energy, error, acceptance_rate


# ---------------------------------------------------------------------------
# Correlation monitoring
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Observables: radial number density and same-spin pair distribution
# ---------------------------------------------------------------------------

# Radial number density
# n(r) = < sum_i delta(r - r_i) >, normalised so that 2 pi int r n(r) dr = N
# estimator: number of particles in the annulus [r_k, r_k+1) / (number of configurations * annulus area)
def radial_number_density(samples, bins):
    # samples has shape (n_configs, N, D)
    radii = onp.linalg.norm(samples, axis=-1).ravel()
    counts, edges = onp.histogram(radii, bins=bins)
    areas = onp.pi * (edges[1:]**2 - edges[:-1]**2)
    centers = 0.5 * (edges[1:] + edges[:-1])
    return centers, counts / (len(samples) * areas), edges

# exact density: sum over both spins of |phi_j(r)|^2 for the occupied orbitals, averaged over the angle
def exact_radial_density(r, n_up, n_down, n_angles=96):
    theta = 2*onp.pi*onp.arange(n_angles)/n_angles
    # points on circles, shape (len(r), n_angles, 2): same layout (n_chains, N, D) that orbital_values expects
    xy = onp.stack([onp.outer(r, onp.cos(theta)), onp.outer(r, onp.sin(theta))], axis=-1)
    density = onp.zeros((len(r), n_angles))
    for n in (n_up, n_down):
        if n > 0:
            phi = onp.asarray(orbital_values(xy, occupied_orbitals(n)))
            density += onp.sum(phi**2, axis=-1)
    return density.mean(axis=1)

# Same-spin pair distribution
# P_same(d) = 1/C < sum_sigma sum_{i<j in sigma} delta(d - |r_i - r_j|) >, with int P_same(d) dd = 1
# C = C(n_up,2) + C(n_down,2) is the number of same-spin pairs (for 1+1, C = 0 and P_same is undefined)
def same_spin_pair_distances(samples, n_up):
    # samples has shape (n_configs, N, D); spin up = particles [0, n_up), spin down = [n_up, N)
    # returns the distances, shape (n_configs, C)
    n_configs, N, _ = samples.shape
    pieces = []
    for start, stop in ((0, n_up), (n_up, N)):
        i, j = onp.triu_indices(stop - start, k=1)
        if len(i) > 0:
            diff = samples[:, start + i, :] - samples[:, start + j, :]
            pieces.append(onp.linalg.norm(diff, axis=-1))
    return onp.concatenate(pieces, axis=1) if pieces else onp.empty((n_configs, 0))

# Reference without exchange correlations: shuffle the configurations independently for every particle.
# Each particle keeps its own one-body distribution, but particles in the same configuration become independent.
def independent_reference_samples(samples, rng=rng):
    ref = samples.copy()
    for i in range(samples.shape[1]):
        ref[:, i, :] = samples[rng.permutation(len(samples)), i, :]
    return ref
