import numpy as onp
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


RNG = onp.random.default_rng(1)

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

# Jastrow widths s_k: fixed, geometrically spaced between R0/2 and 2.5; only the a_k (params, shape (8,)) are variational
R0 = 0.1
S_K = onp.geomspace(R0/2, 2.5, 8)

# jit: called at every Metropolis step, compiled it is ~10x faster than op-by-op execution
@partial(jax.jit, static_argnums=2)
def logpsi(Z, params, n_up):
    M_up, M_down = build_slaters(Z, n_up)
    s_up, D_up = np.linalg.slogdet(M_up)
    s_down, D_down = np.linalg.slogdet(M_down)
    r2 = build_r2(Z, n_up)
    # u(r_ij) for every up-down pair, shape (n_chains, n_up, n_down), then sum over pairs only (not over chains)
    J = np.sum(np.exp(-r2[..., None]/(2*S_K**2)) @ params, axis=(1, 2))
    return D_up + D_down + J


def build_r2(Z, n_up):
    up = Z[:,:n_up,]
    down = Z[:,n_up:,:]
    diff = up[:, :, None, :] - down[:, None, :, :]
    return np.sum(diff**2, axis=-1)
# r.shape = (n_chains, n_up, n_down)

def initialize_J_params(jastrow_cutoff, sigma):
    return RNG.normal(0, sigma, jastrow_cutoff)

def exact_energy(n_up, n_down):
    return sum(nx+ny+1 for nx,ny in occupied_orbitals(n_up)) + \
           sum(nx+ny+1 for nx,ny in occupied_orbitals(n_down))


def metropolis_step(logpsi, Z, logpsi_Z, step_size, params, n_up, RNG=RNG):
    Z_prime = Z.copy()
    # Z has shape (n_chains, N, D)
    Z_prime += RNG.uniform(-step_size, step_size, size=Z.shape)
    logpsi_Z_prime = logpsi(Z_prime, params, n_up)
    R = np.exp(np.minimum(0., 2*(logpsi_Z_prime-logpsi_Z)))
    accept = R > np.asarray(RNG.random(Z.shape[0]))
    Z_new = np.where(accept[:, None, None], Z_prime, Z)
    
    logpsi_new = np.where(
        accept, logpsi_Z_prime, logpsi_Z
    )

    return Z_new, logpsi_new, accept

# E = -.5 sum(lapl L + grad L^2) + .5*(x^2 +y^2)
def local_energy_single(Z, params, n_up, V0):
    return kinetic(Z, params, n_up) + potential(Z, n_up, V0)

# vmap works like an iterator over more complex expressions: it batchifies them
# here it maps over the chains: Z.shape = (n_chains, N, D) -> output.shape = (n_chains,)
# jit: local_energy is now called once per recorded sample, without jit jax would re-trace grad and hessian at every call
@partial(jax.jit, static_argnums=2)
def local_energy(Z, params, n_up, V0):
    f = lambda z: local_energy_single(z, params, n_up, V0)
    return vmap(f)(Z)


def kinetic(z, params, n_up):
    f = lambda x: logpsi(x[None, ...], params, n_up)[0]
    gradL = grad(f)(z)
    hess = hessian(f)(z)
    lapl = np.trace(hess.reshape(z.size, z.size))
    return -0.5 * (lapl + np.sum(gradL**2))

# z has shape (N,D)
def potential(z, n_up, V0, R0=0.1):

    # Harmonic trap
    V_trap = 0.5 * np.sum(z**2)

    # Opposite-spin pair distances
    up = z[:n_up]
    down = z[n_up:]

    diff = up[:, None, :] - down[None, :, :]
    r = np.sqrt(np.sum(diff**2, axis=-1))

    # Interaction potential
    q = np.exp(-2.0 * r / R0)
    V_int = -V0 * np.sum(4.0 * q / (1.0 + q)**2)

    return V_trap + V_int

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
def sample_mcmc(logpsi, n_particles, n_samples, n_discard, step_size, n_up, n_chains, Z0=None, RNG=RNG, sweep_step=1, measures=("energy", "V", "acceptance"), r_edges=None, params=onp.zeros(len(S_K)), V0=0.0):

    measures = set(measures)
    allowed = {"energy", "V", "density", "acceptance"}
    if not measures <= allowed:
        raise ValueError(f"Unknown measures: {measures - allowed}")

    results = {}

    # Initialization
    if Z0 is None:
        Z = np.asarray(RNG.normal(size=(n_chains, n_particles, D)))
    else:
        Z = np.asarray(Z0)
        assert Z.shape == (n_chains, n_particles, D)

    logpsi_Z = logpsi(Z, params, n_up)

    # Thermalisation
    for _ in tqdm(range(n_discard * sweep_step), desc="thermalisation"):
        Z, logpsi_Z, _ = metropolis_step(
            logpsi, Z, logpsi_Z, step_size, params, n_up, RNG
        )

    # Initialize only requested measurements
    if "energy" in measures:
        E_mean = onp.zeros(n_chains)
        E_M2 = onp.zeros(n_chains)

        energy = onp.empty(n_samples)
        error = onp.empty(n_samples)
        sigma = onp.empty(n_samples)

    if "V" in measures:
        V = onp.empty((n_samples, n_chains))

    if "density" in measures:
        if r_edges is None:
            r_edges = onp.linspace(0, 5, 101)

        n_bins = len(r_edges) - 1
        radial_hist = onp.zeros((n_chains, n_bins), dtype=int)

    if "acceptance" in measures:
        n_accepted = onp.zeros(n_chains)

    # Sampling
    for i in tqdm(range(n_samples), desc="sampling"):

        for _ in range(sweep_step):
            Z, logpsi_Z, accept = metropolis_step(
                logpsi, Z, logpsi_Z, step_size, params, n_up, RNG
            )

            if "acceptance" in measures:
                n_accepted += accept

        # Energy
        if "energy" in measures:
            eloc = onp.asarray(local_energy(Z, params, n_up, V0))

            delta = eloc - E_mean
            E_mean += delta / (i + 1)
            E_M2 += delta * (eloc - E_mean)

            energy[i] = E_mean.mean()
            error[i] = E_mean.std(ddof=1) / onp.sqrt(n_chains)
            sigma[i] = (
                onp.sqrt(onp.mean(E_M2) / i)
                if i > 0 else onp.nan
            )

        # Potential
        if "V" in measures:
            V[i] = onp.asarray(vmap(lambda z: potential(z, n_up, V0))(Z))

        # Radial density histogram
        if "density" in measures:
            radii = onp.linalg.norm(onp.asarray(Z), axis=-1)

            bins = onp.searchsorted(r_edges, radii, side="right") - 1
            valid = (bins >= 0) & (bins < n_bins)

            indices = (
                onp.arange(n_chains)[:, None] * n_bins + bins
            )[valid]

            radial_hist += onp.bincount(
                indices, minlength=n_chains * n_bins
            ).reshape(n_chains, n_bins)

    # Return only requested measurements
    if "energy" in measures:
        results["energy"] = {
            "mean": energy,
            "error": error,
            "sigma": sigma
        }

    if "V" in measures:
        results["V"] = V

    if "density" in measures:
        results["density"] = {
            "hist": radial_hist,
            "edges": r_edges
        }

    if "acceptance" in measures:
        results["acceptance"] = (
            n_accepted.mean() / (n_samples * sweep_step)
        )

    return results

def n_exact(r):
    return onp.exp(-r**2) / onp.pi * (
        3 + 2*r**2 + 1.5*r**4
    )


def density_from_hist(hist, edges, n_samples):
    areas = onp.pi * (edges[1:]**2 - edges[:-1]**2)
    centers = (edges[:-1] + edges[1:]) / 2

    # Density estimated separately for every chain
    density_chains = hist / (n_samples * areas[None, :])

    density = density_chains.mean(axis=0)
    error = density_chains.std(axis=0, ddof=1) / onp.sqrt(
        hist.shape[0]
    )

    return centers, density, error

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

