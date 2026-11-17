import numpy as onp
import matplotlib.pyplot as plt
import typing
from math import factorial
# from scipy.special import eval_hermite
from jax import hessian, grad, vmap
from jax import numpy as np
from numpy.polynomial.hermite import herm2poly


rng = onp.random.default_rng(1)
# number of particles
N = 8
n_up = 2
n_down = N - n_up
n_chains = 10
n_samples = 1000

def hermite1d(x, n):
    # c = [0]*n + [1]
    c = [0 for i in range(n)]
    c+=[1]
    
    #creates a polynomial in the hermite basis
    coeff = herm2poly(c)
    #converts from hermite basis to polynomial basis
    H = np.polyval(coeff[::-1], x)
    return H * np.exp(-0.5*x**2) / (
        np.pi**0.25 * np.sqrt(2.0**n * factorial(n))
    )

def hermite2d(nx, ny, x, y):
    return hermite1d(nx, x)*hermite1d(ny,y)

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
        

#Z is the mc samples of dimention (n_chians, N, 2: x,y)
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

for nu, nd in [(1,1), (2,2), (3,3), (6,6)]:
    print(f"N_up={nu}, N_down={nd} | orbitals={occupied_orbitals(nu)} | E0={exact_energy(nu,nd)}")
assert [exact_energy(n,n) for n in (1,2,3,6)] == [2,6,10,28]

# Same-spin swap: one minus sign, no change in |psi|².
X_test = rng.normal(size=(20,6,2))
X_swapped = X_test.copy()
X_swapped[:,[0,1],:] = X_test[:,[1,0],:]
l1 = logpsi(X_test, 3)
l2 = logpsi(X_swapped, 3)
np.testing.assert_allclose(l2, l1, atol=1e-12)
print("Same-spin exchange: sign flips and |psi|² is unchanged.")

    
def metropolis_step(logpsi, Z, logpsi_Z, step_size, n_up, rng=rng):
    Z_prime = Z.copy()
    # Z has shape (n_chains, N, 2)
    Z_prime += rng.uniform(-step_size, step_size, size=Z.shape)
    logpsi_Z_prime = logpsi(Z_prime, n_up)
    R = np.exp(np.minimum(0., 2*(logpsi_Z_prime-logpsi_Z)))
    accept = R > np.asarray(rng.random(Z.shape[0]))
    Z_new = np.where(accept, Z_prime, Z)
    
    logpsi_new = np.where(
        accept, logpsi_Z_prime, logpsi_Z
    )

    return Z_new, logpsi_new, accept
    
# E = -.5 sum(lapl L + grad L^2) + .5*(x^2 +y^2)
def local_energy_single(Z, n_up):
    return kinetic(Z, n_up) + potential(Z)

# vmap works like an iterator over more complex expressions: it batchifies them
def local_energy(Z, n_up):
    return vmap(
        lambda z: local_energy_single(z, n_up)
    )(Z)

def kinetic(Z, n_up):
    f = lambda x: logpsi(x, n_up) 
    gradL = grad(f)(Z)
    hess = hessian(f)(Z)
    lapl = np.trace(hess.reshape(Z.size, Z.size))
    return -0.5 * (lapl + np.sum(gradL**2))

def potential(z):
    return 0.5 * np.sum(z**2)

def sample_mcmc(logpsi, n_particles, n_samples, n_discard, step_size, n_up, n_chains=10, Z0=None, rng=rng):
    # Initial state of every chain
    if Z0 is None:
        Z = np.asarray(rng.normal(size=(n_chains, n_particles, 2)))
    else:
        Z = np.asarray(Z0)
        assert Z.shape == (n_chains, n_particles, 2)
    # Current log wavefunction for each chain
    logpsi_Z = logpsi(Z, n_up)
    samples = onp.empty((n_samples, n_chains, n_particles, 2))
    n_accepted = onp.zeros(n_chains)
    acceptance_rate = onp.zeros(n_chains)
    samples = np.empty((n_samples, n_chains ,n_particles,2))
    
    for i in range(n_discard+n_samples):
        Z_new, logpsi_new, accept = metropolis_step(logpsi, Z, step_size, n_up, rng)
        if i >= n_discard:
            samples[i- n_discard,...] =  Z_new
            n_accepted += accept
            acceptance_rate = n_accepted / n_samples
    
    return samples, acceptance_rate

# Do we need to count the acceptance rate + n_accepted after discarding thermalization? or after


        
        