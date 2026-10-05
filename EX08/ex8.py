import os
from typing import Any, Tuple
from flax import nnx
import flax.linen as nn
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import netket as nk
import numpy as np

from netket.optimizer import SR

os.environ["JAX_PLATFORM_NAME"] = "cpu"


# Setup

# Number of spins
N = 20
# Number of samples
Ns = 1024
# How many Markov chains to run in parallel
n_chains = 16
# How many samples to discard
n_discard = 128

# Graph
g = nk.graph.Chain(N, pbc=True)
# Hilbert space
hi = nk.hilbert.Spin(s=1 / 2, N=g.n_nodes)
# Hamiltonian
ha = nk.operator.Ising(hilbert=hi, graph=g, h=1)
# Sampler (single-spin flips)
sa = nk.sampler.MetropolisLocal(hi, n_chains=n_chains)
# Optimizer
op = nk.optimizer.Sgd(0.01)

# Folder where the figures are saved
os.makedirs("plots", exist_ok=True)

# Analytical solution for the ground state energy of TFIM in 1D with PBC
# (same as past week)
# Gamma (transverse field) is in units of the interaction strength J
def ising1d_energy(L, Gamma):
    def Epsilon(k, h):
        eps = 1 + h**2 + 2 * h * np.cos(k)
        return 2 * np.sqrt(eps)

    i = np.arange(L)
    k = np.pi * (2 * i + 1) / L
    energy = Epsilon(k, Gamma).sum()
    return -0.5 * energy

# Exact diagonalization (lanczos_ed returns an array of eigenvalues)
# E0 = nk.exact.lanczos_ed(ha)[0]
E0 = ising1d_energy(L= N, Gamma=1)
print("Exact ground-state energy:", E0)

results=[]
x = np.linspace(0,2,0.1)
for i in x:
    results.append(ising1d_energy(len(x),i))
    
plt.plot(x , results, true true)
    

class FFNN(nn.Module):
    # Number of hidden neurons per hidden layer
    width: int = 5
    # Number of hidden layers
    n_layers: int = 1

    @nn.compact
    def __call__(self, x):
        # x has shape (n_samples, N)
        # Hidden layers: x^(l) = ReLU(W^(l-1) x^(l-1) + b^(l))
        for _ in range(self.n_layers):
            x = nn.relu(nn.Dense(self.width)(x))
        # Output layer with a single unit: log psi(s) = ReLU(W x + b)
        x = nn.relu(nn.Dense(1)(x))
        # (n_samples, 1) -> (n_samples,)
        return x.squeeze(-1)
    
ma = FFNN(width=N // 4, n_layers=1)
# Variational state object
vs = nk.vqs.MCState(sa, ma, n_samples=Ns, n_discard_per_chain=n_discard)
# VMC optimization driver
vmc = nk.VMC(ha, op, variational_state=vs, preconditioner=SR(diag_shift=0.01))
# Log the energy at every iteration to plot the convergence
log = nk.logging.RuntimeLog()
vmc.run(n_iter=1000, out=log)

ffnn_energy = vs.expect(ha)
error = abs((ffnn_energy.mean - E0) / E0)
print("Optimized energy and relative error: ", ffnn_energy, error)

data = log.data["Energy"]
iters = np.asarray(data.iters)
E_mean = np.asarray(data.Mean.real)
E_sigma = np.asarray(data.Sigma)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

ax1.errorbar(iters, E_mean, yerr=E_sigma, lw=1, label=f"FFNN (width={ma.width}, layers={ma.n_layers})")
ax1.axhline(E0, color="k", ls="--", label="Exact")
ax1.set_xlabel("Iteration")
ax1.set_ylabel("Energy")
ax1.legend()

ax2.semilogy(iters, np.abs((E_mean - E0) / E0))
ax2.set_xlabel("Iteration")
ax2.set_ylabel(r"$|E - E_0| / |E_0|$")

fig.tight_layout()
fig.savefig("plots/ex08_1b_convergence.png", dpi=150)
plt.show()

