"""Stability test at N = 80 (outside the cached run framework): translation-invariant RBM,
initial weights of std 0.01, SR (shift 0.01) with learning rate 0.01, 300 iterations.

    python n80_stability_test.py <cpu id> <h> <seed>

Output of the six runs (h = 0.5, 1; seeds 0-2): results/n80_stability_test.txt
"""
import os, sys
cpu, h, seed = int(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3])
os.sched_setaffinity(0, {cpu})
os.environ["JAX_PLATFORMS"] = "cpu"
import warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, jax, netket as nk, jax.numpy as jnp
from vmc_lib import build_system, ising1d_energy
N = 80
hi, ha = build_system(N, h)
E0 = ising1d_energy(N, h)
init = jax.nn.initializers.normal(0.01)
model = nk.models.RBMSymm(symmetries=nk.graph.Chain(N, pbc=True).translation_group(), alpha=2, param_dtype=jnp.float64,
                          kernel_init=init, hidden_bias_init=init, visible_bias_init=init)
vs = nk.vqs.MCState(nk.sampler.MetropolisLocal(hi, n_chains=128), model, n_samples=1024, n_discard_per_chain=4, seed=seed, sampler_seed=seed + 12345)
drv = nk.VMC(ha, nk.optimizer.Sgd(0.01), variational_state=vs, preconditioner=nk.optimizer.SR(diag_shift=0.01))
Es = []
drv.run(300, callback=lambda s, d, dr: (Es.append(float(d["Energy"].mean.real)), np.isfinite(Es[-1]))[1], show_progress=False)
r = np.abs((np.array(Es) - E0) / E0)
print(f"h={h} seed={seed} init std 0.01, lr 0.01: iterations {len(Es)}, rel. error at 0/2/5/20/50/100/200/299: " +
      " ".join(f"{r[i]:.1e}" if i < len(r) else "-" for i in [0, 2, 5, 20, 50, 100, 200, 299]) + f"  (mean of last 20: {r[-20:].mean():.1e})", flush=True)
