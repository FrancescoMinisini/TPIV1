"""Core of the convergence study: models, exact reference data and a single VMC run.

Every run is described by a flat config dict (see `default_config`) and writes one
`.npz` file with the per-iteration diagnostics and a final, higher-statistics evaluation.
The module is imported inside single-threaded worker processes (see run.py), so JAX
is configured for CPU and one thread *before* it is imported.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import hashlib
import json
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import flax.linen as nn
import jax
import jax.numpy as jnp
import scipy.sparse
import netket as nk
import numpy as np
import scipy.sparse.linalg

HERE = Path(__file__).resolve().parent

# Persistent compilation cache: identical models/shapes are compiled once for all processes
jax.config.update("jax_compilation_cache_dir", str(HERE / ".jax_cache"))
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
RESULTS = HERE / "results"
EXACT_DIR = RESULTS / "exact"


# ---------------------------------------------------------------- models


class FFNN(nn.Module):
    """MLP of the exercise: log psi(s) = phi(W x^(L-1) + b), x^(l) = ReLU(W x^(l-1) + b).

    out_act="relu" is the Ansatz of Eq. (2); out_act="linear" drops the ReLU on the
    output unit (ablation of the 'dead output' failure mode)."""

    width: int = 5
    n_layers: int = 1
    out_act: str = "relu"

    @nn.compact
    def __call__(self, x):
        for _ in range(self.n_layers):
            x = nn.relu(nn.Dense(self.width, param_dtype=jnp.float64)(x))
        x = nn.Dense(1, param_dtype=jnp.float64)(x)
        if self.out_act == "relu":
            x = nn.relu(x)
        return x.squeeze(-1)


def build_model(cfg):
    if cfg["model"] == "ffnn":
        return FFNN(width=cfg["width"], n_layers=cfg["depth"], out_act=cfg["out_act"])
    if cfg["model"] == "rbm":
        if cfg.get("init_scale") is None:  # NetKet default: normal(0.01) for all RBM parameters
            return nk.models.RBM(alpha=cfg["alpha"], param_dtype=jnp.float64)
        init = jax.nn.initializers.normal(cfg["init_scale"])
        return nk.models.RBM(alpha=cfg["alpha"], param_dtype=jnp.float64, kernel_init=init,
                             hidden_bias_init=init, visible_bias_init=init)
    if cfg["model"] == "rbm_symm":  # translation-invariant RBM (alpha filters, shared over the N translations)
        group = nk.graph.Chain(cfg["N"], pbc=True).translation_group()
        return nk.models.RBMSymm(symmetries=group, alpha=cfg["alpha"], param_dtype=jnp.float64)
    raise ValueError(cfg["model"])


# ---------------------------------------------------------------- config


def default_config(**kw):
    cfg = dict(
        study="",
        # system
        N=20,
        h=1.0,
        J=1.0,
        # Ansatz
        model="ffnn",
        width=5,
        depth=1,
        out_act="relu",
        alpha=1.0,
        # optimizer
        optimizer="sgd",
        lr=0.01,
        sr=False,
        diag_shift=0.01,
        # variational state / sampling
        state="mc",  # "mc" (Metropolis) or "full" (exact sum over the Hilbert space)
        n_samples=1024,
        n_chains=128,
        n_discard=4,
        final_samples=16384,
        # run
        n_iter=1000,
        seed=0,
        # optional fields: None means "model default" and is left out of the run id
        init_scale=None,
    )
    for k, v in kw.items():
        if k not in cfg:
            raise KeyError(k)
        cfg[k] = v
    # Fields that do not apply are normalised so that equivalent runs share the same id
    if cfg["model"] in ("rbm", "rbm_symm"):
        cfg.update(width=0, depth=0, out_act="")
    else:
        cfg["alpha"] = 0.0
    if not cfg["sr"]:
        cfg["diag_shift"] = 0.0
    if cfg["state"] == "full":
        cfg.update(n_samples=0, n_chains=0, n_discard=0, final_samples=0)
    return cfg


def run_id(cfg):
    keys = sorted(k for k in cfg if k != "study" and cfg[k] is not None)
    blob = json.dumps({k: cfg[k] for k in keys}, sort_keys=True)
    h = hashlib.sha1(blob.encode()).hexdigest()[:10]
    if cfg["model"] in ("rbm", "rbm_symm"):
        arch = f"{cfg['model'].replace('_symm', 'sym')}{cfg['alpha']:g}"
    else:
        arch = f"ffnn{cfg['depth']}x{cfg['width']}{'' if cfg['out_act'] == 'relu' else 'lin'}"
    opt = cfg["optimizer"] + (f"+sr{cfg['diag_shift']:g}" if cfg["sr"] else "")
    if cfg.get("init_scale") is not None:
        arch += f"init{cfg['init_scale']:g}"
    return f"N{cfg['N']}_h{cfg['h']:g}_{arch}_{opt}_lr{cfg['lr']:g}_{cfg['state']}{cfg['n_samples'] or ''}_s{cfg['seed']}_{h}"


def result_path(cfg):
    # One pool for all studies: a configuration shared by two studies is run only once
    return RESULTS / "runs" / f"{run_id(cfg)}.npz"


# ---------------------------------------------------------------- system and exact data


def build_system(N, h, J=1.0):
    g = nk.graph.Chain(N, pbc=True)
    hi = nk.hilbert.Spin(s=1 / 2, N=N)
    ha = nk.operator.Ising(hilbert=hi, graph=g, h=h, J=J)
    return hi, ha


def staggered(samples):
    """Staggered magnetization m_s = (1/N) sum_i (-1)^i s_i of each configuration."""
    N = samples.shape[-1]
    return samples @ ((-1.0) ** np.arange(N)) / N


def correlations(samples):
    """C(r) = <s_i s_{i+r}> averaged over i and over the samples, r = 0..N/2."""
    s = samples.reshape(-1, samples.shape[-1])
    N = s.shape[-1]
    return np.array([np.mean(s * np.roll(s, -r, axis=1)) for r in range(N // 2 + 1)])


def ising_sparse(N, h, J=1.0):
    """H = J sum_i s_i s_{i+1} - h sum_i sigma^x_i (PBC) as a CSR matrix, with s_i = 2 bit_i(index) - 1.

    Built directly from bit operations: ~10x less memory than operator.to_sparse() at N = 20."""
    dim = 2**N
    idx = np.arange(dim, dtype=np.int32)
    diag = np.zeros(dim)
    for i in range(N):
        si = 2.0 * ((idx >> i) & 1) - 1
        sj = 2.0 * ((idx >> ((i + 1) % N)) & 1) - 1
        diag += J * si * sj
    indices = np.empty((dim, N + 1), dtype=np.int32)
    data = np.full((dim, N + 1), -h)
    for i in range(N):
        indices[:, i] = idx ^ (1 << i)
    indices[:, N] = idx
    data[:, N] = diag
    indptr = np.arange(0, dim * (N + 1) + 1, N + 1, dtype=np.int64)
    return scipy.sparse.csr_matrix((data.ravel(), indices.ravel(), indptr), shape=(dim, dim))


def ising1d_energy(L, Gamma):
    """Analytical ground-state energy of the 1D TFIM with PBC (free fermions), Gamma in units of J.

    The k values pi(2i+1)/L are those of the even-parity (antiperiodic) fermion sector, which
    contains the ground state for even L. For even L the antiferromagnet (J > 0) has the same
    spectrum as the ferromagnet (rotate every second spin), so this applies to our H as well."""

    def Epsilon(k, h):
        eps = 1 + h**2 + 2 * h * np.cos(k)
        return 2 * np.sqrt(eps)

    i = np.arange(L)
    k = np.pi * (2 * i + 1) / L
    energy = Epsilon(k, Gamma).sum()
    return -0.5 * energy


def analytic_gap_sym(L, Gamma):
    """Gap in the sector of the constant state (k = 0, even parity): a pair of fermions at
    k = +-pi/L, i.e. 2 eps(pi/L) with eps(k) = 2 sqrt(1 + Gamma^2 - 2 Gamma cos k)."""
    return 4 * np.sqrt(1 + Gamma**2 - 2 * Gamma * np.cos(np.pi / L))


def symmetric_sector_levels(N, h, J=1.0, k=3):
    """Lowest levels in the sector of |+...+>: translation invariant, spin-flip even, reflection even.

    Lanczos on P H P + C (1 - P), with P the projector on that sector, so that the levels of
    the other sectors are pushed above C. This is the sector the optimisation starts in when the
    initial wave function is (nearly) constant, e.g. the RBM with its small initial weights."""
    H = ising_sparse(N, h, J)
    dim = 2**N
    idx = np.arange(dim, dtype=np.int64)
    rot = ((idx << 1) | (idx >> (N - 1))) & (dim - 1)  # translation by one site
    rev = np.zeros(dim, dtype=np.int64)  # reflection i -> N-1-i
    for i in range(N):
        rev |= ((idx >> i) & 1) << (N - 1 - i)
    flip = idx ^ (dim - 1)

    def proj(v):
        acc = np.zeros_like(v)
        w = v.copy()
        for _ in range(N):
            acc += w
            w = w[rot]
        acc /= N
        acc = 0.5 * (acc + acc[flip])
        return 0.5 * (acc + acc[rev])

    C = 10.0 * (abs(J) + abs(h)) * N
    op = scipy.sparse.linalg.LinearOperator(
        (dim, dim), matvec=lambda v: proj(H @ proj(v)) + C * (v - proj(v)), dtype=float
    )
    v0 = np.ones(dim) / np.sqrt(dim)
    vals = scipy.sparse.linalg.eigsh(op, k=k, which="SA", v0=v0, tol=1e-9, return_eigenvectors=False)
    return np.sort(vals)


def exact_data(N, h, J=1.0, k=6):
    """Lowest eigenvalues, gaps and ground-state observables (cached on disk)."""
    EXACT_DIR.mkdir(parents=True, exist_ok=True)
    path = EXACT_DIR / f"N{N}_h{h:g}_J{J:g}.json"
    if path.exists():
        return json.loads(path.read_text())
    if N > 22:  # too large for exact diagonalisation: free-fermion solution (energy and gap only)
        nan = float("nan")
        out = dict(N=N, h=h, J=J, E0=J * ising1d_energy(N, h / J), gap_sym=J * analytic_gap_sym(N, h / J),
                   gap1=nan, gap_energy=nan, corr=[nan] * (N // 2 + 1), ms2=nan, ms_abs=nan,
                   ms_hist=[nan] * (N + 1), analytic_only=True)
        path.write_text(json.dumps(out))
        return out
    H = ising_sparse(N, h, J)
    vals, vecs = scipy.sparse.linalg.eigsh(H, k=k, which="SA", tol=1e-10)
    order = np.argsort(vals)
    vals, vecs = vals[order], vecs[:, order]
    idx = np.arange(2**N)
    states = (2 * ((idx[:, None] >> np.arange(N)) & 1) - 1).astype(np.int8)
    # Flipping all spins maps basis index i -> 2^N - 1 - i
    parity = [float(vecs[:, n] @ vecs[::-1, n]) for n in range(k)]
    plus = np.ones(H.shape[0]) / np.sqrt(H.shape[0])
    overlap_plus = [float(abs(plus @ vecs[:, n])) for n in range(k)]
    p0 = vecs[:, 0] ** 2
    ms = staggered(states.astype(float))
    corr = [float(p0 @ (states[:, 0] * states[:, r])) for r in range(N // 2 + 1)]
    E0 = float(vals[0])
    gaps = [float(v - E0) for v in vals[1:]]
    # Gap that governs the energy convergence: first level not quasi-degenerate with E0
    gap_energy = next((g_ for g_ in gaps if g_ > 1e-3 * max(1.0, abs(E0) / N)), float("nan"))
    # Gap within the sector of |+...+> (translation invariant, Z2-even, reflection-even)
    sym = symmetric_sector_levels(N, h, J)
    gap_sym = float(sym[1] - sym[0])
    out = dict(
        N=N, h=h, J=J, E0=E0, evals=[float(v) for v in vals], gaps=gaps, parity=parity,
        overlap_plus=overlap_plus, gap1=gaps[0], gap_energy=gap_energy, gap_sym=gap_sym,
        corr=corr, ms2=float(p0 @ ms**2), ms_abs=float(p0 @ np.abs(ms)),
        ms_hist=np.histogram(ms, bins=np.linspace(-1 - 1 / N, 1 + 1 / N, N + 2), weights=p0)[0].tolist(),
    )
    path.write_text(json.dumps(out))
    return out


# ---------------------------------------------------------------- helpers


@jax.jit
def tree_norm(tree):
    return jnp.sqrt(sum(jnp.sum(jnp.abs(x) ** 2) for x in jax.tree_util.tree_leaves(tree)))


def autocorr(chains, max_lag=40):
    """Normalised autocorrelation along each chain, averaged over chains; chains: (n_chains, L)."""
    x = chains - chains.mean(axis=1, keepdims=True)
    var = np.mean(x**2)
    L = x.shape[1]
    if var == 0:
        return np.full(max_lag + 1, np.nan)
    return np.array([np.mean(x[:, : L - t] * x[:, t:]) / var for t in range(min(max_lag, L - 1) + 1)])


def make_optimizer(cfg):
    if cfg["optimizer"] == "sgd":
        return nk.optimizer.Sgd(cfg["lr"])
    if cfg["optimizer"] == "adam":
        return nk.optimizer.Adam(cfg["lr"])
    raise ValueError(cfg["optimizer"])


# ---------------------------------------------------------------- one run


def run(cfg, save=True):
    t_start = time.perf_counter()
    N, h = cfg["N"], cfg["h"]
    hi, ha = build_system(N, h, cfg["J"])
    ex = exact_data(N, h, cfg["J"])
    model = build_model(cfg)
    seed = cfg["seed"]

    if cfg["state"] == "full":
        vs = nk.vqs.FullSumState(hi, model, seed=seed)
    else:
        sa = nk.sampler.MetropolisLocal(hi, n_chains=cfg["n_chains"])
        vs = nk.vqs.MCState(
            sa, model, n_samples=cfg["n_samples"], n_discard_per_chain=cfg["n_discard"],
            seed=seed, sampler_seed=seed + 12345,
        )
    kw = {}
    if cfg["sr"] and cfg["state"] == "full":
        kw["preconditioner"] = nk.optimizer.SR(
            qgt=nk.optimizer.qgt.QGTJacobianDense, solver=nk.optimizer.solver.cholesky, diag_shift=cfg["diag_shift"]
        )
    elif cfg["sr"]:
        kw["preconditioner"] = nk.optimizer.SR(diag_shift=cfg["diag_shift"])
    driver = nk.VMC(ha, make_optimizer(cfg), variational_state=vs, **kw)

    rec = {k: [] for k in ["E", "E_err", "E_var", "R_hat", "tau", "acc", "grad_norm", "upd_norm", "par_norm", "t"]}
    status = {"diverged": False}

    def callback(step, log_data, drv):
        st = log_data["Energy"]
        E = float(st.mean.real)
        rec["E"].append(E)
        rec["E_err"].append(float(st.error_of_mean))
        rec["E_var"].append(float(st.variance))
        rec["R_hat"].append(float(st.R_hat))
        rec["tau"].append(float(st.tau_corr))
        rec["acc"].append(float(log_data["acceptance"]) if "acceptance" in log_data else np.nan)
        rec["grad_norm"].append(float(tree_norm(drv._loss_grad)))
        rec["upd_norm"].append(float(tree_norm(drv._dp)))
        rec["par_norm"].append(float(tree_norm(drv.state.parameters)))
        rec["t"].append(time.perf_counter())
        if not np.isfinite(E) or not np.isfinite(rec["upd_norm"][-1]):
            status["diverged"] = True
            return False
        return True

    error = ""
    try:
        driver.run(n_iter=cfg["n_iter"], callback=callback, show_progress=False)
    except Exception as e:  # e.g. NaNs inside the linear solver
        error = repr(e)[:500]
        status["diverged"] = True

    out = {k: np.asarray(v, dtype=float) for k, v in rec.items()}
    t = out.pop("t")
    t_iter = np.diff(np.concatenate([[t_start], t])) if len(t) else np.array([])
    out["t_iter"] = t_iter
    # Iteration 0 contains the jit compilation: report the steady-state time separately
    meta = dict(
        cfg=cfg, exact=ex, n_params=int(vs.n_parameters), error=error, diverged=status["diverged"],
        n_done=len(t), time_first_iter=float(t_iter[0]) if len(t_iter) else np.nan,
        time_per_iter=float(np.median(t_iter[1:])) if len(t_iter) > 1 else np.nan,
        time_train=float(t[-1] - t_start) if len(t) else np.nan,
    )

    # ---- final evaluation with more statistics
    t0 = time.perf_counter()
    if not status["diverged"]:
        try:
            if cfg["state"] == "full":
                psi = np.asarray(vs.to_array())
                p = np.abs(psi) ** 2
                states = np.asarray(hi.all_states())
                ms = staggered(states)
                Hpsi = ha.to_sparse() @ psi
                E = float(np.real(np.vdot(psi, Hpsi)))
                fin = dict(
                    E=E, E_err=0.0, E_var=float(np.real(np.vdot(Hpsi, Hpsi)) - E**2), R_hat=np.nan, tau=np.nan,
                    ms2=float(p @ ms**2), ms_abs=float(p @ np.abs(ms)), ms_mean=float(p @ ms), ms_R_hat=np.nan,
                    corr=[float(p @ (states[:, 0] * states[:, r])) for r in range(N // 2 + 1)],
                    ms_hist=np.histogram(ms, bins=np.linspace(-1 - 1 / N, 1 + 1 / N, N + 2), weights=p)[0].tolist(),
                    acc=np.nan,
                )
            else:
                vs.n_discard_per_chain = 64
                vs.n_samples = cfg["final_samples"]
                if N > 40:  # evaluate the local energies in chunks (16k samples x N+1 connected states)
                    vs.chunk_size = 1024
                samples = np.asarray(vs.samples).reshape(cfg["n_chains"], -1, N)
                E_loc = np.asarray(vs.local_estimators(ha).real).reshape(cfg["n_chains"], -1)
                ms = staggered(samples)
                sE = nk.stats.statistics(E_loc)
                sM = nk.stats.statistics(ms)
                sM2 = nk.stats.statistics(ms**2)
                fin = dict(
                    E=float(sE.mean.real), E_err=float(sE.error_of_mean), E_var=float(sE.variance),
                    R_hat=float(sE.R_hat), tau=float(sE.tau_corr),
                    ms2=float(sM2.mean.real), ms2_err=float(sM2.error_of_mean), ms_abs=float(np.mean(np.abs(ms))),
                    ms_mean=float(sM.mean.real), ms_R_hat=float(sM.R_hat), ms_tau=float(sM.tau_corr),
                    corr=correlations(samples).tolist(),
                    ms_hist=np.histogram(ms.ravel(), bins=np.linspace(-1 - 1 / N, 1 + 1 / N, N + 2))[0].tolist(),
                    acc=float(vs.sampler_state.acceptance),
                    acf_Eloc=autocorr(E_loc).tolist(), acf_ms=autocorr(ms).tolist(),
                )
            fin["rel_err"] = abs((fin["E"] - ex["E0"]) / ex["E0"])
        except Exception as e:
            fin = dict(E=np.nan, rel_err=np.nan)
            meta["error"] += " | final: " + repr(e)[:300]
    else:
        fin = dict(E=np.nan, rel_err=np.nan)
    meta["final"] = fin
    meta["time_final"] = time.perf_counter() - t0

    # ---- exact imaginary-time evolution from the same initial state (full-sum runs only)
    if cfg["state"] == "full" and cfg["sr"]:
        H = ha.to_sparse()
        psi0 = np.asarray(nk.vqs.FullSumState(hi, model, seed=seed).to_array()).real
        # SR with lr = eta is an Euler step of imaginary time dtau = 2 eta (the energy gradient
        # of a real Ansatz is 2 Re<O* (E_loc - E)>), checked against the SR runs in the pilot
        psi = psi0 / np.linalg.norm(psi0)
        Es = np.empty(cfg["n_iter"])
        for i in range(cfg["n_iter"]):
            Hpsi = H @ psi
            Es[i] = psi @ Hpsi
            psi = psi - 2 * cfg["lr"] * (Hpsi - Es[i] * psi)
            psi /= np.linalg.norm(psi)
        out["ite_E"] = Es

    if save:
        path = result_path(cfg)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, meta=json.dumps(meta, default=float), **out)
        os.replace(tmp, path)
    return meta, out


def load(path):
    d = np.load(path, allow_pickle=False)
    meta = json.loads(str(d["meta"]))
    arrays = {k: d[k] for k in d.files if k != "meta"}
    return meta, arrays
