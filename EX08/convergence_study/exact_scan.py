"""Exact ground-state observables on a fine grid of fields (reference curves for the transition figures).

For every (N, h): energy, transverse magnetization <sigma^x>, moments of the staggered magnetization,
spin-spin correlations, distribution of m_s, and the acceptance rate that single-spin-flip Metropolis
has when it samples the exact |psi_0|^2. Output: results/exact_scan.json

    python exact_scan.py
"""

import json
import multiprocessing as mp

import numpy as np
import scipy.sparse.linalg

from vmc_lib import RESULTS, ising_sparse

H_FINE = [0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.4,
          1.5, 1.75, 2.0, 2.5, 3.0]
SIZES = [12, 16, 20]


def observables(args):
    N, h = args
    H = ising_sparse(N, h)
    # two lowest states: in the ordered phase they are quasi-degenerate (even/odd under spin flip)
    vals, vecs = scipy.sparse.linalg.eigsh(H, k=2, which="SA", tol=1e-9)
    v = vecs[:, np.argmin(vals)]
    E0 = float(vals.min())
    p = v**2
    # The ground state is even under a global spin flip (index i -> 2^N - 1 - i). Symmetrising p removes
    # the arbitrary mixing with the odd partner when the two are numerically degenerate.
    p = 0.5 * (p + p[::-1])
    p /= p.sum()

    idx = np.arange(2**N)
    spin = lambda i: 2.0 * ((idx >> i) & 1) - 1
    k_ms = np.zeros(2**N, dtype=np.int64)  # N * m_s, an integer in {-N, -N+2, ..., N}
    for i in range(N):
        k_ms += ((-1) ** i) * (2 * ((idx >> i) & 1) - 1)
    ms = k_ms / N
    s0 = spin(0)
    corr = [float(p @ (s0 * spin(r))) for r in range(N // 2 + 1)]
    hist = np.bincount((k_ms + N) // 2, weights=p, minlength=N + 1)
    # Metropolis single-spin flip on |psi_0|^2: A = (1/N) sum_i sum_s min(p(s), p(s with spin i flipped))
    acc = float(np.mean([np.minimum(p, p[idx ^ (1 << i)]).sum() for i in range(N)]))
    return dict(
        N=N, h=h, E0=E0, sx=float((N * corr[1] - E0) / (h * N)),
        ms2=float(p @ ms**2), ms4=float(p @ ms**4), ms_abs=float(p @ np.abs(ms)),
        corr=corr, ms_hist=hist.tolist(), acc=acc,
    )


def main():
    out = []
    ctx = mp.get_context("spawn")
    for N in SIZES:
        tasks = [(N, h) for h in H_FINE]
        with ctx.Pool(3 if N >= 20 else 6) as pool:  # N = 20 needs ~1 GB per process
            for r in pool.imap(observables, tasks):
                out.append(r)
                print(f"N={r['N']} h={r['h']:<5g} E0/N={r['E0'] / r['N']:.6f} sx={r['sx']:.4f} ms2={r['ms2']:.4f} "
                      f"acc={r['acc']:.4f}", flush=True)
        (RESULTS / "exact_scan.json").write_text(json.dumps(out))


if __name__ == "__main__":
    main()
