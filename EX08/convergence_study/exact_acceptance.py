"""Acceptance rate of single-spin-flip Metropolis sampling of the *exact* ground state.

    A = (1/N) sum_i sum_s min(p(s), p(s with spin i flipped)),   p = |psi_0|^2

This is the acceptance a perfectly converged Ansatz would show, used as a reference for the
VMC runs. Stored in the exact-data cache (results/exact/*.json) as "acc_local"; N <= 20 only.

    python exact_acceptance.py
"""

import json
import multiprocessing as mp

import numpy as np
import scipy.sparse.linalg


def work(path):
    from vmc_lib import ising_sparse

    d = json.loads(open(path).read())
    if "acc_local" in d or d.get("analytic_only"):
        return path, d.get("acc_local")
    N = d["N"]
    dim = 2**N
    H = ising_sparse(N, d["h"], d["J"])
    # Start from the uniform state: Lanczos then stays in the spin-flip-even sector of the ground state
    val, vec = scipy.sparse.linalg.eigsh(H, k=1, which="SA", v0=np.ones(dim) / np.sqrt(dim), tol=1e-10)
    psi = vec[:, 0]
    psi = 0.5 * (psi + psi[::-1])  # project on the even sector (index i -> 2^N - 1 - i flips all spins)
    psi /= np.linalg.norm(psi)
    assert abs(val[0] - d["E0"]) < 1e-7 * abs(d["E0"]), (path, val[0], d["E0"])
    p = psi**2
    idx = np.arange(dim)
    acc = sum(np.minimum(p, p[idx ^ (1 << i)]).sum() for i in range(N)) / N
    d = json.loads(open(path).read())
    d["acc_local"] = float(acc)
    open(path, "w").write(json.dumps(d))
    return path, float(acc)


if __name__ == "__main__":
    from vmc_lib import EXACT_DIR

    paths = sorted((str(p) for p in EXACT_DIR.glob("*.json")), key=lambda p: -json.loads(open(p).read())["N"])
    with mp.get_context("spawn").Pool(3) as pool:
        for path, acc in pool.imap_unordered(work, paths):
            if acc is not None:
                print(f"{path.split('/')[-1]}: acceptance of the exact ground state = {acc:.4f}", flush=True)
