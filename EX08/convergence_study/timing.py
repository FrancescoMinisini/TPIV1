"""Clean per-iteration cost of every Ansatz of the network-size study, SGD vs SGD+SR.

The timings recorded inside the parallel sweeps are inflated by CPU contention, so this
script runs the configurations one after the other in a single process (one thread),
with nothing else running. Output: results/timing.json

    python timing.py
"""

import json
import sys
import time

import vmc_lib  # noqa: F401  (configures JAX before netket is imported)
import netket as nk

from vmc_lib import RESULTS, build_model, build_system, default_config

sys.path.insert(0, str(RESULTS.parent))
import studies  # noqa: E402


def time_config(cfg, n_iter=20):
    hi, ha = build_system(cfg["N"], cfg["h"])
    sa = nk.sampler.MetropolisLocal(hi, n_chains=cfg["n_chains"])
    vs = nk.vqs.MCState(sa, build_model(cfg), n_samples=cfg["n_samples"], n_discard_per_chain=cfg["n_discard"], seed=0)
    kw = {"preconditioner": nk.optimizer.SR(diag_shift=cfg["diag_shift"])} if cfg["sr"] else {}
    drv = nk.VMC(ha, nk.optimizer.Sgd(cfg["lr"]), variational_state=vs, **kw)
    t0 = time.perf_counter()
    drv.run(1, show_progress=False)
    t_compile = time.perf_counter() - t0
    drv.run(3, show_progress=False)
    t0 = time.perf_counter()
    drv.run(n_iter, show_progress=False)
    return vs.n_parameters, t_compile, (time.perf_counter() - t0) / n_iter


def main():
    out = []
    seen = set()
    for cfg in studies.S4_size():
        key = (cfg["model"], cfg["width"], cfg["depth"], cfg["alpha"], cfg["out_act"], cfg["sr"])
        if key in seen or cfg["out_act"] == "linear":  # the output activation does not change the cost
            continue
        seen.add(key)
        n_par, tc, dt = time_config(cfg, n_iter=10 if cfg["sr"] and cfg["width"] >= 32 else 20)
        row = dict(model=cfg["model"], width=cfg["width"], depth=cfg["depth"], alpha=cfg["alpha"], sr=cfg["sr"],
                   n_params=n_par, t_compile=tc, t_iter=dt)
        out.append(row)
        print(row, flush=True)
    (RESULTS / "timing.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
