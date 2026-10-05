"""Run the configurations of one or more studies in parallel.

    python run.py S1_optimizer S3_fullsum --workers 10

Each worker is a separate single-threaded process (JAX on CPU, one thread): for these
small networks the cost of an iteration is dominated by fixed overheads, so many
single-threaded processes are much faster than one multi-threaded one.
Runs whose result file already exists are skipped, so the script can be interrupted
and restarted.
"""

import argparse
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _init_worker(device="cpu", cpu_queue=None):
    if cpu_queue is not None:
        # Pin the worker to one CPU *before* JAX starts: XLA sizes its thread pool from the
        # affinity mask, so unpinned workers start ~40 threads each and oversubscribe the machine.
        import queue
        from multiprocessing.util import Finalize

        try:
            cpu = cpu_queue.get(timeout=10)
            os.sched_setaffinity(0, {cpu})
            Finalize(None, cpu_queue.put, args=(cpu,), exitpriority=100)  # give the CPU back on exit
        except queue.Empty:
            pass
    os.environ["JAX_PLATFORMS"] = device
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
    os.environ["OMP_NUM_THREADS"] = "1"
    import vmc_lib  # noqa: F401  (import once per worker: jit caches are reused across runs)


_CURRENT_IDS = None


def _current_ids():
    """Run ids of the current study definitions (re-read by every new worker), so that
    configurations removed from studies.py are skipped by an already running queue."""
    global _CURRENT_IDS
    if _CURRENT_IDS is None:
        import studies
        import vmc_lib

        _CURRENT_IDS = {vmc_lib.run_id(c) for f in studies.STUDIES.values() for c in f()}
    return _CURRENT_IDS


def _work(cfg):
    import vmc_lib

    rid = vmc_lib.run_id(cfg)
    path = vmc_lib.result_path(cfg)
    if path.exists() or rid not in _current_ids():
        return rid + " [skipped]", float("nan"), False, 0.0
    # Lock file: several runners can work on the same queue without running a config twice
    lock = path.with_suffix(".running")
    try:
        os.close(os.open(lock, os.O_CREAT | os.O_EXCL))
    except FileExistsError:
        return rid + " [skipped: running elsewhere]", float("nan"), False, 0.0
    try:
        t0 = time.perf_counter()
        meta, _ = vmc_lib.run(cfg)
    finally:
        lock.unlink(missing_ok=True)
    return rid, meta["final"].get("rel_err"), meta["diverged"], time.perf_counter() - t0


def _work_safe(cfg):
    try:
        return _work(cfg)
    except Exception as e:  # report and keep the pool alive
        return f"{cfg}: {e!r}"


def _exact(args):
    import vmc_lib

    vmc_lib.exact_data(*args)
    return args


def n_params_est(cfg):
    if cfg["model"] == "rbm":
        return cfg["alpha"] * cfg["N"] ** 2
    if cfg["model"] == "rbm_symm":
        return cfg["alpha"] * cfg["N"]
    return cfg["width"] * (cfg["N"] + cfg["width"] * (cfg["depth"] - 1))


def is_big(cfg):
    """SR on large networks: dense linear algebra, ~3x faster on the GPU than on one CPU core."""
    return cfg["state"] == "mc" and cfg["sr"] and n_params_est(cfg) >= 2500


def cost(cfg):
    """Rough relative cost, used to start the longest runs first."""
    if cfg["state"] == "full":
        return cfg["n_iter"] * 2 ** cfg["N"] / 4096 * (1.5 if cfg["sr"] else 0.2)
    sr = 1 + cfg["sr"] * n_params_est(cfg) / 1000
    return cfg["n_iter"] * (1 + cfg["n_samples"] / 1024) * (2 if cfg["model"] != "ffnn" else 1) * sr * cfg["N"] / 20


def main():
    import studies
    from vmc_lib import result_path, run_id

    ap = argparse.ArgumentParser()
    ap.add_argument("studies", nargs="+")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--tasks-per-child", type=int, default=15)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--big", default="all", choices=["all", "only", "skip"],
                    help="restrict to (only) or exclude (skip) SR runs on networks with >= 2500 parameters")
    ap.add_argument("--cpus", default="", help="comma-separated CPU ids to pin the workers to (one each)")
    ap.add_argument("--reverse", action="store_true", help="cheapest runs first (to share a queue with a normal runner)")
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()

    cfgs = list({run_id(c): c for s in args.studies for c in studies.STUDIES[s]()}.values())
    todo = [c for c in cfgs if not result_path(c).exists()]
    if args.big != "all":
        todo = [c for c in todo if is_big(c) == (args.big == "only")]
    print(f"{len(cfgs)} configurations, {len(todo)} to run", flush=True)
    if args.dry or not todo:
        return
    todo.sort(key=cost, reverse=not args.reverse)

    ctx = mp.get_context("spawn")
    # Exact diagonalisation first (N=20 needs ~1 GB per process, so fewer workers)
    systems = sorted({(c["N"], c["h"], c["J"]) for c in todo})
    with ProcessPoolExecutor(max_workers=min(3, args.workers), mp_context=ctx, initializer=_init_worker) as ex:
        for s in ex.map(_exact, systems):
            pass
    print(f"exact data ready for {len(systems)} systems", flush=True)

    t0 = time.perf_counter()
    # multiprocessing.Pool can recycle workers (maxtasksperchild), which bounds the memory
    # held by the jit caches of the many different models (~650 MB per worker otherwise)
    cpu_queue = None
    if args.cpus:
        cpu_queue = ctx.Queue()
        for c in args.cpus.split(","):
            cpu_queue.put(int(c))
    with ctx.Pool(args.workers, initializer=_init_worker, initargs=(args.device, cpu_queue),
                  maxtasksperchild=args.tasks_per_child) as pool:
        for i, res in enumerate(pool.imap_unordered(_work_safe, todo), 1):
            if isinstance(res, str):
                print(f"[{i}/{len(todo)}] FAILED {res}", flush=True)
                continue
            rid, err, div, dt = res
            print(f"[{i}/{len(todo)} {time.perf_counter() - t0:7.0f}s] {rid}  rel_err={err:.3e}  "
                  f"{'DIVERGED ' if div else ''}({dt:.0f}s)", flush=True)
    print(f"done in {time.perf_counter() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
