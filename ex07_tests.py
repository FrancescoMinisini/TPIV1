"""
Checks for exercise 07 (VMC). They are meant to be run through 1.py:

    python 1.py                 run every quick check
    python 1.py 1b 1d           run only some checks
    python 1.py --list          list the checks and the VMC runs
    python 1.py vmc-mf-mcmc     run a full VMC optimisation and plot it (slow)

A check reports TODO while a function it needs is not defined yet, so you can
run everything at any point while working through the exercise.

The checks compare against exact results computed by brute force on small
systems (all 2^N basis states, dense hamiltonian), not against a reference
solution, so any correct implementation passes.
"""
import argparse
import os
import time
import traceback

import numpy as np


class Missing(Exception):
    """a function the check needs is not defined in 1.py yet"""


class Failed(Exception):
    """the check ran but the result is wrong"""


def require(cond, msg):
    if not cond:
        raise Failed(msg)


class Exercise:
    """attribute access to what 1.py defines, raising Missing for undefined names"""

    def __init__(self, namespace):
        self._ns = namespace

    def __getattr__(self, name):
        try:
            return self._ns[name]
        except KeyError:
            raise Missing(name) from None


# ---------------------------------------------------------------------------
# brute-force references (small N only)
# ---------------------------------------------------------------------------

def random_spins(N, size):
    return np.random.choice(np.array([-1, 1]), size=(size, N))


def all_states(N):
    """all 2^N basis states; row k is the state with index k (bit 0 <-> spin +1)"""
    bits = (np.arange(2**N)[:, None] >> np.arange(N - 1, -1, -1)) & 1
    return 1 - 2 * bits


def state_index(s):
    s = np.rint(np.asarray(s)).astype(int)
    return ((1 - s) // 2) @ (2 ** np.arange(s.shape[-1] - 1, -1, -1))


def dense_ising(N, G, J):
    """H = J sum_i sz_i sz_i+1 - G sum_i sx_i with pbc, as a dense matrix"""
    sz = np.diag([1.0, -1.0])
    sx = np.array([[0.0, 1.0], [1.0, 0.0]])

    def site_op(op, i):
        out = np.ones((1, 1))
        for j in range(N):
            out = np.kron(out, op if j == i else np.eye(2))
        return out

    H = np.zeros((2**N, 2**N))
    for i in range(N):
        H += J * site_op(sz, i) @ site_op(sz, (i + 1) % N) - G * site_op(sx, i)
    return H


def dense_operator(H, N):
    """wrap a dense matrix into the (x_prime, mels) format of ising_hamiltonian,
    connecting every state to all 2^N states"""
    S = all_states(N)

    def operator(x):
        xp = np.tile(S, (len(x), 1, 1))
        mels = H[state_index(x)]
        return xp, mels

    return operator


def exact_psi(logpsi, params, N):
    """normalised wave-function on all basis states"""
    lp = logpsi(params, all_states(N))
    psi = np.exp(lp - lp.real.max())
    return psi / np.linalg.norm(psi)


def exact_energy(logpsi, params, H, N):
    psi = exact_psi(logpsi, params, N)
    return float(np.real(psi.conj() @ H @ psi))


def finite_diff(f, params, eps=1e-6):
    """central finite-difference derivative of f w.r.t. every entry of params;
    the result has shape f(params).shape + params.shape"""
    params = np.asarray(params, dtype=float)
    cols = []
    for k in range(params.size):
        dp = np.zeros(params.size)
        dp[k] = eps
        dp = dp.reshape(params.shape)
        cols.append((np.asarray(f(params + dp)) - np.asarray(f(params - dp))) / (2 * eps))
    return np.stack(cols, axis=-1).reshape(np.shape(cols[0]) + params.shape)


def tv_distance(samples, p):
    """total variation distance between the histogram of samples and p"""
    counts = np.bincount(state_index(samples), minlength=len(p))
    return 0.5 * np.abs(counts / len(samples) - p).sum()


def check_samples(samples, n, N, name):
    require(np.shape(samples) == (n, N),
            f"{name} returned shape {np.shape(samples)}, expected (N_samples, N) = {(n, N)}")
    require(np.isin(samples, [-1, 1]).all(), f"{name} returned entries other than -1 and +1")


def compare_grad(g, fd, name):
    diff = np.abs(g - fd)
    k = np.unravel_index(diff.argmax(), diff.shape)
    param = k[1] if len(k) == 2 else k[1:]
    require(np.allclose(g, fd, rtol=1e-4, atol=1e-5),
            f"gradient disagrees with finite differences of {name}, worst at sample {k[0]}, "
            f"param {param}: got {g[k]:.5f}, finite diff {fd[k]:.5f}")


def fmt(a):
    return np.array2string(np.asarray(a), precision=3, suppress_small=True)


# ---------------------------------------------------------------------------
# checks
# ---------------------------------------------------------------------------

CHECKS = {}
RUNS = {}


def check(key, title):
    def register(fn):
        CHECKS[key] = (title, fn)
        return fn
    return register


def run(key, title):
    def register(fn):
        RUNS[key] = (title, fn)
        return fn
    return register


@check("1a", "mean-field ansatz")
def check_1a(ex):
    N, n = 6, 10
    params = ex.random_params_mf(N, 0.5)
    s = random_spins(N, n)
    lp = ex.logpsi_mf(params, s)
    require(np.shape(lp) == (n,), f"logpsi_mf returned shape {np.shape(lp)}, expected (n_samples,) = ({n},)")
    require(np.all(np.isfinite(lp)), f"logpsi_mf returned non-finite values: {fmt(lp)}")

    one_by_one = np.array([ex.logpsi_mf(params, s[k:k + 1])[0] for k in range(n)])
    require(np.allclose(lp, one_by_one),
            "logpsi_mf on the whole batch differs from logpsi_mf on its rows one at a time")

    # a product state satisfies log psi(s) + log psi(t) = log psi(s') + log psi(t')
    # where s', t' are s, t with the spin at one site swapped
    t = random_spins(N, n)
    for i in range(N):
        s2, t2 = s.copy(), t.copy()
        s2[:, i], t2[:, i] = t[:, i], s[:, i]
        lhs = lp + ex.logpsi_mf(params, t)
        rhs = ex.logpsi_mf(params, s2) + ex.logpsi_mf(params, t2)
        require(np.allclose(lhs, rhs),
                f"logpsi_mf is not a sum of single-site terms (spin {i} depends on the other spins)")


@check("1b", "gradient of the mean-field ansatz")
def check_1b(ex):
    N, n = 5, 8
    params = ex.random_params_mf(N, 0.5)
    s = random_spins(N, n)
    g = ex.grad_logpsi_mf(params, s)
    expected = (n,) + np.shape(params)
    require(np.shape(g) == expected, f"grad_logpsi_mf returned shape {np.shape(g)}, expected {expected}")

    compare_grad(g, finite_diff(lambda p: ex.logpsi_mf(p, s), params), "logpsi_mf")


@check("1c", "direct sampling")
def check_1c(ex):
    N, n = 4, 20000
    params = ex.random_params_mf(N, 0.5)
    samples = ex.sample_direct_mf(ex.logpsi_mf, params, N, n)
    check_samples(samples, n, N, "sample_direct_mf")

    p = np.abs(exact_psi(ex.logpsi_mf, params, N)) ** 2
    tv = tv_distance(samples, p)
    require(tv < 0.03,
            f"samples do not follow |psi|^2 (total variation distance {tv:.3f})\n"
            f"<s_i> sampled: {fmt(samples.mean(0))}\n"
            f"<s_i> exact:   {fmt(p @ all_states(N))}")


@check("1d", "single spin flip + MCMC sampling")
def check_1d(ex):
    x = np.array([1, -1, 1, 1, -1, -1])
    x0 = x.copy()
    y = ex.single_spin_flip(x)
    require(np.array_equal(x, x0), "single_spin_flip modified its input (it should flip a copy)")
    require(np.shape(y) == x.shape, f"single_spin_flip returned shape {np.shape(y)}, expected {x.shape}")
    require((y != x).sum() == 1, f"single_spin_flip changed {(y != x).sum()} spins, expected exactly 1")
    sites = {int(np.flatnonzero(ex.single_spin_flip(x) != x)[0]) for _ in range(300)}
    require(len(sites) == len(x), f"single_spin_flip only ever flips sites {sorted(sites)}")

    N, n = 4, 20000
    params = ex.random_params_mf(N, 0.5)
    samples = ex.sample_mcmc(ex.logpsi_mf, params, N, n, 500)
    check_samples(samples, n, N, "sample_mcmc")

    p = np.abs(exact_psi(ex.logpsi_mf, params, N)) ** 2
    tv = tv_distance(samples, p)
    moves = np.any(samples[1:] != samples[:-1], axis=1).mean()
    require(tv < 0.06,
            f"samples do not follow |psi|^2 (total variation distance {tv:.3f}, "
            f"acceptance rate {moves:.2f})\n"
            f"<s_i> sampled: {fmt(samples.mean(0))}\n"
            f"<s_i> exact:   {fmt(p @ all_states(N))}")


@check("1e", "connected elements (ising_hamiltonian)")
def check_1e(ex):
    N, G, J = 5, 0.7, 1.3
    x = all_states(N)
    x0 = x.copy()
    xp, mels = ex.ising_hamiltonian(x, G, J)
    require(np.array_equal(x, x0), "ising_hamiltonian modified its input x")
    require(np.shape(xp) == (len(x), N + 1, N),
            f"x_prime has shape {np.shape(xp)}, expected {(len(x), N + 1, N)}")
    require(np.shape(mels) == (len(x), N + 1),
            f"mels has shape {np.shape(mels)}, expected {(len(x), N + 1)}")
    require(np.isin(xp, [-1, 1]).all(), "x_prime contains entries other than -1 and +1")

    # rebuild the full matrix from the connected elements and compare to the exact one
    H = np.zeros((2**N, 2**N))
    rows = np.repeat(np.arange(2**N), N + 1)
    np.add.at(H, (rows, state_index(xp.reshape(-1, N))), np.asarray(mels, dtype=float).reshape(-1))
    H_ref = dense_ising(N, G, J)

    d, d_ref = np.diag(H), np.diag(H_ref)
    k = np.abs(d - d_ref).argmax()
    require(np.allclose(d, d_ref),
            f"diagonal elements are wrong (Gamma={G}, J={J}): "
            f"for s = {x[k]} got <s|H|s> = {d[k]:.3f}, expected {d_ref[k]:.3f}")

    off, off_ref = H - np.diag(d), H_ref - np.diag(d_ref)
    k = np.abs(off - off_ref).sum(1).argmax()
    require(np.allclose(off, off_ref),
            f"off-diagonal elements are wrong (Gamma={G}, J={J}) for s = {x[k]}:\n"
            f"got      {len(np.flatnonzero(off[k]))} connected states with elements {fmt(off[k][off[k] != 0])}\n"
            f"expected {len(np.flatnonzero(off_ref[k]))} connected states with elements "
            f"{fmt(off_ref[k][off_ref[k] != 0])}")


@check("1f", "local energy (compute_eloc)")
def check_1f(ex):
    # uses a dense reference hamiltonian, so this does not depend on 1e
    N = 5
    params = ex.random_params_mf(N, 0.5)
    H = dense_ising(N, 0.7, 1.3)
    S = all_states(N)
    eloc = ex.compute_eloc(dense_operator(H, N), ex.logpsi_mf, params, S)
    require(np.shape(eloc) == (len(S),), f"compute_eloc returned shape {np.shape(eloc)}, expected ({len(S)},)")

    psi = np.exp(ex.logpsi_mf(params, S))
    expected = (H @ psi) / psi
    k = np.abs(eloc - expected).argmax()
    require(np.allclose(eloc, expected),
            f"local energy is wrong: for s = {S[k]} got {np.real(eloc[k]):.4f}, expected {expected[k]:.4f}")


def energy_estimate(ex, sampler, n):
    N = 6
    params = ex.random_params_mf(N, 0.5)
    op = lambda x: ex.ising_hamiltonian(x, 1.0, 1.0)
    E_exact = exact_energy(ex.logpsi_mf, params, dense_ising(N, 1.0, 1.0), N)
    x = sampler(params, N, n)
    E = float(np.real(ex.expect(op, ex.logpsi_mf, params, x)))
    err = float(np.std(ex.compute_eloc(op, ex.logpsi_mf, params, x)) / np.sqrt(n))
    return E, err, E_exact


@check("1g", "energy from direct samples")
def check_1g(ex):
    n = 20000
    E, err, E_exact = energy_estimate(
        ex, lambda p, N, n: ex.sample_direct_mf(ex.logpsi_mf, p, N, n), n)
    require(abs(E - E_exact) < 5 * err + 1e-9,
            f"E = {E:.4f} +- {err:.4f}, but the exact energy of this ansatz is {E_exact:.4f}")
    return f"E = {E:.4f} +- {err:.4f}  (exact {E_exact:.4f})"


@check("1h", "energy from MCMC samples")
def check_1h(ex):
    n = 20000
    E, err, E_exact = energy_estimate(
        ex, lambda p, N, n: ex.sample_mcmc(ex.logpsi_mf, p, N, n, 500), n)
    # err ignores autocorrelation, so allow a few times more
    require(abs(E - E_exact) < 15 * err + 1e-9,
            f"E = {E:.4f} +- {err:.4f} (naive error), but the exact energy of this ansatz is {E_exact:.4f}")
    return f"E = {E:.4f} +- {err:.4f}  (exact {E_exact:.4f})"


@check("1i", "energy gradient (expect_and_grad)")
def check_1i(ex):
    N = 4
    params = ex.random_params_mf(N, 0.5)
    H = dense_ising(N, 1.0, 1.0)

    # a batch whose histogram is |psi|^2 up to rounding, so the estimates are almost exact
    p = np.abs(exact_psi(ex.logpsi_mf, params, N)) ** 2
    x = np.repeat(all_states(N), np.round(p * 20000).astype(int), axis=0)
    E, grad = ex.expect_and_grad(dense_operator(H, N), ex.logpsi_mf, ex.grad_logpsi_mf, params, x)
    grad = np.real(np.asarray(grad))
    require(grad.shape == np.shape(params), f"gradient has shape {grad.shape}, expected {np.shape(params)}")

    exact = finite_diff(lambda q: exact_energy(ex.logpsi_mf, q, H, N), params, eps=1e-5)
    g, e = grad.ravel(), exact.ravel()
    cos = g @ e / (np.linalg.norm(g) * np.linalg.norm(e))
    ratio = g @ e / (e @ e)
    info = f"\ngot   {fmt(g)}\nexact {fmt(e)}"
    require(cos > -0.99, "gradient points the wrong way (sign flipped?)" + info)
    require(cos > 0.99, "gradient does not match dE/dtheta" + info)
    if abs(ratio - 0.5) < 0.03:
        return "gradient is exactly half of dE/dtheta (factor 2 missing), fine for SGD, it just halves eta"
    require(abs(ratio - 1) < 0.03, f"gradient is {ratio:.3f} times dE/dtheta" + info)


@check("1i-sgd", "gradient descent step (sgd)")
def check_1i_sgd(ex):
    params, grad = np.random.normal(size=6), np.random.normal(size=6)
    new = ex.sgd(params.copy(), grad, 0.1)
    require(np.shape(new) == params.shape, f"sgd returned shape {np.shape(new)}, expected {params.shape}")
    require(np.allclose(new, params - 0.1 * grad), "sgd(params, grad, eta) should step by -eta * grad")


@check("1j", "autocorrelation function (corr_fn_fft)")
def check_1j(ex):
    # AR(1) process x_t = a x_t-1 + noise has rho(t) = a^t
    a, n = 0.7, 2**16
    noise = np.random.normal(size=n)
    g = np.zeros(n)
    for t in range(1, n):
        g[t] = a * g[t - 1] + noise[t]
    rho = np.real(np.asarray(ex.corr_fn_fft(g)))
    require(rho.ndim == 1 and len(rho) > 10, f"corr_fn_fft returned shape {rho.shape}")
    require(abs(rho[0] - 1) < 1e-6, f"rho[0] should be 1 (normalised), got {rho[0]:.4f}")
    lags = np.arange(6)
    require(np.allclose(rho[lags], a**lags, atol=0.03),
            f"for an AR(1) chain with a={a} expected rho(t) = a^t:\ngot      {fmt(rho[lags])}\n"
            f"expected {fmt(a**lags)}")


def check_jastrow(ex, logpsi, grad_logpsi, random_params, n_params, value_params, expected_values):
    N, n = 6, 10
    params = random_params(N, 0.3)
    require(np.shape(params) == (n_params,), f"random params have shape {np.shape(params)}, expected ({n_params},)")
    s = random_spins(N, n)
    lp = logpsi(params, s)
    require(np.shape(lp) == (n,), f"logpsi returned shape {np.shape(lp)}, expected ({n},)")
    g = grad_logpsi(params, s)
    require(np.shape(g) == (n, n_params), f"grad_logpsi returned shape {np.shape(g)}, expected ({n}, {n_params})")

    compare_grad(g, finite_diff(lambda p: logpsi(p, s), params), "logpsi")

    require(np.allclose(logpsi(params, np.roll(s, 1, axis=1)), lp),
            "logpsi changes when the chain is translated, is the periodic boundary bond missing?")

    states = np.array([[1] * N, [1, -1] * (N // 2)])
    got = logpsi(np.array(value_params), states)
    require(np.allclose(got, expected_values),
            f"with params {value_params}: all-up state gives {got[0]:.3f} (expected {expected_values[0]:.3f}), "
            f"alternating state gives {got[1]:.3f} (expected {expected_values[1]:.3f})")


@check("2a", "nearest-neighbour Jastrow")
def check_2a(ex):
    N, J1 = 6, 0.3
    check_jastrow(ex, ex.logpsi_jastrow_nearest, ex.grad_logpsi_jastrow_nearest,
                  ex.random_params_jastrow_nearest, 1, [J1], [J1 * N, -J1 * N])


@check("2b", "nearest + next-nearest Jastrow")
def check_2b(ex):
    N, J1, J2 = 6, 0.3, -0.2
    check_jastrow(ex, ex.logpsi_jastrow_next_nearest, ex.grad_logpsi_jastrow_next_nearest,
                  ex.random_params_jastrow_next_nearest, 2, [J1, J2], [(J1 + J2) * N, (-J1 + J2) * N])


# ---------------------------------------------------------------------------
# full VMC runs (the plotting cells of the notebook)
# ---------------------------------------------------------------------------

def run_vmc(ex, title, sample_fn, logpsi, grad_logpsi, params, eta, nsteps):
    import matplotlib.pyplot as plt

    ha = lambda x: ex.ising_hamiltonian(x, 1, 1)
    opt = lambda p, g: ex.sgd(p, g, eta)
    _, energies = ex.vmc(ha, sample_fn, opt, logpsi, grad_logpsi, params, ex.N, nsteps)
    E0 = ex.ising1d_energy(ex.N, 1)
    print(f"         final energy {np.real(energies[-1]):.5f}, exact {E0:.5f}, "
          f"relative error: {ex.err_rel(E0, energies[-1]):.2e}")

    plt.figure()
    plt.plot(np.real(energies))
    plt.plot([0, len(energies)], [E0] * 2, color='k')
    plt.xlabel('step')
    plt.ylabel('Energy')
    plt.title(title)
    plt.show()


def direct_sampler(ex):
    return lambda logpsi, params, N: ex.sample_direct_mf(logpsi, params, N, ex.Ns)


def mcmc_sampler(ex):
    return lambda logpsi, params, N: ex.sample_mcmc(logpsi, params, N, ex.Ns, ex.N_discard)


@run("vmc-mf-direct", "7.1i mean-field, direct sampling (1000 steps)")
def run_mf_direct(ex):
    run_vmc(ex, "mean-field, direct sampling", direct_sampler(ex), ex.logpsi_mf, ex.grad_logpsi_mf,
            ex.random_params_mf(ex.N, 0.1), 0.01, 1000)


@run("vmc-mf-mcmc", "7.1i mean-field, MCMC sampling (1000 steps)")
def run_mf_mcmc(ex):
    run_vmc(ex, "mean-field, MCMC sampling", mcmc_sampler(ex), ex.logpsi_mf, ex.grad_logpsi_mf,
            ex.random_params_mf(ex.N, 0.1), 0.01, 1000)


@run("vmc-j1", "7.2a nearest-neighbour Jastrow (300 steps)")
def run_j1(ex):
    run_vmc(ex, "nearest-neighbour Jastrow", mcmc_sampler(ex), ex.logpsi_jastrow_nearest,
            ex.grad_logpsi_jastrow_nearest, ex.random_params_jastrow_nearest(ex.N, 0.1), 0.001, 300)


@run("vmc-j2", "7.2b next-nearest Jastrow (300 steps)")
def run_j2(ex):
    run_vmc(ex, "next-nearest Jastrow", mcmc_sampler(ex), ex.logpsi_jastrow_next_nearest,
            ex.grad_logpsi_jastrow_next_nearest, ex.random_params_jastrow_next_nearest(ex.N, 0.1),
            0.001, 300)


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------

def indent(text, pad):
    return text.replace("\n", "\n" + pad)


def user_traceback(exc, verbose):
    """the traceback, keeping only frames outside this file unless verbose"""
    frames = traceback.extract_tb(exc.__traceback__)
    if not verbose:
        here = os.path.abspath(__file__)
        frames = [f for f in frames if os.path.abspath(f.filename) != here]
    lines = traceback.format_list(frames) + traceback.format_exception_only(type(exc), exc)
    return "".join(lines).rstrip()


def run_check(ex, key, verbose):
    title, fn = CHECKS[key]
    label = f"7.{key:<7} {title:<40}"
    pad = " " * 9
    np.random.seed(1234)
    t0 = time.time()
    try:
        note = fn(ex)
    except Missing as e:
        print(f"  TODO   {label} needs {e}")
        return "todo"
    except Failed as e:
        print(f"  FAIL   {label}\n{pad}{indent(str(e), pad)}")
        return "fail"
    except Exception as e:
        print(f"  ERROR  {label}\n{pad}{indent(user_traceback(e, verbose), pad)}")
        return "error"
    dt = time.time() - t0
    timing = f"({dt:.1f}s)" if dt > 0.5 else ""
    print(f"  PASS   {label} {timing}")
    if note:
        print(f"{pad}{indent(note, pad)}")
    return "pass"


def run_vmc_run(ex, key, verbose):
    title, fn = RUNS[key]
    print(f"  RUN    {key}: {title}")
    np.random.seed(1234)
    try:
        fn(ex)
    except Missing as e:
        print(f"         needs {e} first")
    except Exception as e:
        print("         " + indent(user_traceback(e, verbose), "         "))


def main(namespace, argv=None):
    parser = argparse.ArgumentParser(prog="1.py", description="Checks for exercise 07 (VMC).")
    parser.add_argument("names", nargs="*",
                        help="checks or VMC runs to execute (default: every check, no VMC runs)")
    parser.add_argument("--list", action="store_true", help="list the available checks and VMC runs")
    parser.add_argument("-v", "--verbose", action="store_true", help="show full tracebacks")
    args = parser.parse_args(argv)

    if args.list:
        print("checks:")
        for key, (title, _) in CHECKS.items():
            print(f"  {key:<15} {title}")
        print("VMC runs (slow, open a plot):")
        for key, (title, _) in RUNS.items():
            print(f"  {key:<15} {title}")
        return

    unknown = [n for n in args.names if n not in CHECKS and n not in RUNS]
    if unknown:
        parser.error(f"unknown name(s) {', '.join(unknown)}; see --list")

    ex = Exercise(namespace)
    names = args.names or list(CHECKS)
    results = []
    for name in names:
        if name in CHECKS:
            results.append(run_check(ex, name, args.verbose))
        else:
            run_vmc_run(ex, name, args.verbose)

    if results:
        counts = {r: results.count(r) for r in ("pass", "fail", "error", "todo")}
        print("\n  " + ", ".join(f"{v} {k}" for k, v in counts.items() if v))
