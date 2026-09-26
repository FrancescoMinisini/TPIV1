"""
Diagnostic plots for exercise 07 (VMC). Run them through 1.py:

    python 1.py plots                    build every figure
    python 1.py plot-convergence         build a single one
    python 1.py plots --save             write PNGs into plots/ instead of showing them
    python 1.py --list                   list the figures

Every number plotted here comes out of the functions in 1.py, so the figures
show your own code. Each figure prints a line saying what to look at.

Building all of them takes a few minutes: the VMC runs are shared between
figures and cached, but there are a lot of Markov chains to walk.
"""
import time

import numpy as np
import matplotlib.pyplot as plt

from EX07.ex07_tests import plot, all_states, dense_ising

# ---------------------------------------------------------------------------
# colours and styling
# ---------------------------------------------------------------------------

# categorical slots: one per ansatz, in fixed order. these four are checked for
# colour-blind separation on a light background, and every series also gets its
# own dash pattern so the curves are never told apart by colour alone
BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"
# ordinal ramp, light -> dark: for quantities that have an order (k, Gamma, step)
RAMP = ("#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281")
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
SOLID, DASH, DOT, DASHDOT = (0, ()), (0, (6, 2)), (0, (1.5, 1.5)), (0, (5, 2, 1, 2))

N_SITES, N_SAMPLES, N_DISCARD = 16, 512, 128

# the mean-field ansatz starts out symmetric and needs a few hundred steps before it
# breaks the symmetry and drops to its real minimum, so it gets a longer run
STEPS_MF, STEPS_JASTROW = 1000, 400


def figure(name, ncols=1, height=4.3):
    fig, axes = plt.subplots(1, ncols, figsize=(5.8 * ncols, height), num=name,
                             constrained_layout=True)
    fig.patch.set_facecolor("#fcfcfb")
    axes = np.atleast_1d(axes)
    for ax in axes:
        ax.set_facecolor("#fcfcfb")
    return fig, axes


def style(ax, xlabel="", ylabel="", title=""):
    ax.set_xlabel(xlabel, color=MUTED, fontsize=10)
    ax.set_ylabel(ylabel, color=MUTED, fontsize=10)
    if title:
        ax.set_title(title, color=INK, fontsize=11, loc="left")
    ax.grid(True, alpha=0.6, linewidth=0.6, color=GRID)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=9)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


def legend(ax, **kw):
    leg = ax.legend(frameon=False, fontsize=9, labelcolor=MUTED, **kw)
    return leg


def exact_line(ax, E0, label="exact"):
    """horizontal reference line for the exact ground-state energy"""
    ax.axhline(E0, color=INK, linewidth=1.1, linestyle=(0, (4, 3)), zorder=1)
    ax.annotate(label, xy=(0.99, E0), xycoords=("axes fraction", "data"),
                ha="right", va="bottom", fontsize=9, color=INK)


def running_mean(y, window):
    if len(y) < window:
        return np.asarray(y, dtype=float)
    c = np.cumsum(np.insert(np.asarray(y, dtype=float), 0, 0.0))
    return (c[window:] - c[:-window]) / window


# ---------------------------------------------------------------------------
# the ansaetze, and a VMC loop that records diagnostics along the way
# ---------------------------------------------------------------------------

class Ansatz:
    def __init__(self, name, logpsi, grad_logpsi, params0, eta, color, dash, sampler="mcmc"):
        self.name = name
        self.logpsi = logpsi
        self.grad_logpsi = grad_logpsi
        self.params0 = params0
        self.eta = eta
        self.color = color
        self.dash = dash
        self.sampler = sampler


def mean_field(ex, sampler="direct", color=BLUE, dash=SOLID):
    return Ansatz(f"mean-field ({sampler})", ex.logpsi_mf, ex.grad_logpsi_mf,
                  lambda N: ex.random_params_mf(N, 0.1), 0.05, color, dash, sampler)


def jastrow1(ex, color=AQUA, dash=DOT):
    return Ansatz("Jastrow J1", ex.logpsi_jastrow_nearest, ex.grad_logpsi_jastrow_nearest,
                  lambda N: ex.random_params_jastrow_nearest(N, 0.1), 0.005, color, dash)


def jastrow2(ex, color=VIOLET, dash=DASHDOT):
    return Ansatz("Jastrow J1+J2", ex.logpsi_jastrow_next_nearest,
                  ex.grad_logpsi_jastrow_next_nearest,
                  lambda N: ex.random_params_jastrow_next_nearest(N, 0.1), 0.005, color, dash)


def neighbour_corr(s, d):
    """sum_i s_i s_(i+d) with pbc, same as in 1.py"""
    return np.sum(s * np.roll(s, -d, axis=1), axis=1)


def jastrow_k(k, color=BLUE, dash=SOLID):
    """Jastrow with k neighbour shells: log psi = sum_d J_d sum_i s_i s_(i+d).
    k=1 is exercise 7.2a, k=2 is 7.2b"""
    def logpsi(params, s):
        return sum(params[d] * neighbour_corr(s, d + 1) for d in range(k))

    def grad_logpsi(params, s):
        return np.stack([neighbour_corr(s, d + 1) for d in range(k)], axis=1)

    return Ansatz(f"Jastrow k={k}", logpsi, grad_logpsi,
                  lambda N: np.random.normal(0, 0.1, size=k), 0.005, color, dash)


_cache = {}


def optimize(ex, ansatz, N=N_SITES, Ns=N_SAMPLES, nsteps=400, N_discard=N_DISCARD,
             G=1.0, J=1.0, sampler=None, eta=None, seed=0, params_init=None, tag=""):
    """the VMC loop of 7.1i, recording energy, variance, parameters and acceptance
    at every step (the vmc() in 1.py only keeps the energies).
    params_init starts from given parameters instead of random ones"""
    sampler = sampler or ansatz.sampler
    eta = ansatz.eta if eta is None else eta
    key = (ansatz.name, N, Ns, nsteps, N_discard, G, J, sampler, eta, seed, tag)
    if key in _cache:
        return _cache[key]

    print(f"         optimising {ansatz.name}: {nsteps} steps, N={N}, Gamma={G}, eta={eta} ... ",
          end="", flush=True)
    np.random.seed(seed)
    params = ansatz.params0(N) if params_init is None else np.array(params_init, dtype=float)
    operator = lambda x: ex.ising_hamiltonian(x, G, J)
    E_hist, var_hist, par_hist, acc_hist = [], [], [], []
    t0 = time.time()

    for step in range(nsteps):
        x = draw(ex, ansatz, params, N, Ns, N_discard, sampler)
        eloc = np.real(ex.compute_eloc(operator, ansatz.logpsi, params, x))
        E, grad = ex.expect_and_grad(operator, ansatz.logpsi, ansatz.grad_logpsi, params, x)
        E_hist.append(float(np.real(E)))
        var_hist.append(float(eloc.var()))
        par_hist.append(np.array(params, dtype=float))
        acc_hist.append(float(np.any(x[1:] != x[:-1], axis=1).mean()))
        params = ex.sgd(params, grad, eta)

    out = dict(E=np.array(E_hist), var=np.array(var_hist), params_hist=np.array(par_hist),
               params=params, x=x, eloc=eloc, acc=np.array(acc_hist),
               seconds=time.time() - t0, ansatz=ansatz, N=N, G=G, J=J, sampler=sampler)
    print(f"{out['seconds']:.1f}s, E = {out['E'][-1]:.4f}")
    _cache[key] = out
    return out


def draw(ex, ansatz, params, N, Ns, N_discard, sampler):
    if sampler == "direct":
        return ex.sample_direct_mf(ansatz.logpsi, params, N, Ns)
    return ex.sample_mcmc(ansatz.logpsi, params, N, Ns, N_discard)


def converged(h, last=50):
    """energy averaged over the last steps of a run, with its spread"""
    tail = h["E"][-last:]
    return float(tail.mean()), float(tail.std())


def standard_runs(ex):
    """the four runs most figures are built from (computed once, then cached)"""
    return {
        "mean-field (direct)": optimize(ex, mean_field(ex, "direct", BLUE, SOLID), nsteps=STEPS_MF),
        "mean-field (MCMC)": optimize(ex, mean_field(ex, "mcmc", ORANGE, DASH), nsteps=STEPS_MF),
        "Jastrow J1": optimize(ex, jastrow1(ex), nsteps=STEPS_JASTROW),
        "Jastrow J1+J2": optimize(ex, jastrow2(ex), nsteps=STEPS_JASTROW),
    }


def local_energies(ex, ansatz, params, N, n, N_discard=500, G=1.0, J=1.0, sampler=None):
    """a long stream of local energies at fixed parameters"""
    sampler = sampler or ansatz.sampler
    operator = lambda x: ex.ising_hamiltonian(x, G, J)
    x = draw(ex, ansatz, params, N, n, N_discard, sampler)
    return np.real(ex.compute_eloc(operator, ansatz.logpsi, params, x)), x


# ---------------------------------------------------------------------------
# the figures
# ---------------------------------------------------------------------------

@plot("plot-convergence", "energy vs optimisation step")
def plot_convergence(ex):
    """The learning curve. Mean-field sits on a plateau for a few hundred steps (the
    symmetric starting point is a saddle) before it drops, and then stops well above
    the exact energy: that remaining gap is the ansatz, not the optimiser. The
    Jastrow runs converge in a few tens of steps. Direct and MCMC sampling follow the
    same path, MCMC is just noisier."""
    runs = standard_runs(ex)
    E0 = ex.ising1d_energy(N_SITES, 1.0)
    fig, (ax1, ax2) = figure("energy convergence", 2)

    for label, h in runs.items():
        a = h["ansatz"]
        smooth = running_mean(h["E"], 20)
        ax1.plot(h["E"], color=a.color, lw=0.8, alpha=0.25)
        ax1.plot(np.arange(len(smooth)) + 10, smooth, color=a.color, ls=a.dash, lw=1.8, label=label)
        ax2.semilogy(np.abs((h["E"] - E0) / E0), color=a.color, lw=0.7, alpha=0.2)
        rel = np.abs((running_mean(h["E"], 20) - E0) / E0)
        ax2.semilogy(np.arange(len(rel)) + 10, rel, color=a.color, ls=a.dash, lw=1.8, label=label)

    exact_line(ax1, E0, "exact ground state")
    # the first few steps run far above the rest, they would flatten everything else
    ax1.set_ylim(E0 - 0.3, E0 + 3.6)
    ax2.set_ylim(3e-3, 1.0)
    style(ax1, "optimisation step", "energy", "Energy (thin line: raw, thick: 20-step mean)")
    legend(ax1, loc="upper right")
    style(ax2, "optimisation step", "relative error", "Relative error to the exact energy")
    legend(ax2, loc="upper right")

    print("         final energies (mean of the last 50 steps):")
    for label, h in runs.items():
        E, sd = converged(h)
        print(f"           {label:<22} {E:8.4f} +- {sd:.4f}   rel. error {abs((E - E0) / E0):.4f}")
    print(f"           {'exact':<22} {E0:8.4f}")


@plot("plot-variance", "variance of the local energy (zero-variance principle)")
def plot_variance(ex):
    """If the ansatz were the exact ground state, every sample would give the same
    local energy, so Var(Eloc) -> 0. The variance is therefore a quality measure
    that needs no exact solution. Right: energy against variance, extrapolated to
    zero variance - it lands close to the exact energy."""
    runs = standard_runs(ex)
    E0 = ex.ising1d_energy(N_SITES, 1.0)
    fig, (ax1, ax2) = figure("energy variance", 2)

    for label, h in runs.items():
        a = h["ansatz"]
        smooth = running_mean(h["var"], 20)
        ax1.semilogy(h["var"], color=a.color, lw=0.8, alpha=0.25)
        ax1.semilogy(np.arange(len(smooth)) + 10, smooth, color=a.color, ls=a.dash, lw=1.8, label=label)

    style(ax1, "optimisation step", "Var($E_{loc}$)", "Variance of the local energy")
    legend(ax1, loc="upper right")

    # one converged (variance, energy) point per ansatz, then a straight line through them
    pts = []
    for label, h in runs.items():
        a = h["ansatz"]
        E, sd = converged(h)
        v = float(h["var"][-50:].mean())
        pts.append((v, E))
        ax2.errorbar(v, E, yerr=sd, marker="o", markersize=7, color=a.color,
                     capsize=3, lw=1.4, label=label)

    pts = np.array(sorted(pts))
    slope, intercept = np.polyfit(pts[:, 0], pts[:, 1], 1)
    xs = np.linspace(0, pts[:, 0].max() * 1.05, 50)
    ax2.plot(xs, slope * xs + intercept, color=MUTED, lw=1.2, ls=(0, (4, 3)),
             label=f"fit, zero-variance limit {intercept:.3f}")
    ax2.axhline(E0, color=INK, lw=1.1, ls=(0, (4, 3)))
    ax2.annotate(f"exact {E0:.3f}", xy=(0.99, E0), xycoords=("axes fraction", "data"),
                 ha="right", va="bottom", fontsize=9, color=INK)
    style(ax2, "Var($E_{loc}$)", "energy", "Zero-variance extrapolation")
    legend(ax2, loc="upper left")
    print(f"         extrapolating the energy to zero variance gives {intercept:.4f}, "
          f"exact is {E0:.4f}")


@plot("plot-eloc", "distribution of the local energy")
def plot_eloc(ex):
    """The histogram the energy estimate is an average of. Random parameters give a
    wide, lumpy distribution; optimised parameters give a narrow one centred lower.
    The width divided by sqrt(N_samples) is your error bar."""
    fig, (ax1, ax2) = figure("local energy distribution", 2)
    E0 = ex.ising1d_energy(N_SITES, 1.0)
    runs = standard_runs(ex)
    # the two panels get their own x-range: optimised parameters give a distribution
    # so much narrower that drawing both on one scale hides it completely
    panels = [(ax1, "random starting parameters"), (ax2, "optimised parameters")]

    for label, color, dash in [("mean-field (direct)", BLUE, SOLID), ("Jastrow J1+J2", VIOLET, DASHDOT)]:
        h = runs[label]
        a = h["ansatz"]
        np.random.seed(3)
        for (ax, _), params in zip(panels, [a.params0(N_SITES), h["params"]]):
            eloc, _ = local_energies(ex, a, params, N_SITES, 8000)
            ax.hist(eloc, bins=70, density=True, histtype="step", lw=1.8, color=color,
                    linestyle=dash, label=f"{label}: mean {eloc.mean():.2f}, sd {eloc.std():.2f}")

    for ax, title in panels:
        ax.axvline(E0, color=INK, lw=1.1, ls=(0, (4, 3)))
        ax.annotate("exact", xy=(E0, 0.55), xycoords=("data", "axes fraction"),
                    ha="right", va="center", fontsize=9, color=INK, rotation=90)
        style(ax, "$E_{loc}$", "density", title)
        legend(ax, loc="upper left")
    ax2.set_xlim(E0 - 4.5, E0 + 3)


@plot("plot-autocorr", "autocorrelation of the Markov chain")
def plot_autocorr(ex):
    """Exercise 7.1j. Successive MCMC samples are correlated, so N samples are worth
    fewer than N independent ones. Left: the correlation rho(t) from corr_fn_fft.
    Right: tau summed up to a cutoff - read tau off the plateau. Direct sampling
    has no correlation at all, so its rho drops to zero immediately."""
    runs = standard_runs(ex)
    fig, (ax1, ax2) = figure("autocorrelation", 2)
    n = 20000

    series = [("mean-field, MCMC", runs["mean-field (MCMC)"], ORANGE, DASH, "mcmc"),
              ("Jastrow J1+J2, MCMC", runs["Jastrow J1+J2"], VIOLET, DASHDOT, "mcmc"),
              ("mean-field, direct", runs["mean-field (direct)"], BLUE, SOLID, "direct")]

    for label, h, color, dash, sampler in series:
        eloc, _ = local_energies(ex, h["ansatz"], h["params"], N_SITES, n, sampler=sampler)
        rho = np.real(ex.corr_fn_fft(eloc))
        lags = np.arange(60)
        ax1.plot(lags, rho[lags], color=color, ls=dash, lw=1.8, label=label)

        # tau as a function of where the sum is cut off
        cutoffs = np.arange(1, 120)
        taus = 0.5 + np.cumsum(rho[1:120])
        ax2.plot(cutoffs, taus, color=color, ls=dash, lw=1.8, label=label)
        jcut = int(np.argmin(rho > 0))
        tau = 0.5 + rho[1:jcut].sum()
        ax2.plot([jcut], [tau], marker="o", markersize=7, color=color)
        print(f"         {label:<22} tau = {tau:5.2f}  (first negative rho at lag {jcut}, "
              f"so {n} samples are worth about {n / max(2 * tau, 1):.0f} independent ones)")

    ax1.axhline(0, color=GRID, lw=1)
    style(ax1, "lag $t$", r"$\rho(t)$", "Autocorrelation of $E_{loc}$")
    legend(ax1, loc="upper right")
    style(ax2, "cutoff lag", r"$\tau$", r"$\tau = 1/2 + \sum_{t\geq1}\rho(t)$ vs cutoff (dot: first negative $\rho$)")
    legend(ax2, loc="upper left")


@plot("plot-thermalization", "thermalisation of the Markov chain")
def plot_thermalization(ex):
    """Why sample_mcmc throws away the first N_discard samples. Chains started from a
    random state, and one started from the all-up state, need some steps before they
    forget where they came from. Right: the running average is only trustworthy once
    every chain sits on the same value."""
    runs = standard_runs(ex)
    h = runs["Jastrow J1+J2"]
    a, params = h["ansatz"], h["params"]
    operator = lambda x: ex.ising_hamiltonian(x, 1.0, 1.0)
    nsteps = 1500

    fig, (ax1, ax2) = figure("thermalisation", 2)
    np.random.seed(7)
    starts = [(None, f"random start {i + 1}", RAMP[i + 1], SOLID) for i in range(3)]
    starts.append((np.ones(N_SITES, dtype=int), "all spins up", ORANGE, DASH))

    for x0, label, color, dash in starts:
        # N_discard=0: we want to see the part that is normally thrown away
        chain = ex.sample_mcmc(a.logpsi, params, N_SITES, nsteps, 0, x0=x0)
        eloc = np.real(ex.compute_eloc(operator, a.logpsi, params, chain))
        ax1.plot(running_mean(eloc, 25), color=color, ls=dash, lw=1.5, label=label)
        ax2.plot(np.cumsum(eloc) / np.arange(1, nsteps + 1), color=color, ls=dash, lw=1.5, label=label)

    for ax in (ax1, ax2):
        ax.axvline(N_DISCARD, color=INK, lw=1.1, ls=(0, (4, 3)))
        ax.annotate(f"  N_discard = {N_DISCARD}", xy=(N_DISCARD, 0.98), xycoords=("data", "axes fraction"),
                    ha="left", va="top", fontsize=9, color=INK)
    E, _ = converged(h)
    ax2.axhline(E, color=MUTED, lw=1.1, ls=(0, (1.5, 1.5)))
    ax1.set_xlim(0, 600)
    style(ax1, "MCMC step", "$E_{loc}$ (25-step mean)", "Local energy along the chain")
    style(ax2, "MCMC step", "running average of $E_{loc}$", "Running estimate of the energy")
    # one legend for the figure: both panels show the same four chains
    legend(ax2, loc="upper right")
    print(f"         acceptance rate during the optimisation: {h['acc'].mean():.2f}")


@plot("plot-error-scaling", "statistical error vs number of samples")
def plot_error_scaling(ex):
    """The error falls as 1/sqrt(N_samples) for both samplers, but MCMC sits higher:
    correlated samples carry less information. The naive error sd/sqrt(N) that ignores
    correlation underestimates the true scatter by about sqrt(2*tau). The right panel
    bounces around because each point is itself a standard deviation estimated from
    only 20 repetitions."""
    runs = standard_runs(ex)
    h = runs["Jastrow J1+J2"]
    a, params = h["ansatz"], h["params"]
    mf = runs["mean-field (direct)"]
    sizes = [32, 64, 128, 256, 512, 1024, 2048, 4096]
    repeats = 20  # the scatter of an estimated standard deviation is itself noisy

    np.random.seed(11)
    true_err, naive_err, direct_err = [], [], []
    print(f"         sampling {repeats} repetitions for each of {len(sizes)} sample sizes ... ",
          end="", flush=True)
    t0 = time.time()
    for n in sizes:
        Es, naive = [], []
        for _ in range(repeats):
            eloc, _ = local_energies(ex, a, params, N_SITES, n, N_discard=N_DISCARD)
            Es.append(eloc.mean())
            naive.append(eloc.std() / np.sqrt(n))
        true_err.append(np.std(Es))
        naive_err.append(np.mean(naive))
        Ed = [local_energies(ex, mf["ansatz"], mf["params"], N_SITES, n, sampler="direct")[0].mean()
              for _ in range(repeats)]
        direct_err.append(np.std(Ed))
    print(f"{time.time() - t0:.1f}s")

    sizes = np.array(sizes, dtype=float)
    true_err, naive_err, direct_err = map(np.array, (true_err, naive_err, direct_err))
    fig, (ax1, ax2) = figure("error scaling", 2)

    ax1.loglog(sizes, direct_err, marker="o", markersize=6, color=BLUE, ls=SOLID, lw=1.8,
               label="direct sampling, true scatter")
    ax1.loglog(sizes, true_err, marker="s", markersize=6, color=VIOLET, ls=DASHDOT, lw=1.8,
               label="MCMC, true scatter")
    ax1.loglog(sizes, naive_err, marker="^", markersize=6, color=ORANGE, ls=DASH, lw=1.8,
               label="MCMC, naive sd/$\\sqrt{N}$")
    # a 1/sqrt(N) guide fitted through the MCMC points, rather than pinned to one of them
    ref = np.exp(np.mean(np.log(true_err) + 0.5 * np.log(sizes))) / np.sqrt(sizes)
    ax1.loglog(sizes, ref, color=MUTED, lw=1.1, ls=(0, (1.5, 1.5)), label="$1/\\sqrt{N}$")
    style(ax1, "number of samples", "error on the energy", "Error of the energy estimate")
    legend(ax1, loc="lower left")

    ax2.semilogx(sizes, true_err / naive_err, marker="s", markersize=6, color=VIOLET, lw=1.8)
    tau = 0.5 + np.real(ex.corr_fn_fft(local_energies(ex, a, params, N_SITES, 20000)[0]))[1:40].sum()
    ax2.axhline(np.sqrt(2 * tau), color=INK, lw=1.1, ls=(0, (4, 3)))
    ax2.annotate(f"$\\sqrt{{2\\tau}}$ = {np.sqrt(2 * tau):.2f}", xy=(0.99, np.sqrt(2 * tau)),
                 xycoords=("axes fraction", "data"), ha="right", va="bottom", fontsize=9, color=INK)
    style(ax2, "number of samples", "true error / naive error",
          "How much the naive error bar lies (MCMC)")


@plot("plot-parameters", "parameters along the optimisation")
def plot_parameters(ex):
    """What the optimiser actually does to the parameters. The Jastrow couplings
    settle on negative J1 (neighbouring spins prefer to be opposite, J>0 is
    antiferromagnetic) and a smaller positive J2. For mean-field, the per-site
    magnetisation develops the alternating up/down pattern."""
    runs = standard_runs(ex)
    fig, (ax1, ax2) = figure("parameters", 2)

    h2 = runs["Jastrow J1+J2"]
    for i, (name, color, dash) in enumerate([("$J_1$", VIOLET, SOLID), ("$J_2$", ORANGE, DASH)]):
        ax1.plot(h2["params_hist"][:, i], color=color, ls=dash, lw=1.8, label=f"{name} (J1+J2 ansatz)")
    h1 = runs["Jastrow J1"]
    ax1.plot(h1["params_hist"][:, 0], color=AQUA, ls=DOT, lw=1.8, label="$J_1$ (J1-only ansatz)")
    ax1.axhline(0, color=GRID, lw=1)
    ax1.set_ylim(-0.36, 0.2)
    style(ax1, "optimisation step", "coupling", "Jastrow couplings")
    legend(ax1, loc="center right")
    print(f"         final couplings: J1+J2 ansatz {np.round(h2['params'], 4)}, "
          f"J1 only {np.round(h1['params'], 4)}")

    # mean-field: sample at a few snapshots and measure <s_i> site by site
    mf = runs["mean-field (direct)"]
    # chosen around the symmetry breaking: flat, flat, dropping, done, done
    snapshots = [0, 300, 520, 600, len(mf["params_hist"]) - 1]
    sites = np.arange(N_SITES)
    for color, step in zip(RAMP, snapshots):
        x = ex.sample_direct_mf(mf["ansatz"].logpsi, mf["params_hist"][step], N_SITES, 4000)
        ax2.plot(sites, x.mean(0), marker="o", markersize=5, color=color, lw=1.6,
                 label=f"step {step}")
    ax2.axhline(0, color=GRID, lw=1)
    ax2.set_ylim(-1.55, 1.15)  # room for the legend under the data
    style(ax2, "site $i$", r"$\langle s_i \rangle$", "Mean-field magnetisation per site")
    legend(ax2, loc="lower center", ncol=3)


@plot("plot-field-scan", "energy across the transverse field")
def plot_field_scan(ex):
    """Both ansaetze are re-optimised at each field Gamma, sweeping downwards and
    starting each run from the previous parameters (annealing). Both are good deep in
    either phase and worst near Gamma = 1, the critical point, where correlations
    reach furthest. The third curve on the right shows why the annealing matters:
    started from scratch, mean-field gets stuck in a domain-wall state at low field -
    a converged run that is still 20% off."""
    fields = [2.2, 1.7, 1.3, 1.0, 0.8, 0.5, 0.2]  # downwards, so each run starts warm
    rows = []
    prev_mf = prev_j2 = None
    for G in fields:
        mf = optimize(ex, mean_field(ex, "direct"), G=G, nsteps=500, params_init=prev_mf, tag="annealed")
        j2 = optimize(ex, jastrow2(ex), G=G, nsteps=250, params_init=prev_j2, tag="annealed")
        cold = optimize(ex, mean_field(ex, "direct"), G=G, nsteps=STEPS_MF)
        prev_mf, prev_j2 = mf["params"], j2["params"]
        rows.append((G, converged(mf)[0], converged(j2)[0], converged(cold)[0],
                     ex.ising1d_energy(N_SITES, G)))
    G, E_mf, E_j2, E_cold, E_ex = map(np.array, zip(*sorted(rows)))

    fig, (ax1, ax2) = figure("field scan", 2)
    ax1.plot(G, E_ex / N_SITES, color=INK, lw=1.4, ls=(0, (4, 3)), label="exact")
    ax1.plot(G, E_mf / N_SITES, marker="o", markersize=6, color=BLUE, ls=SOLID, lw=1.8, label="mean-field")
    ax1.plot(G, E_j2 / N_SITES, marker="s", markersize=6, color=VIOLET, ls=DASHDOT, lw=1.8, label="Jastrow J1+J2")
    style(ax1, r"transverse field $\Gamma$", "energy per site", "Ground-state energy")
    legend(ax1, loc="lower left")

    ax2.semilogy(G, np.abs((E_mf - E_ex) / E_ex), marker="o", markersize=6, color=BLUE, ls=SOLID, lw=1.8,
                 label="mean-field, annealed")
    ax2.semilogy(G, np.abs((E_cold - E_ex) / E_ex), marker="^", markersize=6, color=ORANGE, ls=DASH,
                 lw=1.8, label="mean-field, from scratch")
    ax2.semilogy(G, np.abs((E_j2 - E_ex) / E_ex), marker="s", markersize=6, color=VIOLET, ls=DASHDOT,
                 lw=1.8, label="Jastrow J1+J2")
    ax2.axvline(1.0, color=INK, lw=1.1, ls=(0, (4, 3)))
    ax2.annotate("critical point", xy=(1.0, 0.98), xycoords=("data", "axes fraction"),
                 ha="left", va="top", fontsize=9, color=INK)
    style(ax2, r"transverse field $\Gamma$", "relative error", "Relative error vs the field")
    legend(ax2, loc="lower left")


@plot("plot-neighbours", "how many Jastrow neighbours are needed")
def plot_neighbours(ex):
    """Exercise 7.2b asks how far the Jastrow has to reach. Each run adds one more
    neighbour shell to log psi = sum_d J_d sum_i s_i s_(i+d). The error keeps falling,
    with most of the gain in the first two shells, and the couplings decay with
    distance. These runs use a smaller learning rate than the rest: with more
    parameters, eta=0.005 leaves the fit rattling around in the sampling noise
    instead of settling, which made the k=5 run look worse than k=4."""
    ks = [1, 2, 3, 4, 5, 6]
    E0 = ex.ising1d_energy(N_SITES, 1.0)
    results = [optimize(ex, jastrow_k(k), nsteps=600, eta=0.002) for k in ks]
    errs = np.array([abs((converged(h, 200)[0] - E0) / E0) for h in results])

    fig, (ax1, ax2) = figure("jastrow range", 2)
    ax1.semilogy(ks, errs, marker="o", markersize=7, color=BLUE, lw=1.8)
    for k, e in zip(ks, errs):
        ax1.annotate(f"{e:.1e}", xy=(k, e), xytext=(0, 9), textcoords="offset points",
                     ha="center", fontsize=9, color=MUTED)
    ax1.set_xticks(ks)
    style(ax1, "neighbour shells $k$", "relative error", "Error vs range of the Jastrow")

    for color, k, h in zip(RAMP + RAMP[-1:], ks, results):
        d = np.arange(1, k + 1)
        ax2.plot(d, h["params"], marker="o", markersize=5, color=color, lw=1.6, label=f"k = {k}")
    ax2.axhline(0, color=GRID, lw=1)
    ax2.set_xticks(ks)
    style(ax2, "distance $d$", "$J_d$", "Optimised couplings")
    legend(ax2, loc="lower right", ncol=2)
    for k, h, e in zip(ks, results, errs):
        print(f"         k = {k}: rel. error {e:.4f}, couplings {np.round(h['params'], 4)}")


@plot("plot-correlations", "spin correlations vs exact diagonalisation")
def plot_correlations(ex):
    """Energy is not everything: this is what the wave-functions get wrong. At N=10 the
    exact ground state is available by diagonalisation, so the sampled correlations
    <s_i s_i+d> can be compared directly. All three get the alternating sign right and
    each fails differently: mean-field is a product state, so its correlations are just
    the product of two magnetisations and never decay (flat, too strong at large d);
    J1 alone decays far too fast; J1+J2 is closest at every distance. Note that J1 has
    a better energy than these correlations suggest - a good energy does not imply
    everything else is good."""
    N, G, J = 10, 1.0, 1.0
    H = dense_ising(N, G, J)
    vals, vecs = np.linalg.eigh(H)
    psi0 = vecs[:, 0]
    p = psi0**2
    S = all_states(N)
    ds = np.arange(1, N // 2 + 1)
    C_exact = np.array([float(p @ (neighbour_corr(S, d) / N)) for d in ds])
    print(f"         exact diagonalisation at N={N}: E0 = {vals[0]:.5f}, "
          f"analytic formula {ex.ising1d_energy(N, G):.5f}")

    runs = [("mean-field", optimize(ex, mean_field(ex, "direct"), N=N, nsteps=STEPS_MF), BLUE, SOLID, "o"),
            ("Jastrow J1", optimize(ex, jastrow1(ex), N=N, nsteps=300), AQUA, DOT, "s"),
            ("Jastrow J1+J2", optimize(ex, jastrow2(ex), N=N, nsteps=300), VIOLET, DASHDOT, "D")]

    fig, (ax1, ax2) = figure("spin correlations", 2)
    ax1.plot(ds, C_exact, color=INK, lw=1.4, ls=(0, (4, 3)), marker="*", markersize=9,
             label="exact (diagonalisation)")
    for label, h, color, dash, marker in runs:
        x = draw(ex, h["ansatz"], h["params"], N, 20000, N_DISCARD, h["sampler"])
        per_sample = np.stack([neighbour_corr(x, d) / N for d in ds], axis=1)
        C = per_sample.mean(0)
        err = per_sample.std(0) / np.sqrt(len(x))
        ax1.errorbar(ds, C, yerr=err, color=color, ls=dash, lw=1.8, marker=marker,
                     markersize=6, capsize=3, label=label)
        ax2.semilogy(ds, np.abs(C - C_exact), color=color, ls=dash, lw=1.8, marker=marker,
                     markersize=6, label=label)

    ax1.axhline(0, color=GRID, lw=1)
    ax1.set_ylim(-0.95, 1.45)  # room for the legend above the data
    ax1.set_xticks(ds)
    ax2.set_xticks(ds)
    style(ax1, "distance $d$", r"$\langle s_i s_{i+d} \rangle$", f"Spin correlations, N = {N}")
    legend(ax1, loc="upper center", ncol=2)
    style(ax2, "distance $d$", "|error|", "Distance from the exact correlations")
    legend(ax2, loc="upper left")
