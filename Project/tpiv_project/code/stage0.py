from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

rng = np.random.default_rng(1234)

# plot style and colors: VMC data, analytic result, reference lines
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#e1e0d9", "axes.axisbelow": True, "lines.linewidth": 2})
C_VMC, C_EXACT, C_REF = "#2a78d6", "#eb6834", "#898781"


def metropolis_step(logpsi, x, logpsi_x, step_size, rng=rng):
    # propose a Gaussian displacement of one particle and accept or reject it
    if logpsi_x is None:
        # for the sampling we work with a single sample
        # but our ansatz only supports batches
        # so we add a dummy batch dimension here
        logpsi_x = logpsi(np.expand_dims(x, 0))[0]

    index = rng.integers(x.shape[0])
    # copy: with x_prime = x both names point to the same array,
    # so a rejected move would still have modified x
    x_prime = x.copy()
    x_prime[index] = x[index] + step_size*rng.normal(size=x.shape[1])
    logpsi_x_prime = logpsi(np.expand_dims(x_prime, 0))[0]
    # min(1, |psi(x')|^2 / |psi(x)|^2) written with the logs
    R = np.exp(min(0., 2*(logpsi_x_prime - logpsi_x)))
    accept = R > rng.random()

    if accept:
        return x_prime, logpsi_x_prime, True
    else:
        return x, logpsi_x, False


def sample_mcmc(logpsi, n_particles, n_samples, n_discard, step_size, dim=2, x0=None, rng=rng,
                return_acceptance=False):
    # returns an array with shape (n_samples, n_particles, dim)
    if x0 is None:
        # broad initial position, the burn-in brings the walker in
        x0 = rng.normal(loc=0, scale=2, size=(n_particles, dim))

    x = np.array(x0, dtype=float)
    logpsi_x = None
    n_accepted = 0

    # one sweep = n_particles single-particle moves, one sample is stored per sweep
    # n_discard counts the thermalization sweeps
    samples = np.empty((n_samples, n_particles, dim))
    for sweep in range(n_discard + n_samples):
        for _ in range(n_particles):
            x, logpsi_x, accepted = metropolis_step(logpsi, x, logpsi_x, step_size, rng)
            n_accepted += accepted
        if sweep >= n_discard:
            samples[sweep - n_discard] = x

    if return_acceptance:
        return samples, n_accepted / ((n_discard + n_samples) * n_particles)
    return samples


def local_energy(logpsi, grad_logpsi, laplacian_logpsi, potential, x):
    # x has shape (n_samples, n_particles, dim): sum over particles and coordinates,
    # one value per configuration
    grad = grad_logpsi(x)
    lap = laplacian_logpsi(x)
    e_kin = -.5*np.sum(lap + grad**2, axis=(1, 2))
    V = np.sum(potential(x), axis=(1, 2))
    return e_kin + V


def estimate_energy(eloc, n_blocks=20):
    # mean and standard error of correlated samples
    mean = np.mean(eloc)
    # blocking: successive samples are correlated, means over long blocks are not
    block_size = len(eloc) // n_blocks
    block_means = np.mean(eloc[:n_blocks*block_size].reshape(n_blocks, block_size), axis=1)
    err = np.std(block_means, ddof=1) / np.sqrt(n_blocks)
    return mean, err


def autocorrelation_time(series):
    # tau = 1 + 2 sum_t rho(t), summed until rho(t) first becomes negative
    # tau = 1 for independent samples, the effective sample size is len(series) / tau
    x = series - np.mean(series)
    var = np.mean(x**2)
    if var < 1e-20:
        return np.nan   # constant series, e.g. E_L at alpha = 1/2
    tau = 1.
    for t in range(1, len(x) // 10):
        rho = np.mean(x[:-t]*x[t:]) / var
        if rho < 0:
            break
        tau += 2*rho
    return tau


# Gaussian ansatz: log psi = -alpha r^2

def logpsi_gaussian(params, x):
    # params[0] is alpha; x has shape (n_samples, n_particles, dim)
    return -params[0]*np.sum(x**2, axis=(1, 2))


def grad_logpsi_gaussian(params, x):
    return -2*params[0]*x


def laplacian_logpsi_gaussian(params, x):
    return -2*params[0]*np.ones_like(x)


def harmonic_potential(x):
    return 0.5*x**2


def exact_energy(alpha, dim=2):
    return dim/2*(alpha + 1/(4*alpha))


def vmc_gaussian(alpha, dim=2, n_samples=20000, n_discard=1000, step_size=1.0):
    # one VMC run at fixed alpha: samples, local energies and acceptance rate
    params = np.array([alpha])
    logpsi = lambda x: logpsi_gaussian(params, x)
    grad_logpsi = lambda x: grad_logpsi_gaussian(params, x)
    laplacian_logpsi = lambda x: laplacian_logpsi_gaussian(params, x)

    x, acceptance = sample_mcmc(logpsi, 1, n_samples, n_discard, step_size, dim=dim, return_acceptance=True)
    eloc = local_energy(logpsi, grad_logpsi, laplacian_logpsi, harmonic_potential, x)
    return x, eloc, acceptance


def gaussian_parameter_gradient(params, x, eloc):
    # O_alpha = dlogpsi/dalpha = -r^2,  dE/dalpha = 2 (<E_L O> - <E_L><O>)
    O = -np.sum(x**2, axis=(1, 2))
    return np.array([2*(np.mean(eloc*O) - np.mean(eloc)*np.mean(O))])


def optimize_gaussian(dim=2, n_steps=50, alpha0=0.2, lr=0.1, n_samples=2000, n_discard=100, step_size=1.0):
    params = np.array([alpha0], dtype=float)
    history = {"alpha": [], "energy": [], "error": []}
    x0 = None

    for step in range(n_steps):
        logpsi = lambda x: logpsi_gaussian(params, x)
        grad_logpsi = lambda x: grad_logpsi_gaussian(params, x)
        laplacian_logpsi = lambda x: laplacian_logpsi_gaussian(params, x)

        x = sample_mcmc(logpsi, 1, n_samples, n_discard, step_size, dim=dim, x0=x0)
        eloc = local_energy(logpsi, grad_logpsi, laplacian_logpsi, harmonic_potential, x)
        energy, err = estimate_energy(eloc)
        history["alpha"].append(params[0])
        history["energy"].append(energy)
        history["error"].append(err)

        grad = gaussian_parameter_gradient(params, x, eloc)
        # alpha must stay positive, otherwise psi is not normalizable
        params = np.maximum(params - lr*grad, 1e-3)
        # the next chain starts from the last configuration of this one
        x0 = x[-1]

    return params, {key: np.array(values) for key, values in history.items()}


if __name__ == "__main__":
    out = Path(__file__).resolve().parents[1] / "results" / "stage0"
    out.mkdir(parents=True, exist_ok=True)

    # 1. scan alpha before any optimization
    alphas = np.array([0.2, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.8, 1.0, 1.2])
    runs = [vmc_gaussian(alpha) for alpha in alphas]
    energies, errors = np.array([estimate_energy(eloc) for _, eloc, _ in runs]).T

    print(" alpha   E (VMC)               E (exact)   Var E_L    acceptance   tau     ESS")
    for alpha, (x, eloc, acceptance), energy, err in zip(alphas, runs, energies, errors):
        tau = autocorrelation_time(eloc)
        print(f"{alpha:6.2f}   {energy:.5f} +- {err:.5f}   {exact_energy(alpha):.5f}     {np.var(eloc):.2e}"
              f"   {acceptance:.2f}         {tau:5.1f}   {len(eloc)/tau:6.0f}")

    # 2. proposal width, at alpha = 0.3 (at alpha = 1/2 E_L is constant and has no autocorrelation)
    step_sizes = np.array([0.1, 0.2, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0])
    width_runs = [vmc_gaussian(0.3, step_size=step) for step in step_sizes]
    acceptances = np.array([acceptance for _, _, acceptance in width_runs])
    taus = np.array([autocorrelation_time(eloc) for _, eloc, _ in width_runs])

    print("\n step size   acceptance   tau     ESS        (alpha = 0.3)")
    for step, acceptance, tau in zip(step_sizes, acceptances, taus):
        print(f"{step:8.1f}    {acceptance:.2f}         {tau:6.1f}  {20000/tau:6.0f}")

    # 3. gradient optimization, then a final run with alpha fixed
    params, history = optimize_gaussian(alpha0=0.2)
    x, eloc, acceptance = vmc_gaussian(params[0])
    energy, err = estimate_energy(eloc)
    print(f"\noptimized alpha = {params[0]:.10f}   (exact 0.5)")
    print(f"final E = {energy:.10f} +- {err:.1e}, Var E_L = {np.var(eloc):.1e}   (exact 1)")

    # plots: E(alpha), acceptance and autocorrelation time vs step size
    fig, (ax_E, ax_acc, ax_tau) = plt.subplots(1, 3, figsize=(14, 4))
    alpha_grid = np.linspace(0.15, 1.25, 200)
    ax_E.plot(alpha_grid, exact_energy(alpha_grid), color=C_EXACT, label=r"analytic $\alpha+1/4\alpha$")
    ax_E.axhline(1, color=C_REF, linewidth=1, label="exact $E_0=1$")
    ax_E.errorbar(alphas, energies, yerr=errors, fmt="o", color=C_VMC, markersize=7, markeredgecolor="white",
                  capsize=3, label="VMC")
    ax_E.set(xlabel=r"$\alpha$", ylabel=r"$E(\alpha)$", title="Variational energy")
    ax_E.legend(frameon=False)

    ax_acc.plot(step_sizes, acceptances, "o-", color=C_VMC, markersize=7, markeredgecolor="white")
    ax_acc.set(xscale="log", xlabel="step size", ylabel="acceptance rate", title=r"Acceptance, $\alpha=0.3$", ylim=(0, 1))

    ax_tau.plot(step_sizes, taus, "o-", color=C_VMC, markersize=7, markeredgecolor="white")
    ax_tau.set(xscale="log", yscale="log", xlabel="step size", ylabel=r"$\tau_{\rm int}$ of $E_L$ (sweeps)",
               title=r"Autocorrelation time, $\alpha=0.3$")
    fig.tight_layout()
    fig.savefig(out / "energy_and_sampling.png", dpi=150)

    # plots: local-energy histograms away from and at the exact alpha
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8), sharex=True)
    # bins of width 0.05 centred on E = 1, where the alpha = 1/2 spike is
    bins = np.arange(-1.025, 4.5, 0.05)
    for ax, alpha in zip(axes, [0.3, 0.5, 0.8]):
        eloc = runs[list(alphas).index(alpha)][1]
        ax.hist(eloc, bins=bins, density=True, histtype="stepfilled", color=C_VMC, alpha=0.5)
        ax.axvline(1, color=C_REF, linewidth=1, linestyle=":")
        ax.set(xlabel="$E_L$", title=rf"$\alpha={alpha}$, std$(E_L)={np.std(eloc):.2g}$")
    axes[0].set_ylabel("probability density")
    fig.tight_layout()
    fig.savefig(out / "local_energy_histograms.png", dpi=150)
    print(f"\nplots saved in {out}")
