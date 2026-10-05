"""Experiment grids. Each study is a function returning a list of run configs.

Architectures used throughout (N = 20 unless stated):
  ffnn      : MLP of the exercise, 1 hidden layer of width N/4, ReLU on the output unit (Eq. 2)
  ffnn_lin  : same MLP without the output ReLU (ablation of the 'dead output' problem)
  rbm       : RBM with alpha = 1 (suggested in the meeting notes)
"""

from itertools import product

from vmc_lib import default_config as C

SEEDS = [0, 1, 2, 3]
ARCHS = {
    "ffnn": dict(model="ffnn", width=5, depth=1, out_act="relu"),
    "ffnn_lin": dict(model="ffnn", width=5, depth=1, out_act="linear"),
    "rbm": dict(model="rbm", alpha=1),
}
LRS = [0.003, 0.01, 0.03, 0.1]
SHIFTS = [1e-3, 1e-2, 1e-1, 1.0]


def S1_optimizer():
    """SGD vs SGD+SR (lr x diag_shift) vs Adam (+/- SR) at the critical point, N = 20."""
    s = "S1_optimizer"
    cfgs = []
    for arch, seed in product(ARCHS.values(), SEEDS):
        for lr in LRS:
            cfgs.append(C(study=s, **arch, optimizer="sgd", lr=lr, seed=seed))
            for ds in SHIFTS:
                cfgs.append(C(study=s, **arch, optimizer="sgd", lr=lr, sr=True, diag_shift=ds, seed=seed))
        for lr, sr in product([0.001, 0.01], [False, True]):
            cfgs.append(C(study=s, **arch, optimizer="adam", lr=lr, sr=sr, diag_shift=0.01, seed=seed))
    return cfgs


def S3_fullsum():
    """Noise-free optimisation (exact sums over the 2^12 states), N = 12: what sets the timescale."""
    s = "S3_fullsum"
    rbm = dict(model="rbm", alpha=2)
    ffnns = [dict(model="ffnn", width=6, depth=1, out_act="relu"), dict(model="ffnn", width=6, depth=1, out_act="linear")]
    base = dict(study=s, N=12, state="full", n_iter=600)
    cfgs = []
    # (a) dependence on the field h, i.e. on the gap (SR close to exact imaginary-time evolution)
    for h, seed in product([0.3, 0.5, 0.7, 0.85, 1.0, 1.2, 1.5, 2.0, 3.0], [0, 1, 2]):
        cfgs.append(C(**base, **rbm, h=h, sr=True, diag_shift=1e-4, lr=0.01, seed=seed))
    # (b) dependence on the learning rate
    for h, lr, seed in product([0.5, 1.0, 2.0], LRS, [0, 1, 2]):
        cfgs.append(C(**base, **rbm, h=h, sr=True, diag_shift=1e-4, lr=lr, seed=seed))
    # (c) dependence on the diagonal shift, for the three Ansatz families
    for arch, ds, seed in product([rbm] + ffnns, [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0], [0, 1, 2]):
        cfgs.append(C(**base, **arch, h=1.0, sr=True, diag_shift=ds, lr=0.01, seed=seed))
    # (d) dependence on the system size (gap ~ 1/N at the critical point, finite in the ordered phase)
    for N, h, seed in product([6, 8, 10, 12, 14], [0.5, 1.0], [0, 1]):
        cfgs.append(C(**{**base, "N": N}, **rbm, h=h, sr=True, diag_shift=1e-4, lr=0.01, seed=seed))
    # (e) plain SGD without noise: are the plateaus a property of the landscape?
    for arch, lr, seed in product([rbm] + ffnns, LRS, SEEDS):
        cfgs.append(C(**{**base, "n_iter": 2000}, **arch, h=1.0, lr=lr, seed=seed))
    return cfgs


def S7_plateaus():
    """Many seeds of the baseline settings, to get the distribution of plateau-escape times."""
    s = "S7_plateaus"
    cfgs = []
    for arch, seed in product(ARCHS.values(), range(12)):
        cfgs.append(C(study=s, **arch, optimizer="sgd", lr=0.01, n_iter=3000, seed=seed))
        cfgs.append(C(study=s, **arch, optimizer="sgd", lr=0.01, sr=True, diag_shift=0.01, seed=seed))
    return cfgs


# ---------------------------------------------------------------- phase 2 (settings chosen from S1)

# Reference settings: plain SGD of the original notebook, and the SR setting selected from S1
SGD_REF = dict(optimizer="sgd", lr=0.01, sr=False)
# Chosen from S1: fastest setting that converged every RBM seed with ~3x margin in lr before
# instabilities (lr = 0.1 with shift 1e-3 collapses). 400 iterations at lr = 0.03 correspond to
# 1200 at lr = 0.01, far more than SR needs (S1: 1 % within ~20 iterations, 0.2 % within ~40).
SR_REF = dict(optimizer="sgd", lr=0.03, sr=True, diag_shift=0.01, n_iter=400)


def S2_samples():
    """Number of Monte Carlo samples per iteration (chains scaled so that every chain gives >= 4 samples)."""
    s = "S2_samples"
    cfgs = []
    for arch, opt, Ns, seed in product(ARCHS.values(), [SGD_REF, SR_REF], [128, 256, 512, 1024, 2048, 4096], [0, 1, 2]):
        cfgs.append(C(study=s, **arch, **opt, n_samples=Ns, n_chains=min(128, Ns // 4), seed=seed))
    return cfgs


def S4_size():
    """Systematic scan of the network size: FFNN width x depth (both output activations), RBM alpha."""
    s = "S4_size"
    cfgs = []
    for opt, seed in product([SGD_REF, SR_REF], SEEDS):
        for depth, width, out_act in product([1, 2, 3], [2, 4, 8, 16, 32, 64], ["relu", "linear"]):
            cfgs.append(C(study=s, model="ffnn", width=width, depth=depth, out_act=out_act, **opt, seed=seed))
        for alpha in [0.25, 0.5, 1, 2, 4, 8]:
            cfgs.append(C(study=s, model="rbm", alpha=alpha, **opt, seed=seed))
    return cfgs


H_SCAN = [0.25, 0.5, 0.75, 0.9, 1.0, 1.1, 1.25, 1.5, 2.0, 3.0]


def S5_phase():
    """Field scan across the transition: reliability of convergence, sampling diagnostics, physics."""
    s = "S5_phase"
    cfgs = []
    for h, seed in product(H_SCAN, SEEDS):
        # original notebook setting and the recommended one, N = 20
        cfgs.append(C(study=s, h=h, **ARCHS["ffnn"], **SGD_REF, seed=seed))
        cfgs.append(C(study=s, h=h, **ARCHS["ffnn"], **SR_REF, seed=seed))
        cfgs.append(C(study=s, h=h, **ARCHS["rbm"], **SGD_REF, seed=seed))
        cfgs.append(C(study=s, h=h, **ARCHS["rbm"], **SR_REF, seed=seed))
        # finite-size comparison for the transition
        if seed < 3:
            cfgs.append(C(study=s, N=12, h=h, **ARCHS["rbm"], **SR_REF, seed=seed))
    return cfgs


SAMPLERS = [(16, 128), (16, 4), (128, 4), (512, 2)]  # (n_chains, n_discard per chain per iteration)


def S6_sampler():
    """Few long chains vs many short chains, at the critical point and in the ordered phase."""
    s = "S6_sampler"
    cfgs = []
    for h, (nc, nd), seed in product([0.5, 1.0], SAMPLERS, [0, 1, 2]):
        cfgs.append(C(study=s, h=h, **ARCHS["rbm"], **SR_REF, n_chains=nc, n_discard=nd, seed=seed))
    return cfgs


def S8_init():
    """Initial scale of the RBM parameters: does it set the length of the SGD plateau?"""
    s = "S8_init"
    cfgs = []
    for sc, seed in product([1e-4, 1e-3, 1e-2, 1e-1, 0.5], [0, 1, 2]):
        base = dict(study=s, N=12, state="full", model="rbm", alpha=2, init_scale=sc, seed=seed)
        cfgs.append(C(**base, lr=0.01, n_iter=2000))
        cfgs.append(C(**base, lr=0.01, sr=True, diag_shift=1e-4, n_iter=600))
    for sc, seed in product([1e-3, 1e-2, 1e-1], [0, 1, 2]):
        cfgs.append(C(study=s, **ARCHS["rbm"], init_scale=sc, **SGD_REF, seed=seed))
        cfgs.append(C(study=s, **ARCHS["rbm"], init_scale=sc, **SR_REF, seed=seed))
    return cfgs


def S9_large_N():
    """Beyond exact diagonalisation (reference: free-fermion E0 and gap): does the convergence time
    grow like 1/gap ~ N at the critical point and stay constant in the gapped phase?"""
    s = "S9_large_N"
    archs = [ARCHS["rbm"], dict(model="rbm_symm", alpha=2)]  # dense RBM vs translation-invariant RBM
    return [C(study=s, N=N, h=h, **arch, **{**SR_REF, "n_iter": 600}, seed=seed)
            for arch, N, h, seed in product(archs, [20, 40, 80], [0.5, 1.0], [0, 1, 2])]


STUDIES = {f.__name__: f for f in [S1_optimizer, S3_fullsum, S7_plateaus, S2_samples, S4_size, S5_phase, S6_sampler, S8_init,
                                   S9_large_N]}
