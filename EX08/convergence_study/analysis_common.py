"""Loading, convergence metrics and plotting style shared by the analysis scripts."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import studies
from vmc_lib import EXACT_DIR, RESULTS, result_path

_EXACT = {}


def exact(N, h, J=1.0):
    """Current exact data (the copy stored inside each run may predate later fixes of the cache)."""
    key = (N, h, J)
    if key not in _EXACT:
        _EXACT[key] = json.loads((EXACT_DIR / f"N{N}_h{h:g}_J{J:g}.json").read_text())
    return _EXACT[key]

PLOTS = Path(__file__).resolve().parent / "plots"
PLOTS.mkdir(exist_ok=True)

# ---------------------------------------------------------------- style
# Categorical slots in fixed order; each entity keeps its colour in every figure.
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
ARCH_COLOR = {"ffnn": CAT[0], "ffnn_lin": CAT[1], "rbm": CAT[2], "rbm_symm": CAT[6]}
ARCH_LABEL = {"ffnn": "FFNN, ReLU output (Eq. 2)", "ffnn_lin": "FFNN, linear output", "rbm": "RBM",
              "rbm_symm": "translation-invariant RBM"}
# Single-hue ordinal ramp (blue), light -> dark, for ordered parameters (diag_shift, lr, size, ...)
BLUE = ["#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
SGD_COLOR = "#898781"  # plain SGD (no preconditioner) is the neutral reference

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK, "axes.titlesize": 10,
    "axes.labelsize": 9, "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "legend.fontsize": 8, "legend.frameon": False, "text.color": INK,
    "lines.linewidth": 1.3, "font.family": "sans-serif", "figure.titlesize": 12,
})


def ramp(n):
    """n ordered colours from the blue ramp (light = small, dark = large)."""
    return [BLUE[int(round(i))] for i in np.linspace(0, len(BLUE) - 1, n)]


def save(fig, name):
    fig.savefig(PLOTS / f"{name}.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


# ---------------------------------------------------------------- data


class Run:
    def __init__(self, cfg, path):
        d = np.load(path, allow_pickle=False)
        self.meta = json.loads(str(d["meta"]))
        self.cfg = cfg
        self.a = {k: d[k] for k in d.files if k != "meta"}
        self.ex = exact(cfg["N"], cfg["h"], cfg["J"])
        self.E0 = self.ex["E0"]
        self.final = self.meta["final"]

    @property
    def arch(self):
        c = self.cfg
        if c["model"] in ("rbm", "rbm_symm"):
            return c["model"]
        return "ffnn" if c["out_act"] == "relu" else "ffnn_lin"

    @property
    def E(self):
        return self.a["E"]

    def rel(self, smooth=1):
        """|E(t) - E0| / |E0|; with smooth > 1 the energy is first averaged over a trailing window."""
        E = self.E
        if smooth > 1 and len(E) >= smooth:
            c = np.cumsum(np.insert(E, 0, 0.0))
            Es = np.empty_like(E)
            Es[smooth - 1:] = (c[smooth:] - c[:-smooth]) / smooth
            Es[: smooth - 1] = c[1:smooth] / np.arange(1, smooth)
            E = Es
        return np.abs((E - self.E0) / abs(self.E0))

    def t_reach(self, eps, smooth=10):
        """First iteration at which the (smoothed) relative error is below eps (nan if never)."""
        r = self.rel(smooth)
        hit = np.flatnonzero(r < eps)
        return float(hit[0]) if len(hit) else np.nan

    @property
    def rel_final(self):
        v = self.final.get("rel_err", np.nan)
        return np.nan if v is None else float(v)

    @property
    def collapsed(self):
        """The wave function collapsed on one configuration (E = +J N, zero variance) or diverged."""
        return self.meta["diverged"] or not np.isfinite(self.rel_final) or self.final.get("E", 0) > 0


def load(study, **filters):
    runs, seen = [], set()
    for cfg in studies.STUDIES[study]():
        if any(cfg.get(k) != v for k, v in filters.items()):
            continue
        p = result_path(cfg)
        if p.exists() and p not in seen:  # a study can list the same configuration twice
            seen.add(p)
            runs.append(Run(cfg, p))
    return runs


def select(runs, **kw):
    out = []
    for r in runs:
        ok = True
        for k, v in kw.items():
            val = r.arch if k == "arch" else r.cfg[k]
            if isinstance(v, float) and isinstance(val, float):
                ok &= bool(np.isclose(val, v))
            else:
                ok &= val == v
        if ok:
            out.append(r)
    return out


def fit_rate(rel, hi=1e-2, lo=1e-4, min_pts=4):
    """Exponential decay rate (per iteration) of a relative-error curve, fitted where the error
    goes from hi down to lo (the range that matters in practice; it avoids the initial transient
    and the late tails set by the accuracy limit of the Ansatz or by tiny symmetry-breaking components)."""
    rel = np.asarray(rel)
    if not np.all(np.isfinite(rel)):
        return np.nan
    a = np.flatnonzero(rel < hi)
    if not len(a):
        return np.nan
    i0 = a[0]
    b = np.flatnonzero((rel < lo) & (np.arange(len(rel)) > i0))
    if not len(b):
        return np.nan
    i1 = b[0] + 1
    if i1 - i0 < min_pts:
        return np.nan
    t = np.arange(i0, i1)
    return -np.polyfit(t, np.log(rel[i0:i1]), 1)[0]


def ite_rate(lr, gap):
    """Predicted decay rate per iteration of E - E0 for SR = Euler step of imaginary time dtau = 2 lr:
    each step multiplies the excited amplitude by (1 - 2 lr gap), the energy error by its square.
    For small lr this is 4 lr gap."""
    x = 2 * lr * gap
    return -2 * np.log(1 - x) if x < 1 else np.inf


def strip(ax, x, ys, color, marker="o", size=22, jitter=0.12, label=None, median=True, hollow=None):
    """All individual runs as dots at position x (slightly jittered), plus a short median bar."""
    ys = np.asarray(ys, dtype=float)
    rng = np.random.default_rng(int(abs(x) * 1000) % 2**31)
    xs = x + rng.uniform(-jitter, jitter, len(ys))
    hollow = np.zeros(len(ys), bool) if hollow is None else np.asarray(hollow)
    ax.scatter(xs[~hollow], ys[~hollow], s=size, color=color, edgecolor=SURFACE, linewidth=0.6, zorder=3,
               marker=marker, label=label)
    if hollow.any():
        ax.scatter(xs[hollow], ys[hollow], s=size, facecolor="none", edgecolor=color, linewidth=1.0, zorder=3,
                   marker=marker, label=None if (~hollow).any() else label)
    if median and np.isfinite(ys).any():
        ax.plot([x - 0.25, x + 0.25], [np.nanmedian(ys)] * 2, color=INK, lw=1.4, zorder=4)
