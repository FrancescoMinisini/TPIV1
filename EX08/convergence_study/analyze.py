"""Figures and summary tables for all studies.

    python analyze.py            # every study that has results
    python analyze.py S1 S3      # only some
"""

import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from analysis_common import (
    ARCH_COLOR, ARCH_LABEL, CAT, INK, INK2, MUTED, SGD_COLOR, fit_rate, ite_rate, load, ramp, save, select, strip,
)
from studies import ARCHS, LRS, SHIFTS

EPS = 2e-2  # "converged": smoothed relative error below 2 % (separates the two FFNN minima, ~1.3 % vs ~5.5 %)
EPS2 = 2e-3  # "accurate": below 0.2 %
TABLES = []


def table(title, header, rows):
    lines = [f"### {title}", "", "| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    TABLES.append("\n".join(lines) + "\n")


def fmt(x, p=2):
    return "–" if x is None or not np.isfinite(x) else f"{x:.{p}g}"


def opt_label(r):
    c = r.cfg
    name = "Adam" if c["optimizer"] == "adam" else "SGD"
    return name + (f" + SR (shift {c['diag_shift']:g})" if c["sr"] else "")


def rel_axis(ax, lo=1e-5, hi=0.5):
    ax.set_yscale("log")
    ax.set_ylim(lo, hi)


# ====================================================================== S1: optimizer / SR grid


def S1():
    runs = load("S1_optimizer")
    if not runs:
        return
    variants = [("SGD", dict(optimizer="sgd", sr=False), SGD_COLOR)] + [
        (f"SR, shift {ds:g}", dict(optimizer="sgd", sr=True, diag_shift=ds), c) for ds, c in zip(SHIFTS, ramp(len(SHIFTS))[::-1])
    ]

    # --- trajectories: one figure per Ansatz, rows = optimizer variant, columns = learning rate, every seed shown
    for arch in ARCHS:
        fig, axes = plt.subplots(len(variants), len(LRS), figsize=(13, 11), sharex=True, sharey=True)
        for i, (vname, sel, color) in enumerate(variants):
            for j, lr in enumerate(LRS):
                ax = axes[i, j]
                for r in select(runs, arch=arch, lr=lr, **sel):
                    ax.plot(r.rel(5), color=color, lw=0.9, alpha=0.85)
                rel_axis(ax, 1e-4, 0.5)
                ax.axhline(EPS, color=MUTED, lw=0.8, ls=":")
                if i == 0:
                    ax.set_title(f"learning rate {lr:g}")
                if j == 0:
                    ax.set_ylabel(f"{vname}\n|E-E0|/|E0|")
                if i == len(variants) - 1:
                    ax.set_xlabel("iteration")
        fig.suptitle(f"{ARCH_LABEL[arch]}, N=20, h=1: relative error during training "
                     f"(each line one seed, 5-iteration moving average; dotted: 2 %)", x=0.5)
        fig.tight_layout()
        save(fig, f"S1_trajectories_{arch}")

    # --- summary: final error, time to 2 % and to 0.2 %, late-time fluctuations; every run shown
    fig, axes = plt.subplots(4, 3, figsize=(15, 13), sharey="row")
    for col, arch in enumerate(ARCHS):
        rows_tab = []
        x = 0
        ticks = []
        for lr in LRS:
            for vname, sel, color in variants:
                rs = select(runs, arch=arch, lr=lr, **sel)
                if not rs:
                    continue
                fe = np.array([r.rel_final if not r.collapsed else 0.9 for r in rs])
                tc = np.array([r.t_reach(EPS) for r in rs])
                tc2 = np.array([r.t_reach(EPS2) for r in rs])
                late = np.array([np.std(r.E[-200:]) / abs(r.E0) if not r.collapsed else np.nan for r in rs])
                strip(axes[0, col], x, fe, color, hollow=[r.collapsed for r in rs])
                strip(axes[1, col], x, np.where(np.isfinite(tc), tc, 1100), color, hollow=~np.isfinite(tc))
                strip(axes[2, col], x, np.where(np.isfinite(tc2), tc2, 1100), color, hollow=~np.isfinite(tc2))
                strip(axes[3, col], x, late, color)
                ok = np.array([not r.collapsed for r in rs])
                fe_ok = fe[ok] if ok.any() else np.array([np.nan])
                rows_tab.append((f"{lr:g}", vname, len(rs), f"{(~ok).sum()}", fmt(np.nanmedian(fe_ok) * 100), fmt(np.nanmin(fe_ok) * 100),
                                 fmt(np.nanmax(fe_ok) * 100),
                                 f"{np.isfinite(tc).sum()}/{len(rs)}", fmt(np.nanmedian(tc), 3),
                                 f"{np.isfinite(tc2).sum()}/{len(rs)}", fmt(np.nanmedian(tc2), 3)))
                ticks.append(x)
                x += 1
            for ax in axes[:, col]:
                ax.axvline(x - 0.5 + 0.3, color=MUTED, lw=0.6)
            axes[3, col].text(x - len(variants) / 2 - 0.5, -0.1, f"lr {lr:g}", transform=axes[3, col].get_xaxis_transform(),
                              ha="center", va="top", fontsize=9, color=INK2)
            x += 0.6
        for ax in axes[:, col]:
            ax.set_xticks(ticks)
            ax.set_xticklabels([""] * len(ticks))
            ax.grid(axis="x", visible=False)
        axes[0, col].set_title(ARCH_LABEL[arch])
        table(f"S1 – {ARCH_LABEL[arch]} (N=20, h=1, 1000 iterations)",
              ["lr", "optimizer", "seeds", "collapsed", "median final rel. err. [%] (non-collapsed)", "best seed [%]", "worst seed [%]",
               "seeds reaching 2 %", "median it. to 2 %", "seeds reaching 0.2 %", "median it. to 0.2 %"], rows_tab)
    for ax in axes[0]:
        rel_axis(ax, 1e-5, 1.2)
        ax.axhline(EPS, color=MUTED, lw=0.8, ls=":")
    for ax in axes[1:3].ravel():
        ax.set_ylim(-30, 1150)
        ax.axhline(1000, color=MUTED, lw=0.8, ls=":")
    axes[3, 0].set_yscale("log")
    axes[0, 0].set_ylabel("final |E-E0|/|E0|\n(16k samples; hollow = collapsed)")
    axes[1, 0].set_ylabel("iterations to reach 2 %\n(hollow at top = never)")
    axes[2, 0].set_ylabel("iterations to reach 0.2 %\n(hollow at top = never)")
    axes[3, 0].set_ylabel("std of E over last 200 it. / |E0|\n(late-time fluctuations)")
    handles = [Line2D([], [], marker="o", ls="", color=c, label=v) for v, _, c in variants] + [
        Line2D([], [], color=INK, lw=1.4, label="median over seeds")]
    fig.legend(handles=handles, loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Optimizer study, N=20, h=1, 1000 iterations: every dot is one seed")
    fig.tight_layout(rect=(0, 0.025, 1, 1))
    save(fig, "S1_summary")

    # --- Adam vs SGD, with and without SR, all three Ansatz families
    combos = [("SGD", dict(optimizer="sgd", sr=False, lr=0.01), SGD_COLOR),
              ("SGD + SR", dict(optimizer="sgd", sr=True, diag_shift=0.01, lr=0.01), CAT[0]),
              ("Adam lr 0.001", dict(optimizer="adam", sr=False, lr=0.001), CAT[3]),
              ("Adam lr 0.01", dict(optimizer="adam", sr=False, lr=0.01), CAT[1]),
              ("Adam lr 0.001 + SR", dict(optimizer="adam", sr=True, lr=0.001), CAT[6]),
              ("Adam lr 0.01 + SR", dict(optimizer="adam", sr=True, lr=0.01), CAT[7])]
    fig, axes = plt.subplots(len(ARCHS), len(combos), figsize=(17, 8), sharex=True, sharey=True)
    rows_tab = []
    for i, arch in enumerate(ARCHS):
        for j, (name, sel, color) in enumerate(combos):
            ax = axes[i, j]
            rs = select(runs, arch=arch, **sel)
            for r in rs:
                ax.plot(r.rel(5), color=color, lw=0.9)
            rel_axis(ax, 1e-4, 1.2)
            ax.axhline(EPS, color=MUTED, lw=0.8, ls=":")
            if i == 0:
                ax.set_title(name)
            if j == 0:
                ax.set_ylabel(f"{ARCH_LABEL[arch]}\n|E-E0|/|E0|")
            if i == len(ARCHS) - 1:
                ax.set_xlabel("iteration")
            fe = [r.rel_final for r in rs]
            rows_tab.append((ARCH_LABEL[arch], name, " / ".join(fmt(100 * v) for v in fe)))
    fig.suptitle("SGD vs Adam, with and without SR (N=20, h=1, SR shift 0.01; every line one seed)")
    fig.tight_layout()
    save(fig, "S1_adam_vs_sgd")
    table("S1 – Adam vs SGD: final relative error per seed [%]", ["Ansatz", "optimizer", "seeds 0/1/2/3"], rows_tab)


# ====================================================================== S3: noise-free optimisation, N = 12


def S3():
    runs = load("S3_fullsum")
    if not runs:
        return
    rbm = [r for r in runs if r.arch == "rbm" and r.cfg["sr"]]

    # --- (a) SR follows exact imaginary-time evolution: one panel per h
    hs = [0.3, 0.5, 0.7, 0.85, 1.0, 1.2, 1.5, 2.0, 3.0]
    fig, axes = plt.subplots(3, 3, figsize=(13, 10), sharex=True)
    for ax, h in zip(axes.ravel(), hs):
        rs = select(rbm, N=12, h=h, lr=0.01, diag_shift=1e-4)
        for k, r in enumerate(rs):
            ax.plot(r.rel(), color=CAT[2], lw=1.2, label="RBM + SR (each seed)" if k == 0 else None)
            ax.plot(np.abs((r.a["ite_E"] - r.E0) / r.E0), color=INK, lw=0.9, ls="--",
                    label="exact imaginary time, same start" if k == 0 else None)
        if rs:
            ex = rs[0].ex
            t = np.arange(600)
            ax.plot(t, 0.3 * np.exp(-ite_rate(0.01, ex["gap_sym"]) * t), color=CAT[1], lw=1, ls=":",
                    label=r"slope $\approx e^{-4\eta\Delta t}$")
            ax.set_title(f"h = {h:g}   (Δ = {ex['gap_sym']:.3f})")
        rel_axis(ax, 1e-9, 1)
        ax.set_ylabel("|E-E0|/|E0|")
    for ax in axes[-1]:
        ax.set_xlabel("iteration (learning rate η = 0.01)")
    axes[0, 0].legend(loc="lower left")
    fig.suptitle("N=12, exact sums (no Monte Carlo noise): SR (shift 1e-4) tracks imaginary-time evolution with dτ = 2η.\n"
                 "Δ = gap to the first excited state with the symmetry of the initial state")
    fig.tight_layout()
    save(fig, "S3a_sr_vs_imaginary_time")

    # --- (b) measured decay rate vs the gap prediction 4 η Δ
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    pts = []
    for r in rbm:
        if r.cfg["diag_shift"] != 1e-4:
            continue
        rate = fit_rate(r.rel())
        rate_ite = fit_rate(np.abs((r.a["ite_E"] - r.E0) / r.E0))
        pts.append((r.cfg["N"], r.cfg["h"], r.cfg["lr"], r.ex["gap_sym"], r.ex["gap1"], rate, rate_ite))
    pts = np.array(pts)
    if len(pts):
        x = np.array([ite_rate(p[2], p[3]) for p in pts])
        groups = [("vary h (N=12, η=0.01)", (pts[:, 0] == 12) & np.isclose(pts[:, 2], 0.01), CAT[0], "o"),
                  ("vary η (N=12, h∈{0.5,1,2})", (pts[:, 0] == 12) & ~np.isclose(pts[:, 2], 0.01), CAT[1], "s"),
                  ("vary N (h∈{0.5,1}, η=0.01)", pts[:, 0] != 12, CAT[2], "D")]
        for name, m, c, mk in groups:
            axes[0].scatter(x[m], pts[m, 5], color=c, marker=mk, s=30, label=name, zorder=3)
            axes[0].scatter(x[m], pts[m, 6], facecolor="none", edgecolor=c, marker=mk, s=60, lw=0.8, zorder=2)
        lim = [min(x.min(), np.nanmin(pts[:, 5])) * 0.7, max(x.max(), np.nanmax(pts[:, 5])) * 1.4]
        axes[0].plot(lim, lim, color=INK, lw=0.9, ls="--", label="measured = predicted")
        axes[0].set_xscale("log")
        axes[0].set_yscale("log")
        axes[0].set_xlabel("predicted rate  -2 ln(1-2ηΔ) ≈ 4ηΔ  (per iteration)")
        axes[0].set_ylabel("measured decay rate of E - E0 (per iteration)\nfitted between 1e-2 and 1e-4")
        axes[0].set_title("filled: RBM + SR,  hollow: exact imaginary time")
        axes[0].legend(fontsize=7)
        rows_tab = [(int(p[0]), f"{p[1]:g}", f"{p[2]:g}", f"{p[3]:.3f}", f"{p[4]:.3g}", f"{ite_rate(p[2], p[3]):.4f}",
                     fmt(p[5], 3), fmt(p[6], 3)) for p in pts[np.lexsort((pts[:, 2], pts[:, 1], pts[:, 0]))]]
        table("S3 – decay rate of the energy error vs gap (RBM α=2, SR shift 1e-4, exact sums)",
              ["N", "h", "η", "Δ (sym. sector)", "E1-E0", "predicted rate", "rate SR", "rate exact ITE"], rows_tab)

    # gap vs h and N from exact diagonalisation
    import json
    from vmc_lib import EXACT_DIR
    exs = [json.loads(p.read_text()) for p in EXACT_DIR.glob("*.json")]
    for N, c in zip([12, 20], [CAT[0], CAT[1]]):
        e = sorted([x for x in exs if x["N"] == N], key=lambda x: x["h"])
        if e:
            axes[1].plot([x["h"] for x in e], [x["gap_sym"] for x in e], "o-", color=c, label=f"Δ (symmetric sector), N={N}")
            axes[1].plot([x["h"] for x in e], [x["gap1"] for x in e], "x--", color=c, lw=0.9, label=f"E1-E0, N={N}")
    axes[1].set_xlabel("transverse field h")
    axes[1].set_ylabel("gap")
    axes[1].set_yscale("log")
    axes[1].set_title("Gaps from exact diagonalisation")
    axes[1].legend(fontsize=7)
    for h, c in zip([0.5, 1.0], [CAT[0], CAT[1]]):
        e = sorted([x for x in exs if np.isclose(x["h"], h)], key=lambda x: x["N"])
        axes[2].plot([x["N"] for x in e], [x["gap_sym"] for x in e], "o-", color=c, label=f"Δ symmetric sector, h={h:g}")
    Ns = np.array([6, 8, 10, 12, 14, 20])
    axes[2].plot(Ns, 8 * np.sin(np.pi / (2 * Ns)), color=INK, ls=":", lw=1, label="8 sin(π/2N)  (critical, free fermions)")
    axes[2].set_xlabel("N")
    axes[2].set_ylabel("gap")
    axes[2].set_title("Gap vs system size")
    axes[2].legend(fontsize=7)
    fig.tight_layout()
    save(fig, "S3b_rate_vs_gap")

    # --- (c) learning rate: iterations to converge ~ 1 / eta, error vs eta * t collapses
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    cols = ramp(len(LRS))
    for ax, h in zip(axes, [0.5, 1.0, 2.0]):
        for lr, c in zip(LRS, cols):
            for k, r in enumerate(select(rbm, N=12, h=h, lr=lr, diag_shift=1e-4)):
                t = np.arange(len(r.E)) * lr
                ax.plot(t, r.rel(), color=c, lw=1.1, label=f"η = {lr:g}" if k == 0 else None)
        rel_axis(ax, 1e-9, 1)
        ax.set_xlim(0, 6)
        ax.set_xlabel("η × iteration  (= imaginary time / 2)")
        ax.set_title(f"h = {h:g}")
    axes[0].set_ylabel("|E-E0|/|E0|")
    axes[0].legend()
    fig.suptitle("RBM + SR (shift 1e-4), N=12, exact sums: for η ≤ 0.03 the curves collapse when plotted against η·t; "
                 "η = 0.1 (darkest) is unstable and never converges")
    fig.tight_layout()
    save(fig, "S3c_learning_rate_collapse")

    # --- (d) diagonal shift, three Ansatz families
    shifts = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]
    archs = ["ffnn", "ffnn_lin", "rbm"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    rows_tab = []
    for j, arch in enumerate(archs):
        ax = axes[0, j]
        for ds, c in zip(shifts, ramp(len(shifts))[::-1]):
            rs = select(runs, N=12, h=1.0, lr=0.01, sr=True, diag_shift=ds, arch=arch)
            for k, r in enumerate(rs):
                ax.plot(r.rel(), color=c, lw=1, label=f"shift {ds:g}" if k == 0 else None)
            fe = [r.rel_final for r in rs]
            rows_tab.append((ARCH_LABEL[arch], f"{ds:g}", " / ".join(fmt(100 * v) for v in fe),
                             " / ".join(fmt(r.t_reach(EPS, 1), 3) for r in rs)))
            strip(axes[1, j], np.log10(ds), fe, ARCH_COLOR[arch], jitter=0.08)
        sgd = select(runs, N=12, h=1.0, lr=0.01, sr=False, arch=arch)
        for k, r in enumerate(sgd):
            ax.plot(r.rel()[:600], color=SGD_COLOR, lw=1, ls="--", label="SGD, no SR" if k == 0 else None)
        strip(axes[1, j], 1.0, [r.rel()[599] for r in sgd], SGD_COLOR, jitter=0.08)
        rel_axis(ax, 1e-8, 1)
        ax.set_title(ARCH_LABEL[arch] + (" (width 6)" if arch != "rbm" else " (α = 2)"))
        ax.set_xlabel("iteration (η = 0.01)")
        rel_axis(axes[1, j], 1e-8, 1)
        axes[1, j].set_xticks([-5, -4, -3, -2, -1, 0, 1])
        axes[1, j].set_xticklabels(["1e-5", "1e-4", "1e-3", "0.01", "0.1", "1", "SGD"])
        axes[1, j].set_xlabel("diagonal shift")
    axes[0, 0].set_ylabel("|E-E0|/|E0|")
    axes[1, 0].set_ylabel("|E-E0|/|E0| after 600 iterations")
    axes[0, 2].legend(fontsize=7)
    fig.suptitle("Effect of the SR diagonal shift (N=12, h=1, exact sums, 3 seeds each)")
    fig.tight_layout()
    save(fig, "S3d_diag_shift")
    table("S3 – diagonal shift (N=12, h=1, exact sums, η=0.01): final rel. error [%] and iterations to 2 % per seed",
          ["Ansatz", "shift", "final error [%]", "iterations to 2 %"], rows_tab)

    # --- (e) plain SGD without noise: plateaus are deterministic; time scales as 1/eta
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharey=True)
    for ax, arch in zip(axes, archs):
        for lr, c in zip(LRS, ramp(len(LRS))):
            for k, r in enumerate(select(runs, N=12, h=1.0, sr=False, lr=lr, arch=arch)):
                ax.plot(np.arange(len(r.E)) * lr, r.rel(), color=c, lw=1, label=f"η = {lr:g}" if k == 0 else None)
        rel_axis(ax, 1e-5, 1)
        ax.set_xscale("log")
        ax.set_xlim(0.1, 200)
        ax.set_xlabel("η × iteration  (gradient-flow time)")
        ax.set_title(ARCH_LABEL[arch])
    axes[0].set_ylabel("|E-E0|/|E0|")
    axes[0].legend()
    fig.suptitle("Plain SGD without Monte Carlo noise (N=12, h=1, 4 seeds per η): plateaus are a property of the landscape")
    fig.tight_layout()
    save(fig, "S3e_sgd_noise_free")


# ====================================================================== S7: plateau escape statistics


def S7():
    runs = load("S7_plateaus")
    if not runs:
        return
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    rows_tab = []
    for j, arch in enumerate(ARCHS):
        ax = axes[0, j]
        sgd = select(runs, arch=arch, sr=False)
        sr = select(runs, arch=arch, sr=True)
        for k, r in enumerate(sgd):
            ax.plot(r.rel(10), color=SGD_COLOR, lw=0.8, label="SGD η=0.01 (12 seeds)" if k == 0 else None)
        for k, r in enumerate(sr):
            ax.plot(r.rel(10), color=ARCH_COLOR[arch], lw=0.9, label="SGD + SR η=0.01, shift 0.01 (12 seeds)" if k == 0 else None)
        rel_axis(ax, 1e-4, 0.5)
        ax.set_xscale("symlog", linthresh=100)
        ax.set_xlim(0, 3000)
        ax.axhline(EPS, color=MUTED, lw=0.8, ls=":")
        ax.set_title(ARCH_LABEL[arch])
        ax.set_xlabel("iteration")
        ax.legend(fontsize=7, loc="lower left")
        ax = axes[1, j]
        for rs, c, name in [(sgd, SGD_COLOR, "SGD"), (sr, ARCH_COLOR[arch], "SGD + SR")]:
            for eps, ls in [(1e-1, ":"), (EPS, "-"), (EPS2, "--")]:
                t = np.sort([r.t_reach(eps) for r in rs])
                t = t[np.isfinite(t)]
                n = len(rs)
                tt = np.concatenate([[0], np.repeat(t, 2), [3000]])
                ff = np.concatenate([[0, 0], np.repeat(np.arange(1, len(t) + 1) / n, 2)])[: len(tt)]
                ax.plot(tt, ff, color=c, ls=ls, lw=1.3, label=f"{name}, error < {eps * 100:g} %")
                rows_tab.append((ARCH_LABEL[arch], name, f"{eps * 100:g} %", f"{len(t)}/{n}",
                                 fmt(np.median(t), 3) if len(t) else "–", fmt(t.min(), 3) if len(t) else "–",
                                 fmt(t.max(), 3) if len(t) else "–"))
        ax.set_xscale("symlog", linthresh=100)
        ax.set_xlim(0, 3000)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("iteration")
        ax.set_ylabel("fraction of seeds below threshold")
        ax.legend(fontsize=7, loc="upper left")
    fig.suptitle("Plateau escape: 12 seeds each, N=20, h=1 (SGD runs 3000 iterations, SR runs 1000)")
    fig.tight_layout()
    save(fig, "S7_plateau_escape")
    table("S7 – time to reach a given relative error (12 seeds, N=20, h=1, η=0.01)",
          ["Ansatz", "optimizer", "threshold", "seeds reaching it", "median iteration", "fastest", "slowest"], rows_tab)



# ====================================================================== S2: number of samples


def S2():
    import studies
    from studies import SGD_REF, SR_REF
    from vmc_lib import run_id
    s1_ids = {run_id(c) for c in studies.S1_optimizer()}
    runs = load("S2_samples")
    if not runs:
        return
    Nss = [128, 256, 512, 1024, 2048, 4096]
    opts = [("SGD", SGD_REF), ("SGD + SR", SR_REF)]
    fig, axes = plt.subplots(3, 2, figsize=(13, 11), sharex=True)
    rows_tab = []
    for j, (oname, opt) in enumerate(opts):
        for k, arch in enumerate(ARCHS):
            off = (k - 1) * 0.07
            for Ns in Nss:
                rs = select(runs, arch=arch, n_samples=Ns, **opt)
                if not rs:
                    continue
                x = np.log2(Ns) + off
                fe = [r.rel_final if not r.collapsed else 0.9 for r in rs]
                tc = [r.t_reach(EPS) for r in rs]
                strip(axes[0, j], x, fe, ARCH_COLOR[arch], jitter=0.03, median=False,
                      label=ARCH_LABEL[arch] if Ns == Nss[0] else None)
                strip(axes[1, j], x, np.where(np.isfinite(tc), tc, 1100), ARCH_COLOR[arch], jitter=0.03, median=False,
                      hollow=~np.isfinite(tc))
                shared = [run_id(r.cfg) in s1_ids for r in rs]
                strip(axes[2, j], x, [r.meta["time_per_iter"] * 1e3 for r in rs], ARCH_COLOR[arch], jitter=0.03, median=False,
                      hollow=shared)
                rows_tab.append((oname, ARCH_LABEL[arch], Ns, " / ".join(fmt(100 * v) for v in fe),
                                 " / ".join(fmt(t, 3) for t in tc), fmt(np.median([r.meta["time_per_iter"] for r in rs]) * 1e3, 3)))
        axes[0, j].set_title(oname + (f" (η={opt['lr']:g}, shift {opt['diag_shift']:g})" if opt["sr"] else f" (η={opt['lr']:g})"))
        rel_axis(axes[0, j], 1e-5, 1.2)
        axes[0, j].axhline(EPS, color=MUTED, lw=0.8, ls=":")
        axes[1, j].set_ylim(-30, 1150)
        axes[2, j].set_yscale("log")
        axes[2, j].set_xticks(np.log2(Nss))
        axes[2, j].set_xticklabels([str(n) for n in Nss])
        axes[2, j].set_xlabel("samples per iteration")
    axes[0, 0].set_ylabel("final |E-E0|/|E0|")
    axes[1, 0].set_ylabel("iterations to reach 2 %\n(hollow at top = never)")
    axes[2, 0].set_ylabel("ms per iteration, one core per run\n(hollow: run shared with S1, measured\nunder a heavier machine load)")
    axes[0, 0].legend(loc="lower left")
    fig.suptitle("Number of Monte Carlo samples per iteration (N=20, h=1, 1000 iterations, 3 seeds; every dot one run)")
    fig.tight_layout()
    save(fig, "S2_samples")
    table("S2 – samples per iteration: final rel. error [%] and iterations to 2 % per seed",
          ["optimizer", "Ansatz", "samples", "final error [%]", "iterations to 2 %", "ms/iteration"], rows_tab)


# ====================================================================== S4: network size


def S4():
    import json
    from studies import SGD_REF, SR_REF
    from vmc_lib import RESULTS
    runs = load("S4_size")
    if not runs:
        return
    timing = {}
    tp = RESULTS / "timing.json"
    if tp.exists():
        for t in json.loads(tp.read_text()):
            timing[(t["model"], t["width"], t["depth"], t["alpha"], t["sr"])] = t
    families = [("FFNN, ReLU output, depth 1", dict(model="ffnn", out_act="relu", depth=1), CAT[0], "o"),
                ("FFNN, ReLU output, depth 2", dict(model="ffnn", out_act="relu", depth=2), CAT[0], "s"),
                ("FFNN, ReLU output, depth 3", dict(model="ffnn", out_act="relu", depth=3), CAT[0], "^"),
                ("FFNN, linear output, depth 1", dict(model="ffnn", out_act="linear", depth=1), CAT[1], "o"),
                ("FFNN, linear output, depth 2", dict(model="ffnn", out_act="linear", depth=2), CAT[1], "s"),
                ("FFNN, linear output, depth 3", dict(model="ffnn", out_act="linear", depth=3), CAT[1], "^"),
                ("RBM", dict(model="rbm"), CAT[2], "D")]
    opts = [("SGD", SGD_REF), ("SGD + SR", SR_REF)]

    def size_key(r):
        return r.cfg["alpha"] if r.cfg["model"] == "rbm" else r.cfg["width"]

    # --- quality, speed, stability, cost vs number of parameters; every seed shown
    fig, axes = plt.subplots(4, 2, figsize=(14, 16), sharex=True)
    rows_tab = []
    for j, (oname, opt) in enumerate(opts):
        for fname, sel, color, mk in families:
            rs = select(runs, **sel, **opt)
            sizes = sorted({size_key(r) for r in rs})
            med = []
            for sz in sizes:
                g = [r for r in rs if size_key(r) == sz]
                npar = g[0].meta["n_params"]
                fe = np.array([r.rel_final if not r.collapsed else 0.9 for r in g])
                tc = np.array([r.t_reach(EPS) for r in g])
                late = np.array([np.std(r.E[-200:]) / abs(r.E0) if not r.collapsed else np.nan for r in g])
                c = g[0].cfg
                tkey = (c["model"], c["width"], c["depth"], c["alpha"], c["sr"])
                t_it = timing[tkey]["t_iter"] if tkey in timing else np.median([r.meta["time_per_iter"] for r in g])
                for ax, ys, hol in [(axes[0, j], fe, np.array([r.collapsed for r in g])),
                                    (axes[1, j], np.where(np.isfinite(tc), tc, 1100), ~np.isfinite(tc)),
                                    (axes[2, j], late, None)]:
                    xs = npar * np.exp(np.random.default_rng(npar).uniform(-0.06, 0.06, len(ys)))
                    hol = np.zeros(len(ys), bool) if hol is None else hol
                    ax.scatter(xs[~hol], ys[~hol], color=color, marker=mk, s=20, edgecolor="#fcfcfb", lw=0.5, zorder=3)
                    ax.scatter(xs[hol], ys[hol], facecolor="none", edgecolor=color, marker=mk, s=20, lw=0.9, zorder=3)
                axes[3, j].scatter([npar], [t_it * 1e3], color=color, marker=mk, s=26, zorder=3)
                med.append((npar, np.median(fe), t_it))
                rows_tab.append((oname, fname, sz, npar, " / ".join(fmt(100 * v) for v in fe),
                                 f"{np.isfinite(tc).sum()}/{len(g)}", fmt(np.nanmedian(tc), 3), fmt(t_it * 1e3, 3)))
            if med:
                med = np.array(med)
                ls = {"o": "-", "s": "--", "^": ":", "D": "-"}[mk]
                axes[0, j].plot(med[:, 0], med[:, 1], color=color, lw=1, ls=ls, label=fname)
                axes[3, j].plot(med[:, 0], med[:, 2] * 1e3, color=color, lw=1, ls=ls)
        axes[0, j].set_title(oname + (f" (η={opt['lr']:g}, shift {opt['diag_shift']:g})" if opt["sr"] else f" (η={opt['lr']:g})"))
        rel_axis(axes[0, j], 1e-5, 1.2)
        axes[0, j].axhline(EPS, color=MUTED, lw=0.8, ls=":")
        axes[1, j].set_ylim(-30, 1150)
        axes[2, j].set_yscale("log")
        axes[3, j].set_yscale("log")
        axes[3, j].set_xscale("log")
        axes[3, j].set_xlabel("number of variational parameters")
    axes[0, 0].set_ylabel("final |E-E0|/|E0|  (line: median)\nhollow = collapsed/diverged")
    axes[1, 0].set_ylabel("iterations to reach 2 %\n(hollow at top = never)")
    axes[2, 0].set_ylabel("std of E over last 200 iterations / |E0|")
    axes[3, 0].set_ylabel("ms per iteration (single process, one core)\nFFNN: same cost for both output activations")
    axes[0, 1].legend(fontsize=7, loc="lower left")
    fig.suptitle("Network size, N=20, h=1, 4 seeds (SGD: 1000 iterations, SR: 400); FFNN widths 2-64 x depths 1-3, RBM α = 0.25-8")
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    save(fig, "S4_size")
    table("S4 – network size: final rel. error per seed [%], seeds reaching 2 %, median iterations to 2 %, cost",
          ["optimizer", "Ansatz", "width / α", "parameters", "final error [%]", "reach 2 %", "median it. to 2 %", "ms/iteration"], rows_tab)

    # --- accuracy vs compute: final error against time-to-solution (iterations to 2 % x time per iteration)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
    for j, (oname, opt) in enumerate(opts):
        ax = axes[j]
        for fname, sel, color, mk in families:
            for r in select(runs, **sel, **opt):
                c = r.cfg
                tkey = (c["model"], c["width"], c["depth"], c["alpha"], c["sr"])
                t_it = timing[tkey]["t_iter"] if tkey in timing else r.meta["time_per_iter"]
                tc = r.t_reach(EPS)
                x = (tc if np.isfinite(tc) else c["n_iter"]) * t_it
                ax.scatter(x, r.rel_final if not r.collapsed else 0.9, color=color, marker=mk, s=18,
                           facecolor=color if np.isfinite(tc) else "none", lw=0.8)
        ax.set_xscale("log")
        rel_axis(ax, 1e-5, 1.2)
        ax.axhline(EPS, color=MUTED, lw=0.8, ls=":")
        ax.set_xlabel("compute time to reach 2 % [s]  (hollow: never reached, plotted at the full run length)")
        ax.set_title(oname)
    axes[0].set_ylabel("final |E-E0|/|E0|")
    handles = [Line2D([], [], marker=mk, ls="", color=c, label=f) for f, _, c, mk in families]
    axes[1].legend(handles=handles, fontsize=7, loc="upper right")
    fig.suptitle("Cost vs accuracy, N=20, h=1: every point one run (single-core cost per iteration; SGD 1000 iterations, SR 400)")
    fig.tight_layout()
    save(fig, "S4_cost_vs_accuracy")


# ====================================================================== S5: field scan / phase transition


def S5():
    import json
    from studies import H_SCAN, SGD_REF, SR_REF
    from vmc_lib import EXACT_DIR
    runs = load("S5_phase")
    if not runs:
        return
    ex = {(d["N"], d["h"]): d for d in (json.loads(p.read_text()) for p in EXACT_DIR.glob("*.json"))}
    methods = [("FFNN (ReLU out) + SGD", dict(arch="ffnn", sr=False, N=20), SGD_COLOR, "o"),
               ("FFNN (ReLU out) + SR", dict(arch="ffnn", sr=True, N=20), CAT[0], "s"),
               ("RBM + SGD", dict(arch="rbm", sr=False, N=20), CAT[3], "^"),
               ("RBM + SR", dict(arch="rbm", sr=True, N=20), CAT[2], "D")]
    xs = {h: i for i, h in enumerate(H_SCAN)}

    # --- convergence across h: training curves (small multiples) and final errors
    fig, axes = plt.subplots(len(methods), len(H_SCAN), figsize=(22, 9), sharex=True, sharey=True)
    for i, (mname, sel, color, _) in enumerate(methods):
        for j, h in enumerate(H_SCAN):
            ax = axes[i, j]
            for r in select(runs, h=h, **sel):
                ax.plot(r.rel(5), color=color, lw=0.9)
            rel_axis(ax, 1e-5, 1)
            ax.axhline(EPS, color=MUTED, lw=0.6, ls=":")
            if i == 0:
                ax.set_title(f"h = {h:g}")
            if j == 0:
                ax.set_ylabel(mname, fontsize=8)
            if i == len(methods) - 1:
                ax.set_xlabel("iteration")
    fig.suptitle("Training curves across the transition (N=20, 4 seeds per panel, |E-E0|/|E0|)")
    fig.tight_layout()
    save(fig, "S5_trajectories_vs_h")

    # --- physics + sampling diagnostics vs h
    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    axes = axes.ravel()
    rows_tab = []
    for k, (mname, sel, color, mk) in enumerate(methods + [("RBM + SR, N=12", dict(arch="rbm", sr=True, N=12), CAT[6], "v")]):
        off = (k - 2) * 0.08
        for h in H_SCAN:
            rs = [r for r in select(runs, h=h, **sel) if not r.collapsed]
            if not rs:
                continue
            x = xs[h] + off
            def sc(ax, ys):
                ax.scatter(np.full(len(ys), x), ys, color=color, marker=mk, s=16, zorder=3,
                           label=mname if h == H_SCAN[0] else None)
            sc(axes[0], [r.rel_final for r in rs])
            sc(axes[1], [r.final["ms2"] for r in rs])
            sc(axes[2], [r.final["acc"] for r in rs])
            sc(axes[3], [r.final["R_hat"] for r in rs])
            sc(axes[4], [r.final["ms_R_hat"] for r in rs])
            sc(axes[5], [r.final["tau"] for r in rs])
            sc(axes[6], [r.final.get("ms_tau", np.nan) for r in rs])
            sc(axes[7], [r.t_reach(EPS) for r in rs])
            rows_tab.append((mname, f"{h:g}", " / ".join(fmt(100 * r.rel_final) for r in rs),
                             " / ".join(fmt(r.final['acc'], 2) for r in rs), " / ".join(fmt(r.final['ms_R_hat'], 3) for r in rs),
                             " / ".join(fmt(r.final.get('ms_tau', np.nan), 2) for r in rs)))
    for N, ls in [(20, "-"), (12, "--")]:
        e = [ex.get((N, h)) for h in H_SCAN]
        if all(e):
            axes[1].plot(range(len(H_SCAN)), [d["ms2"] for d in e], color=INK, ls=ls, lw=1, label=f"exact, N={N}")
    e20 = [ex.get((20, h)) for h in H_SCAN]
    if all(e20):
        axes[7].plot(range(len(H_SCAN)), [np.log(10) / (4 * SR_REF["lr"] * d["gap_sym"]) for d in e20], color=INK, ls="--",
                     lw=1, label="ln(10)/(4ηΔ), N=20 (SR, η=%g)" % SR_REF["lr"])
    titles = ["final |E-E0|/|E0|", r"$\langle m_s^2\rangle$ (order parameter)", "Metropolis acceptance rate",
              r"$\hat R$ of the local energy (between/within chains)", r"$\hat R$ of $m_s$ (chains in different Néel sectors?)",
              r"autocorrelation time $\tau$ of $E_{loc}$ (samples)", r"autocorrelation time $\tau$ of $m_s$ (samples)",
              "iterations to reach 2 %"]
    for ax, t in zip(axes, titles):
        ax.set_title(t)
        ax.set_xticks(range(len(H_SCAN)))
        ax.set_xticklabels([f"{h:g}" for h in H_SCAN])
        ax.set_xlabel("transverse field h")
        ax.axvline(xs[1.0], color=MUTED, lw=0.8, ls=":")
    axes[0].set_yscale("log")
    axes[5].set_yscale("log")
    axes[6].set_yscale("log")
    axes[7].set_yscale("log")
    axes[0].legend(fontsize=7)
    axes[1].legend(fontsize=7)
    axes[7].legend(fontsize=7)
    fig.suptitle("Across the transition (dotted: h = J = 1). Final states sampled with 16k samples, 128 chains; every point one seed")
    fig.tight_layout()
    save(fig, "S5_phase_diagnostics")
    table("S5 – per seed: final rel. error [%], acceptance, R̂(m_s), τ(m_s)",
          ["method", "h", "rel. error [%]", "acceptance", "R̂ of m_s", "τ of m_s"], rows_tab)

    # --- correlations and m_s histograms for a few fields (RBM + SR vs exact)
    hs = [0.5, 0.9, 1.0, 1.5, 3.0]
    fig, axes = plt.subplots(2, len(hs), figsize=(20, 7.5))
    for j, h in enumerate(hs):
        e = ex.get((20, h))
        for mname, sel, color, mk in methods[1:]:
            for k, r in enumerate(select(runs, h=h, **sel)):
                if r.collapsed:
                    continue
                rr = np.arange(len(r.final["corr"]))
                axes[0, j].plot(rr, (-1.0) ** rr * np.array(r.final["corr"]), color=color, lw=0.9, marker=mk, ms=3,
                                label=mname if k == 0 else None)
                hist = np.array(r.final["ms_hist"], float)
                centers = np.linspace(-1, 1, 21)
                axes[1, j].plot(centers, hist / hist.sum(), color=color, lw=0.9, drawstyle="steps-mid")
        if e:
            rr = np.arange(len(e["corr"]))
            axes[0, j].plot(rr, (-1.0) ** rr * np.array(e["corr"]), color=INK, ls="--", lw=1.4, label="exact")
            axes[1, j].plot(np.linspace(-1, 1, 21), e["ms_hist"], color=INK, ls="--", lw=1.4, drawstyle="steps-mid")
        axes[0, j].set_title(f"h = {h:g}")
        axes[0, j].set_xlabel("distance r")
        axes[1, j].set_xlabel(r"$m_s$")
    axes[0, 0].set_ylabel(r"$(-1)^r\langle\sigma^z_0\sigma^z_r\rangle$")
    axes[1, 0].set_ylabel("probability")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Staggered correlations and distribution of the staggered magnetization (N=20; every line one seed)")
    fig.tight_layout()
    save(fig, "S5_correlations")


# ====================================================================== S6: sampler settings


def S6():
    from studies import SAMPLERS
    runs = load("S6_sampler")
    if not runs:
        return
    fig, axes = plt.subplots(1, 5, figsize=(22, 4.5))
    rows_tab = []
    for k, (h, color) in enumerate([(0.5, CAT[0]), (1.0, CAT[1])]):
        for i, (nc, nd) in enumerate(SAMPLERS):
            rs = select(runs, h=h, n_chains=nc, n_discard=nd)
            if not rs:
                continue
            x = i + (k - 0.5) * 0.3
            lab = f"h = {h:g}" if i == 0 else None
            strip(axes[0], x, [r.rel_final for r in rs], color, jitter=0.05, median=False, label=lab)
            strip(axes[1], x, [np.nanmean(r.a["R_hat"][-200:]) for r in rs], color, jitter=0.05, median=False)
            strip(axes[2], x, [r.final["ms_R_hat"] for r in rs], color, jitter=0.05, median=False)
            strip(axes[3], x, [np.nanmean(r.a["tau"][-200:]) for r in rs], color, jitter=0.05, median=False)
            strip(axes[4], x, [r.meta["time_per_iter"] * 1e3 for r in rs], color, jitter=0.05, median=False)
            rows_tab.append((f"{h:g}", nc, nd, " / ".join(fmt(100 * r.rel_final) for r in rs),
                             fmt(np.mean([np.nanmean(r.a['R_hat'][-200:]) for r in rs]), 3),
                             fmt(np.mean([r.final['ms_R_hat'] for r in rs]), 3),
                             fmt(np.median([r.meta['time_per_iter'] for r in rs]) * 1e3, 3)))
    titles = ["final |E-E0|/|E0|", r"$\hat R(E_{loc})$ during training (last 200 it.)", r"$\hat R(m_s)$ of the final state",
              r"$\tau(E_{loc})$ during training (samples)", "ms per iteration (one core per run)"]
    for k, (ax, t) in enumerate(zip(axes, titles)):
        ax.set_title(t)
        ax.set_xticks(range(len(SAMPLERS)))
        ax.set_xticklabels([f"{nc} chains\n{1024 // nc} samples/chain\ndiscard {nd}" for nc, nd in SAMPLERS], fontsize=7)
        if k not in (1, 2):
            ax.set_yscale("log")
    axes[0].legend()
    fig.suptitle("Sampler settings at fixed 1024 samples per iteration (RBM + SR, N=20, 3 seeds; every dot one run)")
    fig.tight_layout()
    save(fig, "S6_sampler")
    table("S6 – sampler settings (1024 samples/iteration)",
          ["h", "chains", "discard", "final error per seed [%]", "mean R̂(E_loc)", "mean R̂(m_s)", "ms/iteration"], rows_tab)


# ====================================================================== S8: initial parameter scale


def S8():
    runs = load("S8_init")
    if not runs:
        return
    scales = [1e-4, 1e-3, 1e-2, 1e-1, 0.5]
    cols = dict(zip(scales, ramp(len(scales))))
    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    rows_tab = []
    for i, (state, N, label) in enumerate([("full", 12, "N=12, exact sums, η=0.01"), ("mc", 20, "N=20, Monte Carlo")]):
        for j, sr in enumerate([False, True]):
            ax = axes[i, j]
            for sc in scales:
                rs = [r for r in runs if r.cfg["state"] == state and r.cfg["sr"] == sr and np.isclose(r.cfg["init_scale"], sc)]
                for k, r in enumerate(rs):
                    ax.plot(r.rel(1 if state == "full" else 5), color=cols[sc], lw=1, label=f"init std {sc:g}" if k == 0 else None)
                if rs:
                    t_esc = [r.t_reach(0.15, 1 if state == "full" else 5) for r in rs]
                    strip(axes[i, 2], np.log10(sc) + (0.08 if sr else -0.08), t_esc, ARCH_COLOR["rbm"] if sr else SGD_COLOR,
                          jitter=0.03, median=False, label=("SGD + SR" if sr else "SGD") if sc == scales[2] else None)
                    rows_tab.append((label, "SGD + SR" if sr else "SGD", f"{sc:g}", " / ".join(fmt(t, 3) for t in t_esc),
                                     " / ".join(fmt(100 * r.rel_final) for r in rs)))
            rel_axis(ax, 1e-7 if state == "full" else 1e-5, 0.5)
            ax.set_xscale("symlog", linthresh=100)
            ax.set_title(f"{label}: {'SGD + SR' if sr else 'plain SGD'}")
            ax.set_xlabel("iteration")
            ax.set_ylabel("|E-E0|/|E0|")
            ax.legend(fontsize=7, loc="lower left")
        axes[i, 2].set_xlabel("log10(initial std of the RBM parameters)")
        axes[i, 2].set_ylabel("iterations to leave the plateau (error < 15 %)")
        axes[i, 2].set_title(label)
        axes[i, 2].legend(fontsize=8)
    fig.suptitle("RBM: the plain-SGD plateau at E ≈ -hN (constant wave function) lasts ~ ln(1/initial scale)/η; SR removes it")
    fig.tight_layout()
    save(fig, "S8_init_scale")
    table("S8 – iterations to leave the initial plateau (error < 15 %) and final error per seed",
          ["system", "optimizer", "init std", "iterations", "final error [%]"], rows_tab)


# ====================================================================== S9: beyond exact diagonalisation


def S9():
    from studies import SR_REF
    runs = load("S9_large_N")
    if not runs:
        return
    Ns = [20, 40, 80]
    cols = dict(zip(Ns, ramp(3)))
    lr = SR_REF["lr"]

    def final_rel(r):
        # the final 16k-sample evaluation of the first N = 80 runs ran out of GPU memory:
        # fall back on the average of the last 50 training iterations
        v = r.rel_final
        return v if np.isfinite(v) else abs(np.mean(r.E[-50:]) - r.E0) / abs(r.E0)

    archs = [("rbm", "dense RBM (α=1)"), ("rbm_symm", "translation-invariant RBM (α=2)")]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    rows_tab = []
    for i, (arch, alabel) in enumerate(archs):
        for j, h in enumerate([1.0, 0.5]):
            ax = axes[i, j]
            for N in Ns:
                rs = select(runs, N=N, h=h, arch=arch)
                for k, r in enumerate(rs):
                    ax.plot(r.rel(5), color=cols[N], lw=1,
                            label=f"N = {N}  (Δ = {r.ex['gap_sym']:.3f}, {r.meta['n_params']} parameters)" if k == 0 else None)
                    if N == 80 and h == 0.5 and arch == "rbm" and k == 0:
                        ax.axhline(r.ex["gap_sym"] / abs(r.E0), color=INK, ls=":", lw=1, label="Δ/|E0|: first excited state of the sector")
            rel_axis(ax, 1e-6, 1)
            ax.set_xlabel("iteration")
            ax.set_ylabel(f"{alabel}\n|E-E0|/|E0|  (E0: free fermions)")
            ax.set_title(f"h = {h:g}" + ("  (critical: Δ ≈ 4π/N)" if h == 1.0 else "  (ordered: Δ ≈ 4(1-h))"))
            ax.legend(fontsize=7, loc="upper right")
        ax = axes[i, 2]
        for h, mk in [(1.0, "o"), (0.5, "s")]:
            for eps, c in [(1e-2, CAT[0]), (1e-3, CAT[1])]:
                for N in Ns:
                    rs = select(runs, N=N, h=h, arch=arch)
                    for r in rs:
                        t = r.t_reach(eps, 5)
                        x = 1 / ite_rate(lr, r.ex["gap_sym"])
                        ax.scatter(x, t if np.isfinite(t) else 650, marker=mk, s=30, color=c,
                                   facecolor=c if np.isfinite(t) else "none")
                    if rs:
                        rows_tab.append((alabel, N, f"{h:g}", f"{eps * 100:g} %", f"{rs[0].ex['gap_sym']:.3f}",
                                         fmt(1 / ite_rate(lr, rs[0].ex["gap_sym"]), 3),
                                         " / ".join(fmt(r.t_reach(eps, 5), 3) for r in rs),
                                         " / ".join(fmt(100 * final_rel(r)) for r in rs)))
        xx = np.linspace(0, 1 / ite_rate(lr, 8 * np.sin(np.pi / 160)) * 1.05, 50)
        for eps, c in [(1e-2, CAT[0]), (1e-3, CAT[1])]:
            ax.plot(xx, np.log(0.5 / eps) * xx, color=c, ls="--", lw=1, label=f"ln(0.5/ε)/(4ηΔ), ε = {eps * 100:g} %")
        ax.scatter([], [], marker="o", color=MUTED, label="h = 1")
        ax.scatter([], [], marker="s", color=MUTED, label="h = 0.5")
        ax.axhline(600, color=MUTED, lw=0.8, ls=":")
        ax.set_ylim(0, 680)
        ax.set_xlabel("1 / (4ηΔ)  [iterations], Δ from the free-fermion solution")
        ax.set_ylabel("iterations to reach ε  (hollow at 650: never)")
        ax.set_title(f"{alabel}: convergence time vs gap (every point one seed)")
        ax.legend(fontsize=7)
    fig.suptitle(f"Beyond exact diagonalisation, SR (η={lr:g}, shift {SR_REF['diag_shift']:g}), 3 seeds: convergence time grows with N at h = 1 "
                 "(dashed: gap estimate, an upper bound); the dense RBM gets trapped at E0 + Δ at N = 80, h = 0.5")
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    save(fig, "S9_large_N")
    table("S9 – large N (reference: free-fermion solution): iterations to ε per seed vs the gap time scale 1/(4ηΔ)",
          ["Ansatz", "N", "h", "ε", "Δ", "1/(4ηΔ)", "iterations to ε", "final error [%]"], rows_tab)


# ====================================================================== acceptance rate of every Monte Carlo run


def acc_final(r):
    """Acceptance of the final state (16k-sample evaluation); if that evaluation is missing,
    the mean over the last 50 training iterations. nan for diverged runs."""
    v = r.final.get("acc")
    if v is not None and np.isfinite(v):
        return float(v)
    a = r.a["acc"]
    return float(np.mean(a[-50:])) if len(a) >= 100 and np.all(np.isfinite(a[-50:])) else np.nan


def ACC():
    import json
    import studies
    from studies import H_SCAN
    from vmc_lib import EXACT_DIR, result_path
    s5 = load("S5_phase")
    if not s5:
        return
    ex = {(d["N"], d["h"]): d for d in (json.loads(p.read_text()) for p in EXACT_DIR.glob("*.json"))}
    methods = [("FFNN (ReLU out) + SGD", dict(arch="ffnn", sr=False, N=20), SGD_COLOR, "o"),
               ("FFNN (ReLU out) + SR", dict(arch="ffnn", sr=True, N=20), CAT[0], "s"),
               ("RBM + SGD", dict(arch="rbm", sr=False, N=20), CAT[3], "^"),
               ("RBM + SR", dict(arch="rbm", sr=True, N=20), CAT[2], "D")]
    hcol = dict(zip(H_SCAN, ramp(len(H_SCAN))))
    fig, axes = plt.subplots(2, 4, figsize=(22, 10.5))

    # --- row 1: acceptance during training, one line per run, coloured by the field
    for j, (mname, sel, _, _) in enumerate(methods):
        ax = axes[0, j]
        for h in H_SCAN:
            for r in select(s5, h=h, **sel):
                ax.plot(np.arange(len(r.a["acc"])), r.a["acc"], color=hcol[h], lw=0.9)
            e = ex.get((20, h), {})
            if "acc_local" in e:
                ax.plot([1.0, 1.035], [e["acc_local"]] * 2, color=INK, lw=1.2, transform=ax.get_yaxis_transform(), clip_on=False)
        ax.set_xscale("symlog", linthresh=10)
        ax.set_xlim(0, 1000)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("iteration")
        ax.set_title(f"{mname}: acceptance during training (4 seeds per field)")
    axes[0, 0].set_ylabel("Metropolis acceptance rate (single spin flips)")
    handles = [Line2D([], [], color=hcol[h], lw=2, label=f"h = {h:g}") for h in H_SCAN]
    handles.append(Line2D([], [], color=INK, lw=1.2, label="tick at the right edge: exact ground state"))
    field_handles = handles

    # --- (e) final acceptance vs field, every run of the field scan
    ax = axes[1, 0]
    xs = {h: i for i, h in enumerate(H_SCAN)}
    rows_tab = []
    allm = methods + [("RBM + SR, N=12", dict(arch="rbm", sr=True, N=12), CAT[6], "v")]
    for k, (mname, sel, color, mk) in enumerate(allm):
        for h in H_SCAN:
            rs = select(s5, h=h, **sel)
            ys = [acc_final(r) for r in rs]
            ax.scatter(np.full(len(ys), xs[h] + (k - 2) * 0.09), ys, color=color, marker=mk, s=16, zorder=3,
                       label=mname if h == H_SCAN[0] else None)
            e = ex.get((sel["N"], h), {})
            rows_tab.append((mname, f"{h:g}", " / ".join(fmt(y, 3) for y in ys), fmt(e.get("acc_local", np.nan), 3)))
    for N, ls in [(20, "-"), (12, "--")]:
        e = [ex.get((N, h), {}).get("acc_local") for h in H_SCAN]
        if all(v is not None for v in e):
            ax.plot(range(len(H_SCAN)), e, color=INK, ls=ls, lw=1, label=f"exact ground state, N={N}", zorder=2)
    ax.set_xticks(range(len(H_SCAN)))
    ax.set_xticklabels([f"{h:g}" for h in H_SCAN])
    ax.axvline(xs[1.0], color=MUTED, lw=0.8, ls=":")
    ax.set_xlabel("transverse field h")
    ax.set_ylabel("acceptance rate of the final state")
    ax.set_ylim(0, 1.02)
    ax.set_title("Final state vs field (every dot one run)")
    ax.legend(fontsize=7, loc="upper left")

    # --- every Monte Carlo run at h = 1, N = 20, from all studies
    seen, pool = set(), []
    for name in studies.STUDIES:
        for r in load(name):
            c = r.cfg
            pth = result_path(c)
            if c["state"] == "mc" and c["N"] == 20 and c["h"] == 1.0 and pth not in seen:
                seen.add(pth)
                pool.append(r)
    fam = [("ffnn", "FFNN, ReLU output"), ("ffnn_lin", "FFNN, linear output"), ("rbm", "RBM"), ("rbm_symm", "transl.-inv. RBM")]
    e20 = ex.get((20, 1.0), {})
    for ax, xkey in [(axes[1, 1], "ms2"), (axes[1, 2], "rel")]:
        n_shown = 0
        for arch, alabel in fam:
            for sr in (False, True):
                rs = [r for r in pool if r.arch == arch and r.cfg["sr"] == sr and np.isfinite(acc_final(r))
                      and r.final.get("ms2") is not None and np.isfinite(r.rel_final)]
                if not rs:
                    continue
                x = [r.final["ms2"] if xkey == "ms2" else r.rel_final for r in rs]
                y = [acc_final(r) for r in rs]
                n_shown += len(rs)
                c = ARCH_COLOR[arch]
                ax.scatter(x, y, s=14, marker="o", facecolor=c if sr else "none", edgecolor=c, linewidth=0.8, alpha=0.8,
                           label=f"{alabel}, {'SR' if sr else 'no SR'} ({len(rs)})", zorder=3)
        if "acc_local" in e20:
            if xkey == "ms2":
                ax.scatter([e20["ms2"]], [e20["acc_local"]], marker="*", s=220, color=INK, zorder=5, label="exact ground state")
            else:
                ax.axhline(e20["acc_local"], color=INK, lw=1, ls="--", label="exact ground state", zorder=2)
        ax.set_ylim(0, 1.02)
        ax.set_ylabel("acceptance rate of the final state")
        ax.legend(fontsize=7, loc="upper right" if xkey == "ms2" else "upper left")
        if xkey == "ms2":
            ax.set_xlabel(r"order parameter $\langle m_s^2\rangle$ of the final state")
            ax.set_title(f"All {n_shown} Monte Carlo runs at h=1, N=20: acceptance vs order")
        else:
            ax.set_xscale("log")
            ax.set_xlabel("final |E-E0|/|E0|")
            ax.set_title("Same runs: acceptance vs energy error")

    # --- acceptance vs system size
    ax = axes[1, 3]
    s9 = load("S9_large_N")
    for h, mk in [(0.5, "s"), (1.0, "o")]:
        for arch, alabel in [("rbm", "dense RBM"), ("rbm_symm", "transl.-inv. RBM")]:
            c = ARCH_COLOR[arch]
            pts = [(r.cfg["N"] * (0.96 if arch == "rbm" else 1.04), acc_final(r)) for r in select(s9, h=h, arch=arch)]
            pts = [q for q in pts if np.isfinite(q[1])]
            if pts:
                ax.scatter(*zip(*pts), color=c, marker=mk, s=22, zorder=3, label=f"{alabel} + SR, h = {h:g}")
        ee = [(N, ex[(N, h)]["acc_local"]) for N in (12, 14, 20) if "acc_local" in ex.get((N, h), {})]
        if ee:
            ax.plot(*zip(*ee), color=INK, lw=1, marker="x", ms=5, label=f"exact ground state, h = {h:g}" if h == 0.5 else None)
            ax.axhline(ee[-1][1], color=INK, lw=0.7, ls=":")
    ax.set_xscale("log")
    ax.set_xticks([12, 20, 40, 80])
    ax.set_xticklabels(["12", "20", "40", "80"])
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_xlabel("chain length N")
    ax.set_ylabel("acceptance rate of the final state")
    ax.set_title("Acceptance vs system size (3 seeds; dotted: exact value at N=20)")
    ax.legend(fontsize=7)
    fig.suptitle("Metropolis acceptance rate of every Monte Carlo run (single-spin-flip proposals; 1 sample = N proposals)")
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.subplots_adjust(hspace=0.42)
    # field legend in the gap between the two rows, below the x labels of the first row
    y_mid = 0.5 * (axes[0, 0].get_position().y0 + axes[1, 0].get_position().y1)
    fig.legend(handles=field_handles, loc="center", ncol=len(field_handles), bbox_to_anchor=(0.5, y_mid - 0.012), fontsize=9,
               title="colour = transverse field h (first row)", title_fontsize=9)
    save(fig, "A_acceptance")
    table("ACC – acceptance of the final state across the field scan, per seed, and of the exact ground state",
          ["method", "h", "acceptance per seed", "exact ground state"], rows_tab)


# ====================================================================== the phase transition


def PHASE():
    import json
    from studies import H_SCAN, SR_REF
    from vmc_lib import EXACT_DIR, analytic_gap_sym
    runs = [r for r in load("S5_phase") if r.arch == "rbm" and r.cfg["sr"] and not r.collapsed]
    if not runs:
        return
    ex = {(d["N"], d["h"]): d for d in (json.loads(p.read_text()) for p in EXACT_DIR.glob("*.json"))}
    NCOL = {12: CAT[1], 20: CAT[0], 80: CAT[2]}
    hh = np.linspace(0.02, 3.1, 600)

    def binder(hist, N):
        w = np.asarray(hist, float)
        w = w / w.sum()
        m = np.linspace(-1, 1, N + 1)
        return 1 - (w @ m**4) / (3 * (w @ m**2) ** 2)

    def mx_analytic(N, h):
        """Transverse magnetization <sigma^x> = -(1/N) dE0/dh from the free-fermion solution."""
        k = np.pi * (2 * np.arange(N) + 1) / N
        return np.mean((h + np.cos(k)) / np.sqrt(1 + h**2 + 2 * h * np.cos(k)))

    def chi_analytic(N, h):
        """Transverse susceptibility d<sigma^x>/dh = -(1/N) d^2 E0/dh^2."""
        k = np.pi * (2 * np.arange(N) + 1) / N
        return np.mean(np.sin(k) ** 2 / (1 + h**2 + 2 * h * np.cos(k)) ** 1.5)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10.5))
    rows_tab = []

    def vmc_points(ax, N, fn, label=True):
        rs = select(runs, N=N)
        ax.scatter([r.cfg["h"] for r in rs], [fn(r) for r in rs], color=NCOL[N], s=26, edgecolor="#fcfcfb", linewidth=0.6,
                   zorder=4, label=f"VMC, N={N} (RBM + SR, {len(select(rs, h=1.0))} seeds per field)" if label else None)

    def exact_line(ax, N, fn):
        pts = [(h, fn(ex[(N, h)])) for h in H_SCAN if (N, h) in ex]
        ax.plot(*zip(*pts), color=NCOL[N], lw=1.1, zorder=2, label=f"exact diagonalisation, N={N}")

    # (a) order parameter
    ax = axes[0, 0]
    for N in (12, 20):
        exact_line(ax, N, lambda d: d["ms2"])
        vmc_points(ax, N, lambda r: r.final["ms2"])
    hc = np.linspace(0, 1, 400)
    ax.plot(np.r_[hc, 3.1], np.r_[(1 - hc**2) ** 0.25, 0], color=INK, ls="--", lw=1, label=r"$N\to\infty$: $(1-h^2)^{1/4}$")
    ax.set_ylabel(r"$\langle m_s^2\rangle$,  $m_s=\frac{1}{N}\sum_i(-1)^i\sigma^z_i$")
    ax.set_title("Order parameter (staggered magnetization squared)")

    # (b) Binder cumulant
    ax = axes[0, 1]
    for N in (12, 20):
        exact_line(ax, N, lambda d, N=N: binder(d["ms_hist"], N))
        vmc_points(ax, N, lambda r, N=N: binder(r.final["ms_hist"], N), label=False)
    ax.axhline(2 / 3, color=MUTED, lw=0.8, ls=":")
    ax.text(3.05, 2 / 3 - 0.012, "2/3: ordered", ha="right", va="top", fontsize=8, color=INK2)
    ax.axhline(0, color=MUTED, lw=0.8, ls=":")
    ax.text(3.05, 0.012, "0: disordered (Gaussian)", ha="right", va="bottom", fontsize=8, color=INK2)
    ax.set_ylabel(r"$U = 1-\langle m_s^4\rangle/3\langle m_s^2\rangle^2$")
    ax.set_title("Binder cumulant: the curves for different N cross near the transition")

    # (c) transverse magnetization
    ax = axes[0, 2]
    for N in (12, 20):
        ax.plot(hh, [mx_analytic(N, h) for h in hh], color=NCOL[N], lw=1.1, zorder=2, label=f"free fermions, N={N}")
        vmc_points(ax, N, lambda r: (r.final["corr"][1] - r.final["E"] / r.cfg["N"]) / r.cfg["h"], label=False)
    ax.plot(hh, [mx_analytic(4000, h) for h in hh], color=INK, ls="--", lw=1, label=r"free fermions, $N\to\infty$")
    ax.set_ylabel(r"$\langle\sigma^x\rangle = -\frac{1}{N}\,\partial E_0/\partial h$")
    ax.set_title("Transverse magnetization")

    # (d) susceptibility (analytic)
    ax = axes[1, 0]
    for N in (12, 20, 80):
        ax.plot(hh, [chi_analytic(N, h) for h in hh], color=NCOL[N], lw=1.2, label=f"N={N}")
    ax.plot(hh, [chi_analytic(4000, h) for h in hh], color=INK, ls="--", lw=1, label="N=4000")
    ax.set_ylabel(r"$\partial\langle\sigma^x\rangle/\partial h = -\frac{1}{N}\,\partial^2E_0/\partial h^2$")
    ax.set_title("Transverse susceptibility (free-fermion solution): the peak grows like ln N")

    # (e) gap
    ax = axes[1, 1]
    for N in (12, 20, 80):
        ax.plot(hh, [analytic_gap_sym(N, h) for h in hh], color=NCOL[N], lw=1.2, label=f"N={N}")
    ax.plot(hh, 4 * np.abs(1 - hh), color=INK, ls="--", lw=1, label=r"$N\to\infty$: $4|1-h|$")
    ax.set_ylim(0, 4.2)
    ax.set_ylabel(r"gap $\Delta$ in the sector of the ground state")
    ax.set_title(r"Gap $\Delta = 4\sqrt{1+h^2-2h\cos(\pi/N)}$: closes at the transition")

    # (f) convergence time
    ax = axes[1, 2]
    lr = SR_REF["lr"]
    for N in (12, 20):
        ax.plot(hh, [np.log(100) / ite_rate(lr, analytic_gap_sym(N, h)) for h in hh], color=NCOL[N], lw=1.1, ls="-",
                label=f"gap estimate ln(100)/(4ηΔ), N={N}")
        vmc_points(ax, N, lambda r: r.t_reach(EPS2), label=False)
        for h in H_SCAN:
            rs = select(runs, N=N, h=h)
            rows_tab.append((N, f"{h:g}", " / ".join(fmt(r.final["ms2"], 3) for r in rs), fmt(ex[(N, h)]["ms2"], 3),
                             " / ".join(fmt(binder(r.final["ms_hist"], N), 3) for r in rs), fmt(binder(ex[(N, h)]["ms_hist"], N), 3),
                             " / ".join(fmt((r.final["corr"][1] - r.final["E"] / N) / h, 3) for r in rs), fmt(mx_analytic(N, h), 3),
                             " / ".join(fmt(r.t_reach(EPS2), 3) for r in rs)))
    ax.set_ylim(0, 75)
    ax.set_ylabel(f"iterations to reach 0.2 %  (SR, η = {lr:g})")
    ax.set_title("VMC convergence time (dots: every seed) peaks near the transition")

    for ax in axes.ravel():
        ax.axvline(1.0, color=MUTED, lw=0.8, ls=":", zorder=1)
        ax.set_xlim(0, 3.1)
        ax.set_xlabel("transverse field h  (J = 1)")
        ax.legend(fontsize=8)
    fig.suptitle("The quantum phase transition of the transverse-field Ising chain at h = J: antiferromagnet (h < 1) to paramagnet (h > 1)")
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    save(fig, "P_phase_transition")
    table("PHASE – RBM + SR across the transition, per seed, against exact values",
          ["N", "h", "⟨m_s²⟩ VMC", "exact", "Binder U VMC", "exact", "⟨σˣ⟩ VMC", "exact", "iterations to 0.2 %"], rows_tab)


if __name__ == "__main__":
    which = sys.argv[1:] or ["S1", "S3", "S7", "S2", "S4", "S5", "S6", "S8", "S9", "ACC", "PHASE"]
    for name in which:
        globals()[name]()
    from analysis_common import PLOTS
    # tables.md holds every study: only rewrite it on a full run, partial runs go to their own file
    name = "tables.md" if not sys.argv[1:] else "tables_" + "_".join(sys.argv[1:]) + ".md"
    (PLOTS / name).write_text("\n".join(TABLES))
    print("tables written:", name)
