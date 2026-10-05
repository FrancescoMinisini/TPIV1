# Convergence of VMC for the 1D transverse-field Ising chain

What controls convergence, what SR does, and a setup that converges reliably.

System: $H = J\sum_i \sigma^z_i\sigma^z_{i+1} - h\sum_i\sigma^x_i$, periodic chain, $J = 1$ (antiferromagnet), NetKet 3.17.
Main size $N = 20$ (as in the exercise). Reference energies: exact diagonalisation (Lanczos, $N \le 20$),
cross-checked against the free-fermion solution $E_0 = -\tfrac12\sum_k \varepsilon_k$, which agrees to $2\cdot10^{-13}$ for
all 32 systems and is used alone for $N = 40, 80$.

All numbers below are relative errors $|E-E_0|/|E_0|$. Every figure shows every seed. About 1,220 VMC runs in total,
≈ 4 h of wall time on 6 CPU cores + the GPU. All figures are in [`plots/`](plots/); the per-seed numbers behind every statement
are in [`plots/tables.md`](plots/tables.md).

---

## 1. Short answers

**What sets the convergence time.** With SR, VMC is imaginary-time evolution: one SR step with learning rate $\eta$ is an
Euler step of imaginary time $d\tau = 2\eta$ (verified against exact imaginary-time evolution from the same initial state,
[S3a](plots/S3a_sr_vs_imaginary_time.png)). The energy error then decays like $e^{-4\eta\Delta\,t}$, so

$$ t_{\rm conv} \approx \frac{\ln(\epsilon_{\rm start}/\epsilon)}{4\,\eta\,\Delta}\quad\text{iterations,}$$

where $\Delta$ is the gap to the first excited state *in the symmetry sector of the initial state* (translation invariant,
spin-flip even), $\Delta = 2\varepsilon(\pi/N) = 4\sqrt{1+h^2-2h\cos(\pi/N)}$. This was checked quantitatively: as a function of
$h$, $\eta$ and $N$ the measured rates fall on the prediction ([S3b](plots/S3b_rate_vs_gap.png)). With Monte Carlo up to
$N = 80$ the time still grows with $N$ at $h = 1$ and stays flat at $h = 0.5$, but there the formula is only an upper bound
([S9](plots/S9_large_N.png)). Consequences:
- the slowest point is the critical point $h \approx J$, where $\Delta \approx 4\pi/N$: convergence time grows with $N$
  (critical slowing down);
- in the ordered phase the tiny splitting $E_1 - E_0 \sim e^{-N/\xi}$ between the two Néel states does **not** matter
  (it is $10^{-13}$ at $N = 20$, $h = 0.25$, yet convergence is fast): the relevant gap is $\approx 4(1-h)$;
- larger $\eta$ is faster only up to a stability limit ($\eta = 0.1$ with diagonal shift $\le 10^{-3}$ collapses);
- a larger diagonal shift slows convergence down (directions with QGT eigenvalue below the shift are updated like plain SGD).

**What the plateaus are.** A plateau is a saddle point of the energy, and leaving it takes a time $\propto \ln(1/\text{overlap})/\eta$
with the direction that leads down. Three kinds appear:
1. *Plain SGD at $E \approx -hN$* (relative error ≈ 0.22): the nearly constant initial wave function. Its length is
   $\propto \ln(1/\sigma_0)/\eta$, with $\sigma_0$ the initial parameter scale: 176 → 117 → 58 → 7 iterations for
   $\sigma_0 = 10^{-4} \to 10^{-1}$ ([S8](plots/S8_init_scale.png)). SR, which is invariant under rescaling of the
   parameters, removes it (3–7 iterations for every $\sigma_0 \le 0.1$).
2. *FFNN local minima*: the exercise's FFNN (width 5) ends either at ≈ 1.3 % (its accuracy limit) or at ≈ 5–6 % (a wrong,
   less ordered state). Which one is set by the initialization, **for every optimizer**, with or without SR
   ([S1](plots/S1_summary.png), [S7](plots/S7_plateau_escape.png)). These are present without any Monte Carlo noise
   ([S3e](plots/S3e_sgd_noise_free.png)): they are a property of the landscape, not of sampling.
3. *Excited eigenstates*: at $N = 80$ in the ordered phase the dense RBM gets stuck at exactly $E_0 + \Delta$, the first excited
   state of the sector (two domain walls). An excited eigenstate is a stationary point of imaginary-time evolution, so SR
   cannot leave it quickly ([S9](plots/S9_large_N.png)).

**Effect of SR.** For the RBM, SR changes everything. At the same $\eta = 0.01$: final error 0.008 % instead of 0.62 %,
2 % reached after 32 iterations instead of 78, 0.2 % after ≈ 100 iterations instead of ≈ 2,600, and essentially no spread
between seeds (12/12 seeds reach 0.2 % between iterations 97 and 114). SR also tolerates a larger step: at $\eta = 0.03$ it
reaches 0.2 % in ≈ 40 iterations. For the small FFNN, SR only makes the good
runs faster: it does not change which minimum a seed ends in.

**Reliable configuration.** RBM ($\alpha = 1$) + plain SGD + SR, $\eta = 0.03$, diagonal shift 0.01, 1024 samples
(128 chains, 4 discarded sweeps), 400 iterations: **40/40 runs (10 fields $h = 0.25$–3, 4 seeds) converged to better than
0.04 %** (median 0.002 %), 30/30 at $N = 12$, and the order parameter matches the exact one across the transition
([S5](plots/S5_phase_diagnostics.png)). Code in §4. For larger chains ($N \gtrsim 80$) use the translation-invariant RBM
with small initial weights and a smaller $\eta$ (§3.8).

**Network size.** With plain SGD, more parameters do not give better convergence (FFNN: 0–4 of 4 seeds converge at any size,
essentially at random). With SR, wider is better and more reliable up to a saturation: FFNN error ≈ 0.8 % (width 8) →
0.04–0.23 % (16) → 0.01–0.15 % (32) → 0.014–0.022 % (64); the RBM saturates at $\alpha \approx 1$. Depth costs time and does
not help ([S4](plots/S4_size.png)).

**Cost.** Per iteration, small networks cost 10–50 ms on one CPU core; SR adds 40–130 % for small networks and dominates for
large ones (3×64 FFNN: 884 ms). Best accuracy per second: RBM $\alpha = 1$ + SR (≈ 1 s of compute to reach 2 %, final error
$\sim10^{-4}$, [S4 cost](plots/S4_cost_vs_accuracy.png)).

**Your notebook was slow because it ran on the GPU.** 1000 iterations took 9:53 (≈ 590 ms/iteration). The same settings
take ≈ 236 ms/iteration on the GTX 1650 but ≈ 30 ms on one CPU core; with 128 chains and 4 discarded sweeps per iteration
instead of 16 chains and 128, ≈ 20 ms. For these small networks set `JAX_PLATFORMS=cpu` **before** importing jax.

---

## 2. Method and how the study was kept cheap

- **Parallel, single-threaded runs.** One iteration of these small networks is dominated by fixed overheads, so many
  single-threaded processes (one per configuration, each pinned to its own core) are much faster than one multi-threaded
  process. Unpinned JAX processes start ~40 threads each and oversubscribe the CPU (dense linear algebra became 10× slower);
  pinning fixed this. The large-network SR runs (≥ 2,500 parameters) ran on the GPU in parallel, where they are ~3× faster.
- **Cheaper sampling.** 128 chains × 8 samples + 4 discarded sweeps per iteration instead of 16 × 64 + 128 discarded: the
  chains are kept between iterations, so a long re-thermalization is not needed. This halves the cost at the same accuracy (S6).
- **Noise-free runs on a smaller system.** The questions about the optimization itself (timescale, diagonal shift, learning
  rate, plateaus) were answered with exact sums over all $2^{12}$ states ($N = 12$): no sampling noise, ~5–40 ms per iteration,
  and a direct comparison with exact imaginary-time evolution. Monte Carlo cost hardly depends on $N$ for these sizes, so the
  Monte Carlo studies stayed at $N = 20$.
- **Two phases.** The optimizer grid ran first; the settings for the other studies were chosen from it, and 400 iterations
  were used for SR runs in phase 2 (S1 shows SR needs < 100).
- Everything is resumable and cached: a configuration shared by several studies runs once; JAX compilations are cached on disk.

"CN" in your request was read as the chain length $N$. "ESR" in the meeting notes was not clear to me, so nothing was done for it.

---

## 3. Results by study

### 3.1 SR, learning rate, optimizer (S1) — $N = 20$, $h = 1$, 4 seeds, 1000 iterations
Grid: SGD and SGD + SR with $\eta \in \{0.003, 0.01, 0.03, 0.1\}$ × shift $\in \{10^{-3}, 10^{-2}, 10^{-1}, 1\}$, plus Adam
± SR; three Ansätze: the exercise FFNN (width 5, ReLU output), the same with a linear output, and an RBM ($\alpha = 1$).
Figures: [summary](plots/S1_summary.png), trajectories for [FFNN](plots/S1_trajectories_ffnn.png),
[FFNN linear output](plots/S1_trajectories_ffnn_lin.png), [RBM](plots/S1_trajectories_rbm.png),
[Adam vs SGD](plots/S1_adam_vs_sgd.png).

| RBM, median of 4 seeds | final error | iterations to 2 % | iterations to 0.2 % |
|---|---|---|---|
| SGD, $\eta = 0.01$ | 0.62 % | 78 | never |
| SGD + SR, $\eta = 0.01$, shift 0.01 | 0.008 % | 32 | 104 |
| SGD + SR, $\eta = 0.03$, shift 0.01 | 0.007 % | 15 | 39 |
| SGD + SR, $\eta = 0.1$, shift 0.01 | 0.006 % | 12 | 19 |
| SGD + SR, $\eta = 0.1$, shift 0.001 | **collapsed (4/4 seeds)** | – | – |
| SGD + SR, $\eta = 0.03$, shift 1 | 0.11 % | 53 | 492 |
| Adam, $\eta = 0.01$ | 0.04–0.06 % | 32 | ≈ 290 |
| Adam + SR, $\eta = 0.01$ | 0.004–0.008 % | 33 | ≈ 110 |

- Iterations to converge scale as $1/\eta$ (96 → 32 → 15 iterations to 2 % for $\eta$ = 0.003 → 0.01 → 0.03, shift 0.01), as
  expected for imaginary-time evolution.
- The diagonal shift trades stability for speed: 0.01 is fast and stable up to $\eta = 0.1$; $10^{-3}$ is slightly faster but
  collapses at $\eta = 0.1$; shift 1 is close to plain SGD.
- Adam + SR reaches the same accuracy as SGD + SR but slower, and Adam + SR at $\eta = 0.001$ is noisy for the FFNN
  (late-time fluctuations 5× larger). This supports the note from the meeting: SR is meant to be used with plain SGD.
  Adam rescales every parameter separately, which destroys the natural-gradient (imaginary-time) step.
- FFNN width 5 (both output layers): two outcomes for every optimizer, ≈ 1.2–1.3 % and ≈ 5–6 %. Seed 0 lands in the bad
  one for almost every setting. SR reaches the good minimum 2–5× faster but does not change the split.

### 3.2 Why the timescale is what it is: noise-free runs (S3, S8) — $N = 12$, exact sums
- **SR = imaginary time** ([S3a](plots/S3a_sr_vs_imaginary_time.png)): with a small shift ($10^{-4}$) the RBM energy follows
  the exact imaginary-time curve started from the same wave function, down to the Ansatz's accuracy limit ($10^{-5}$–$10^{-7}$).
- **Rate = gap** ([S3b](plots/S3b_rate_vs_gap.png)): fitted between $10^{-2}$ and $10^{-4}$, all decay rates (9 fields, 4 learning
  rates, 5 sizes) lie on or slightly above $-2\ln(1-2\eta\Delta) \approx 4\eta\Delta$. The gap sets the slowest (asymptotic)
  rate, higher excited states make the early decay faster. The gap is minimal at $h = 1$ and decreases as $8\sin(\pi/2N)$
  with size.
- **Learning rate** ([S3c](plots/S3c_learning_rate_collapse.png)): for $\eta \le 0.03$ the curves collapse when plotted
  against $\eta t$. $\eta = 0.1$ with shift $10^{-4}$ is unstable at all fields: an Euler step multiplies each excited
  component by $1 - 2\eta(E_n - E)$, which exceeds 1 in magnitude for high-energy components. So the largest usable
  $\eta$ is set by the energy range the step can reach, while the gap sets the speed.
- **Diagonal shift** ([S3d](plots/S3d_diag_shift.png)): for the RBM the error after 600 iterations grows monotonically with
  the shift ($1.5\cdot10^{-6}$ at $10^{-5}$ → $3\cdot10^{-3}$ at 1, plain SGD $7\cdot10^{-3}$). The width-6 FFNN is limited by
  the network (≈ 0.8 %) for any shift up to 0.1.
- **Plain SGD without noise** ([S3e](plots/S3e_sgd_noise_free.png)): the plateaus are there without Monte Carlo noise. For
  the RBM, the escape happens at the same $\eta t \approx 0.7$ for every $\eta$: a deterministic gradient-flow time.
- **Initial scale** ([S8](plots/S8_init_scale.png)): plain SGD stays on the $E \approx -hN$ plateau for ≈ 57 iterations per
  decade of $1/\sigma_0$ ($N = 12$; 100 → 60 → 12 iterations at $N = 20$), as expected for leaving an unstable stationary
  point exponentially. SR leaves it in 3–7 iterations for any $\sigma_0 \le 0.1$. A large initial scale (0.5) is worse, not
  better: SGD then stays stuck for 400–1,800 iterations and SR needs ≈ 25 iterations.

### 3.3 Plateau statistics with many seeds (S7) — $N = 20$, $h = 1$, 12 seeds
[Figure](plots/S7_plateau_escape.png). SGD runs: 3000 iterations; SR runs: 1000.

| | reach 2 % | median iteration | reach 0.2 % |
|---|---|---|---|
| FFNN (ReLU output), SGD | 6/12 | 374 (198–2,870) | 0/12 |
| FFNN (ReLU output), SGD + SR | 6/12 | 118 (90–203) | 0/12 |
| FFNN (linear output), SGD | 5/12 | 186 | 0/12 |
| FFNN (linear output), SGD + SR | 9/12 | 175 | 0/12 |
| RBM, SGD | 12/12 | 82 | 10/12 (median 2,640) |
| RBM, SGD + SR | 12/12 | 32 (31–33) | 12/12 (97–114) |

The escape times of plain SGD spread over an order of magnitude between seeds; with SR the RBM runs are almost identical.

### 3.4 Number of samples (S2) — 128 to 4096 samples per iteration, 3 seeds
[Figure](plots/S2_samples.png). The number of samples sets the final noise floor, not the timescale. RBM + SR: final error
0.087 % (128 samples) → 0.03 % (256) → 0.015 % (512) → ≈ 0.01 % (1024–4096), always 14–21 iterations to 2 %. The cost per
iteration grows roughly linearly with the number of samples, so 512–1024 is the sweet spot here. Plain SGD and the small
FFNNs do not improve with more samples: they are limited by the landscape, not by noise.

### 3.5 Network size (S4) — FFNN widths 2–64 × depths 1–3 (both output layers), RBM $\alpha$ = 0.25–8, 4 seeds
[Figure](plots/S4_size.png), [cost vs accuracy](plots/S4_cost_vs_accuracy.png); timings from a separate single-process benchmark.

- **Plain SGD:** no systematic improvement with size for the FFNN (between 0 and 4 of 4 seeds converge at every size);
  deep ReLU-output networks often die at $E = -hN$ (output ReLU zero for all samples, zero gradient). The RBM improves
  slowly (0.6 % at $\alpha = 1$, 0.1–0.3 % at $\alpha = 8$).
- **SR:** error decreases steadily with width. With the linear output layer every seed converges from width 8 on (with the
  ReLU output only at depth 1; deep ReLU-output networks still collapse occasionally, 2/4 seeds at 2×8, 1/4 at 3×8 and 3×32):

  | SR, 4 seeds | parameters | final error | ms/iteration |
  |---|---|---|---|
  | FFNN 1×8 | 177 | 0.75–0.9 % | 18 |
  | FFNN 1×16 | 353 | 0.04–0.23 % | 27 |
  | FFNN 1×32 | 705 | 0.01–0.15 % | 34 |
  | FFNN 1×64 | 1,409 | 0.014–0.022 % | 47 |
  | FFNN 3×64 | 9,729 | 0.015–0.039 % | 884 |
  | RBM $\alpha = 1$ | 440 | 0.002–0.013 % | 48 |
  | RBM $\alpha = 2$ | 860 | 0.002–0.008 % | 120 |

- Depth does not help: 2–3 hidden layers reach the same error as one layer with the same width, at up to 20× the cost.
- The linear output removes these remaining failures: a collapse to $E = -hN$ means the output ReLU is zero for every
  sampled configuration, so the gradient and the QGT both vanish and SR cannot help.
- The notes' remark "more parameters doesn't mean better convergence" holds for plain SGD. With SR, more parameters give
  better results until the error saturates (here at a few hundred parameters for the RBM, ~1,000 for the FFNN).

### 3.6 Transition, acceptance, chains (S5) — $h$ = 0.25 … 3, $N = 20$ (and 12), 4 seeds
Figures: [diagnostics](plots/S5_phase_diagnostics.png), [training curves](plots/S5_trajectories_vs_h.png),
[correlations and $m_s$ distributions](plots/S5_correlations.png).

| 40 runs per method ($N = 20$) | below 2 % | below 0.2 % | worst run |
|---|---|---|---|
| FFNN + SGD (notebook setting) | 10 | 4 | 16.9 % |
| FFNN + SR | 15 | 4 | 16.2 % |
| RBM + SGD | 37 | 17 | 16.6 % |
| RBM + SR | **40** | **40** | **0.034 %** |

- **Physics.** RBM + SR reproduces the exact $\langle m_s^2\rangle$ at every field (0.985 / 0.934 / 0.826 / 0.679 / 0.489 /
  … vs exact 0.985 / 0.934 / 0.826 / 0.682 / 0.495), the staggered correlations, and the full $m_s$ distribution, including
  the finite-size difference between $N = 12$ and $N = 20$. The hardest region is $h \approx 0.75$–1 (errors up to $3\cdot10^{-4}$).
- **The small FFNN fails across the whole range**, not only at the critical point. In the ordered phase many of the
  seeds end in a disordered state ($\langle m_s^2\rangle \approx 0.04$, error 10–17 %) or a state with domain walls
  (correlations changing sign); in the paramagnet every seed is too ordered (error 1.5–6 %). Plain SGD with the RBM fails
  deep in the ordered phase (3 of 4 seeds at 17 % for $h = 0.25$).
- **Acceptance rate** rises with the field: 0.8 % ($h = 0.25$), 3.6 % (0.5), 29 % (1), 61 % (1.5), 82 % (3). In the ordered
  phase a single spin flip creates two domain walls, so almost every proposal is rejected.
- **Correlation within a chain vs between chains.** $\hat R$ of the local energy stays below 1.011 everywhere and its
  autocorrelation time is < 1 sample: the energy estimate is fine. For the order parameter it is different: the converged
  RBM is (correctly) $\mathbb{Z}_2$ symmetric, but local Metropolis chains cannot go from one Néel sector to the other. In
  the ordered phase each chain stays in one sector, so $\hat R(m_s) = 1.41 \approx \sqrt2$ (half the chains in each sector)
  and $\tau(m_s)$ equals the chain length. This decays through the transition ($\hat R(m_s)$ = 1.20 at $h = 1$, 1.03 at 1.5).
  Symmetric quantities ($E$, $m_s^2$, correlations) are unaffected; odd ones such as $\langle m_s\rangle$ would be wrong.
  The symmetry-broken states of the FFNN and of SGD show $\hat R(m_s) \approx 1$ only because the wave function itself
  picked one sector.
- **Convergence speed vs field:** RBM + SR needs most iterations near $h \approx 0.9$ (51 to 0.2 %) and fewest at large $h$
  (11 at $h = 3$), following the gap; in the ordered phase the first ≈ 20 iterations, where the order forms, add a constant.

### 3.7 Sampler settings (S6) — 1024 samples per iteration, RBM + SR, $h$ = 0.5 and 1, 3 seeds
[Figure](plots/S6_sampler.png). The accuracy does not depend on how the samples are distributed over chains (16 × 64, 128 × 8,
512 × 2). The cost is set by the sequential sweeps: 16 chains with 128 discarded sweeps (the notebook setting) cost
138–160 ms/iteration, 128 chains with 4 discarded sweeps 76–91 ms. $\hat R$ itself depends on the chain length: with 2 samples
per chain it rises to 1.22 although the energy is equally good, so for a meaningful $\hat R$ keep ≥ 8–16 samples per chain.

### 3.8 Beyond exact diagonalisation (S9) — $N$ = 20, 40, 80, reference from the free-fermion solution
[Figure](plots/S9_large_N.png). SR ($\eta = 0.03$, shift 0.01), 600 iterations, 3 seeds; dense RBM ($\alpha = 1$) vs
translation-invariant RBM (`nk.models.RBMSymm`, $\alpha = 2$). Reference energy and gap from the free-fermion solution.

- **Critical slowing down.** At $h = 1$ the iterations to 0.1 % grow with $N$: dense RBM 46 → 87 → not reached
  ($N$ = 20 → 40 → 80), translation-invariant RBM 35 → 50 → 64. At $h = 0.5$ they stay at ≈ 30–50. At these sizes the
  estimate $\ln(\epsilon_{\rm start}/\epsilon)/(4\eta\Delta)$ is only an upper bound: modes with larger gaps carry most of the
  error down to $\sim10^{-4}$, so the measured growth is slower than $\propto N$.
- **The dense RBM fails at $N = 80$.** At $h = 0.5$ all 3 seeds stop at 2.4 %, i.e. at $E_0 + \Delta$ with $\Delta = 2.003$:
  the first excited state of the sector (two domain walls, low energy variance, acceptance twice that of $N = 40$). At
  $h = 1$ it drifts slowly and ends at 0.2 % (6,560 parameters for 1,024 samples per iteration).
- **The translation-invariant RBM (163 parameters at $N = 80$) converges**: ≈ $10^{-5}$ at $h = 0.5$, ≈ $10^{-4}$ at $h = 1$.
  One seed sits on the same $E_0 + \Delta$ plateau for ≈ 300 iterations and then converges: exactly the
  "plateau, then convergence after some time" behaviour, with a waiting time set by the (random) overlap with the ground state.
- **Stability at $N = 80$.** With NetKet's default initialization, 1 of 3 seeds diverged at the second step for each field
  (it starts at 80–90 % error). Ramping the diagonal shift (0.1 → 0.01) or the learning rate over the first 50 iterations did
  not fix it. Initial weights of std 0.01 fixed that seed, but another seed then diverged at step 4 ($h = 1$) and one stayed
  at 6 % ($h = 0.5$). So $\eta = 0.03$ is at the edge of stability at $N = 80$: the energy range a step has to handle grows
  with $N$, so $\eta$ should be reduced for larger chains. A test with $\eta = 0.01$ was still running when this was written.

---

## 4. Recommended setup

```python
import os
os.environ["JAX_PLATFORMS"] = "cpu"          # before importing jax: small networks are much faster on the CPU
import netket as nk

g = nk.graph.Chain(N, pbc=True)
hi = nk.hilbert.Spin(s=1/2, N=N)
ha = nk.operator.Ising(hilbert=hi, graph=g, h=h)

model = nk.models.RBM(alpha=1, param_dtype=float)        # or an FFNN of width >= 16-32 with a linear output layer
# for large chains (N >~ 80): nk.models.RBMSymm(symmetries=g.translation_group(), alpha=2, param_dtype=float,
#     kernel_init=init, hidden_bias_init=init, visible_bias_init=init)  with init = jax.nn.initializers.normal(0.01),
#     and a smaller learning rate (eta = 0.03 is at the edge of stability at N = 80, see 3.8)
sampler = nk.sampler.MetropolisLocal(hi, n_chains=128)
vs = nk.vqs.MCState(sampler, model, n_samples=1024, n_discard_per_chain=4)

driver = nk.VMC(ha, nk.optimizer.Sgd(learning_rate=0.03), variational_state=vs,
                preconditioner=nk.optimizer.SR(diag_shift=0.01))
driver.run(n_iter=400)
```

- Expect convergence after ≈ $\ln(\epsilon_{\rm start}/\epsilon)/(4\eta\Delta)$ iterations, with $\Delta = 4\sqrt{1+h^2-2h\cos(\pi/N)}$;
  near $h = 1$ this grows like $N$, so scale the number of iterations (or $\eta$, within the stability limit) with $N$.
- Keep the diagonal shift around 0.01: smaller is not faster in practice and becomes unstable at large $\eta$; larger is slower.
- Use plain SGD with SR, not Adam.
- Check convergence without $E_0$: the energy variance should drop by ~100× and the energy should stop moving on the scale
  of its error bar. A low-variance state can still be an excited state (§3.8, $E_0 + \Delta$), so compare a few seeds.
- For observables that are odd under spin flip ($\langle m_s\rangle$) in the ordered phase, local Metropolis chains do not
  mix between sectors: measure $\langle m_s^2\rangle$ or $|m_s|$, or add a global spin-flip move.

---

## 5. Files

| | |
|---|---|
| `vmc_lib.py` | models, exact data (Lanczos, symmetric-sector gap, free-fermion formula), one VMC run with all diagnostics |
| `studies.py` | every experiment grid (S1–S9) |
| `run.py` | parallel runner: `python run.py S1_optimizer --workers 6 --cpus 0,2,4,6,8,10`; skips finished runs |
| `timing.py` | single-process cost per iteration of every network size |
| `analyze.py`, `analysis_common.py` | all figures (`plots/*.png`) and tables (`plots/tables.md`) |
| `results/runs/*.npz` | one file per run (per-iteration energy, variance, $\hat R$, $\tau$, acceptance, gradient norms, final 16k-sample evaluation) |

Reproduce everything: `python run.py S1_optimizer S3_fullsum S7_plateaus S2_samples S4_size S5_phase S6_sampler S8_init S9_large_N`,
then `python timing.py` and `python analyze.py`.
