# Stage 0 - Learn VMC on one trapped particle

Use a single particle in a $d$-dimensional isotropic harmonic trap. In the
main project $d=2$, and oscillator units $\hbar=m=\omega=1$ are used.

## Physics to understand
Hamiltonian in oscillator units:

$$H = -\frac{1}{2}\nabla^2 + \frac{1}{2}r^2.$$

In d dimensions, the exact ground-state wave function is Gaussian and the ground-state energy is d/2.

The student should derive:
- normalization of the Gaussian;
- local energy $E_L = (H\Psi)/\Psi$;
- why $E_L$ is constant for an exact eigenstate;
- the variational principle;
- Metropolis detailed balance;
- sampling error and autocorrelation.

## Wave function
Start with

$$\Psi_\alpha(\mathbf r) = \exp(-\alpha r^2).$$

Alpha is the only variational parameter. The exact value is known.

## Simulation procedure
1. Draw an initial position $\mathbf r$ from a broad Gaussian.
2. Propose $\mathbf r'=\mathbf r+\boldsymbol\delta$, with
   $\boldsymbol\delta$ from a symmetric Gaussian of specified width.
3. Accept with $\min\left(1, |\Psi(\mathbf r')|^2 / |\Psi(\mathbf r)|^2\right)$.
4. Discard burn-in.
5. Measure local energy.
6. Block the time series and estimate uncertainty.
7. Optimize alpha by a simple scan first; only then use gradient-based optimization.

## Required results
- $E(\alpha)$ curve from Monte Carlo with analytic curve overlaid.
- Exact optimum recovered within uncertainty.
- Acceptance rate versus proposal step size.
- Autocorrelation time versus proposal step size.
- Local-energy histogram at nonoptimal alpha and at exact alpha.

## Benchmarks
Analytic calculation only. No literature benchmark is necessary.

## Pass criteria
- Relative energy error < 1e-3 is easy here and should be achieved.
- Statistical uncertainty decreases approximately as $1/\sqrt{M_{\mathrm{eff}}}$.
- Student can explain why an exact eigenstate has zero local-energy variance.

## Reading
- Foulkes, Mitas, Needs, Rajagopal, Rev. Mod. Phys. 73, 33 (2001): Monte Carlo, VMC, trial wave functions.
- Any standard Metropolis-Hastings reference for detailed balance and autocorrelation.
