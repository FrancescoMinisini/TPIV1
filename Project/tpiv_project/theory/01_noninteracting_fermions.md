# Stage 1 - Noninteracting trapped fermions

Here $N=N_\uparrow+N_\downarrow$ is the total particle number, and
$N_\uparrow$/$N_\downarrow$ are the spin-resolved populations. Use
$\mathbf r=(x,y)$ and oscillator units $\hbar=m=\omega=1$.

## Physics to understand
For noninteracting spin-1/2 fermions in a 2D isotropic harmonic trap, fill single-particle oscillator orbitals subject to Pauli exclusion. Closed shells are especially simple and connect to the few-fermion experiments.

The student should derive:
- 2D harmonic-oscillator spectrum;
- shell degeneracies;
- exact ground-state energy for chosen $N_\uparrow$, $N_\downarrow$;
- construction and antisymmetry of a Slater determinant.

## Wave function

$$\Psi(X) = D_\uparrow(X_\uparrow)D_\downarrow(X_\downarrow),$$

where each determinant is built from exact 2D harmonic-oscillator orbitals.

Suggested first systems:
- $N = 2$: $1$ up + $1$ down;
- $N = 4$: $2 + 2$;
- $N = 6$: $3 + 3$, a closed shell in 2D.

## Computation
1. Implement orbital values and derivatives.
2. Implement log determinant and determinant sign/phase robustly.
3. Use automatic differentiation for kinetic energy initially; compare against analytic orbital derivatives for one test case.
4. Sample $|\Psi|^2$ with particle-by-particle Metropolis moves.

## Required results
- Ground-state energy for each $N$ versus exact shell-filling result.
- Radial density $n(r)$, with its normalization convention stated.
- Same-spin pair distribution showing the exchange hole, with its estimator
  and normalization convention stated.
- Numerical check: swapping two same-spin coordinates changes the wave-function sign but not $|\Psi|^2$.

## Benchmarks
- Exact harmonic-oscillator energies and densities.
- Experimental context: Bayha et al. prepared closed-shell 2, 6, and 12-particle systems in a 2D harmonic trap.

## Pass criteria
- Energy agrees with exact value within MC error.
- Density integrates to N.
- Same-spin exchange test passes to machine precision.

## Reading
- Bayha et al., Nature 587, 583-587 (2020), DOI: 10.1038/s41586-020-2936-y.
- Foulkes et al., Rev. Mod. Phys. 73, 33 (2001), sections on Slater determinants and VMC.
