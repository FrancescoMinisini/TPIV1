# Assignment 1: noninteracting trapped fermions

## Tasks

1. Implement normalized 2D oscillator orbitals and their derivatives.
2. Build separate spin-up and spin-down Slater determinants.
3. Evaluate determinant sign/log magnitude robustly.
4. Compare automatic differentiation with analytic kinetic-energy derivatives.
5. Sample $N=2,4,6$ and verify same-spin antisymmetry numerically.

## Required deliverables

- orbital and determinant unit tests;
- exact shell-energy table and VMC estimates with error bars;
- radial density $n(r)$ and same-spin pair distribution;
- an exchange test showing $\Psi(\ldots,\mathbf r_i,\ldots,\mathbf r_j,\ldots)
  =-\Psi(\ldots,\mathbf r_j,\ldots,\mathbf r_i,\ldots)$;
- a short explanation of why the two spin determinants are separately
  antisymmetric.

## Definition of done

The energy and density agree with the independent shell-filling calculation,
the density integrates to $N$, and the exchange test passes to numerical
precision. Use the matching [solution](../../solutions/stage_01_noninteracting_fermions.md)
after completing the derivation.
