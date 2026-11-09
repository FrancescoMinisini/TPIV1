# Assignment 0: single-particle VMC

## Tasks

1. Derive the normalized Gaussian trial state and its local energy.
2. Implement symmetric Metropolis sampling of $|\Psi_\alpha|^2$.
3. Scan $\alpha$ before attempting optimization.
4. Measure acceptance, autocorrelation time, effective sample size, and
   block uncertainty.

## Required deliverables

- `code/tests/test_stage0.py`
- one run record using [`templates/run_record.yaml`](../../templates/run_record.yaml)
- plots of $E(\alpha)$, acceptance, autocorrelation, and local-energy histograms
- a short paragraph explaining zero local-energy variance

## Definition of done

The exact optimum, energy, and zero-variance result agree within the stated
uncertainties. The matching [solution](../../solutions/stage_00_single_particle.md)
must be used as a self-check.
