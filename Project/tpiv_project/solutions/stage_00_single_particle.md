# Solution 0: single-particle VMC

This stage establishes the basic identity used throughout VMC:

$$
E[\Psi]=
\frac{\int d^dr\,\Psi^*(\mathbf r)H\Psi(\mathbf r)}
{\int d^dr\,|\Psi(\mathbf r)|^2}
=\left\langle E_L(\mathbf r)\right\rangle_{|\Psi|^2},
\qquad
E_L=\frac{H\Psi}{\Psi}.
$$

The Monte Carlo calculation is therefore not sampling the energy directly. It
samples configurations from $|\Psi|^2$ and averages the local energy evaluated
on those configurations.

## 1. Analytic model

In oscillator units,

$$
H=-\frac12\nabla^2+\frac12r^2.
$$

Use the unnormalized trial state

$$
\Psi_\alpha(\mathbf r)=\mathcal N_\alpha e^{-\alpha r^2},
\qquad \alpha>0.
$$

The condition $\alpha>0$ is required for normalizability. The exact ground
state corresponds to $\alpha=1/2$.

Using

$$
\nabla^2 e^{-\alpha r^2}
=\left(-2\alpha d+4\alpha^2r^2\right)e^{-\alpha r^2},
$$

the local energy is

$$
E_L(\mathbf r;\alpha)
=d\alpha+\left(\frac12-2\alpha^2\right)r^2.
$$

The probability density is proportional to $e^{-2\alpha r^2}$, so

$$
\left\langle r^2\right\rangle_{|\Psi_\alpha|^2}
=\frac{d}{4\alpha}.
$$

Consequently,

$$
E(\alpha)
=\frac d2\left(\alpha+\frac{1}{4\alpha}\right).
$$

Minimizing gives

$$
\frac{dE}{d\alpha}
=\frac d2\left(1-\frac{1}{4\alpha^2}\right)=0,
\qquad
\alpha_\star=\frac12,
\qquad
E_0=\frac d2.
$$

For the project's two-dimensional problem, $E_0=1$ for one particle.

## 2. Physical intuition

Small $\alpha$ produces a broad cloud. The kinetic energy is then small, but
the harmonic potential energy is large because particles explore large $r$.
Large $\alpha$ produces a narrow cloud. The potential energy decreases, but
the rapid spatial variation of the wavefunction increases the kinetic energy.
The variational minimum balances these two effects.

At $\alpha=1/2$, the trial state is an exact eigenstate. Therefore
$E_L(\mathbf r)$ is the same number at every configuration, not merely on
average. This is the zero-variance principle. A nonzero measured variance at
the exact parameter indicates an implementation, derivative, or floating
point problem rather than physical uncertainty.

## 3. Numerical implementation

Store $\log|\Psi_\alpha|=-\alpha r^2$ rather than the raw amplitude. For a
symmetric Gaussian proposal

$$
\mathbf r'=\mathbf r+\boldsymbol\eta,
\qquad
\boldsymbol\eta\sim\mathcal N(0,\sigma_{\mathrm{move}}^2 I),
$$

the Metropolis acceptance probability is

$$
A(\mathbf r\to\mathbf r')
=\min\left(1,
\exp\left[2\log|\Psi_\alpha(\mathbf r')|
-2\log|\Psi_\alpha(\mathbf r)|\right]\right).
$$

Compute this in log space to avoid underflow. The proposal is symmetric, so
there is no proposal-density ratio in this first test.

For each retained configuration, measure:

1. $E_L(\mathbf r)$;
2. $r^2$;
3. whether the move was accepted.

Discard an explicit burn-in and estimate the integrated autocorrelation time
from the retained energy series. Use blocking or batch means for the standard
error; treating every correlated sample as independent is incorrect.

## 4. Correct diagnostic sequence

1. Evaluate the analytic local energy on hand-selected positions.
2. Compare the analytic local energy with an automatic-differentiation result.
3. Run at $\alpha=1/2$ and check that the local-energy variance is at
   numerical precision.
4. Scan $\alpha$ over a broad positive interval.
5. Repeat selected values with several proposal widths.
6. Only then attempt gradient-based optimization.

The scan is deliberately first: it makes the variational landscape visible and
prevents an optimizer from hiding a sign or derivative mistake.

## 5. Expected plots and interpretation

- **$E(\alpha)$:** a smooth minimum at $\alpha=1/2$.
- **Acceptance versus proposal width:** acceptance decreases for overly large
  moves, but the highest acceptance is not necessarily the best mixing.
- **Autocorrelation versus proposal width:** choose a width using effective
  sample size, not acceptance alone.
- **Local-energy histogram:** broad away from the optimum and a narrow
  numerical spike at the exact state.

## 6. Common failures

- Using $\Psi$ instead of $|\Psi|^2$ in the Metropolis ratio.
- Forgetting the factor of two in the probability-density exponent.
- Reporting the noisy minimum of a scan as the exact optimum.
- Computing a standard deviation divided by $\sqrt{M}$ for correlated samples
  without an autocorrelation or blocking correction.
- Letting the optimizer change $\alpha$ during the final measurement chain.

## Acceptance criteria

- $E(\alpha)$ agrees with the analytic curve.
- $\alpha_\star=1/2$ and $E_0=d/2$ are recovered within uncertainty.
- The exact-state local-energy variance is consistent with numerical precision.
- Error bars decrease when the effective, not merely raw, sample count grows.
