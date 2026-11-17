# Solution 1: noninteracting trapped fermions

This solution is deliberately explicit. Every symbol used by the orbital,
determinant, density, and pair-distribution calculations should have an
implementation-level definition before the Monte Carlo code is written.

The stage introduces two structures that must remain correct in every later
fermionic ansatz:

1. the one-body harmonic-oscillator basis;
2. antisymmetry under exchange of identical fermions.

Because the interaction is absent, the many-body answer is known exactly. This
is a code and convention benchmark, not a fitting exercise.

## 1. Hamiltonian, coordinates, and units

The physical one-particle Hamiltonian is

$$
\hat h
=-\frac{\hbar^2}{2m}\nabla^2
+\frac12m\omega^2 r^2,
\qquad r^2=x^2+y^2.
$$

Use the oscillator length

$$
a_{\mathrm{ho}}=\sqrt{\frac{\hbar}{m\omega}}
$$

to define dimensionless coordinates

$$
\tilde x=\frac{x}{a_{\mathrm{ho}}},
\qquad
\tilde y=\frac{y}{a_{\mathrm{ho}}},
\qquad
\tilde r^2=\tilde x^2+\tilde y^2.
$$

Energies are measured in $\hbar\omega$. After dropping tildes, the
dimensionless one-particle Hamiltonian is

$$
\hat h
=-\frac12\left(\frac{\partial^2}{\partial x^2}
+\frac{\partial^2}{\partial y^2}\right)
+\frac12(x^2+y^2).
$$

For $N=N_\uparrow+N_\downarrow$ particles, the noninteracting Hamiltonian is

$$
\hat H_0=\sum_{i=1}^{N}\hat h_i
=\sum_{i=1}^{N}
\left[-\frac12\nabla_i^2+\frac12r_i^2\right].
$$

Here $\mathbf r_i=(x_i,y_i)$ is the position of particle $i$ and
$\nabla_i^2=\partial_{x_i}^2+\partial_{y_i}^2$. The spin label
$\sigma_i\in\{\uparrow,\downarrow\}$ is not a spatial coordinate.

## 2. One-dimensional oscillator functions

The one-dimensional dimensionless oscillator eigenfunction is

$$
\phi_n(x)
=\frac{1}{\pi^{1/4}\sqrt{2^n n!}}\,
H_n(x)e^{-x^2/2},
\qquad n=0,1,2,\ldots,
$$

where $H_n(x)$ is the physicists' Hermite polynomial. It can be defined by

$$
H_n(x)=(-1)^n e^{x^2}\frac{d^n}{dx^n}e^{-x^2},
$$

or generated recursively from

$$
H_0(x)=1,\qquad H_1(x)=2x,
$$

$$
H_{n+1}(x)=2xH_n(x)-2nH_{n-1}(x).
$$

The first functions are

$$
\phi_0(x)=\pi^{-1/4}e^{-x^2/2},
$$

$$
\phi_1(x)=\sqrt{2}\,x\,\phi_0(x),
$$

$$
\phi_2(x)=\frac{1}{\sqrt2}(2x^2-1)\phi_0(x).
$$

They satisfy

$$
\int_{-\infty}^{\infty}\phi_n(x)\phi_m(x)\,dx=\delta_{nm},
$$

where $\delta_{nm}=1$ for $n=m$ and $0$ otherwise.

Useful derivative identities for implementation are

$$
\frac{d\phi_n}{dx}
=\sqrt{\frac n2}\,\phi_{n-1}(x)
-\sqrt{\frac{n+1}{2}}\,\phi_{n+1}(x),
$$

and, from the one-dimensional eigenvalue equation,

$$
\frac{d^2\phi_n}{dx^2}
=\left[x^2-(2n+1)\right]\phi_n(x).
$$

The second identity is particularly useful for testing automatic
differentiation. It avoids taking numerical finite differences of noisy
wavefunction values.

## 3. Two-dimensional Cartesian orbitals

The Cartesian orbital labelled by the pair of nonnegative integers
$\mathbf n=(n_x,n_y)$ is the product

$$
\varphi_{n_x,n_y}(x,y)
=\phi_{n_x}(x)\phi_{n_y}(y).
$$

It is normalized because each one-dimensional factor is normalized:

$$
\int_{\mathbb R^2}
\varphi_{n_x,n_y}(x,y)\varphi_{m_x,m_y}(x,y)\,dx\,dy
=\delta_{n_xm_x}\delta_{n_ym_y}.
$$

The orbital energy is

$$
\varepsilon_{n_x,n_y}
=\left(n_x+\frac12\right)
+\left(n_y+\frac12\right)
=n_x+n_y+1.
$$

The shell index is

$$
q=n_x+n_y.
$$

All orbitals with the same $q$ have energy $\varepsilon=q+1$. The number of
nonnegative integer pairs $(n_x,n_y)$ satisfying $n_x+n_y=q$ is $q+1$.
Therefore the spatial degeneracy of shell $q$ is $q+1$. Each spatial orbital
can be occupied once by an up fermion and once by a down fermion.

For derivatives,

$$
\partial_x\varphi_{n_x,n_y}
=\phi_{n_x}'(x)\phi_{n_y}(y),
\qquad
\partial_y\varphi_{n_x,n_y}
=\phi_{n_x}(x)\phi_{n_y}'(y),
$$

and

$$
\nabla^2\varphi_{n_x,n_y}
=\left[x^2+y^2-2(n_x+n_y+1)\right]\varphi_{n_x,n_y}.
$$

The last result is equivalent to

$$
\hat h\,\varphi_{n_x,n_y}
=\varepsilon_{n_x,n_y}\varphi_{n_x,n_y}.
$$

## 4. Many-body configuration and determinants

Write a configuration as

$$
X=(X_\uparrow,X_\downarrow),
$$

where

$$
X_\uparrow=(\mathbf r_1^\uparrow,\ldots,\mathbf r_{N_\uparrow}^\uparrow),
\qquad
X_\downarrow=(\mathbf r_1^\downarrow,\ldots,
\mathbf r_{N_\downarrow}^\downarrow).
$$

For a chosen list of $N_\sigma$ occupied orbitals
$a=1,\ldots,N_\sigma$, define

$$
D_\sigma(X_\sigma)
=\frac{1}{\sqrt{N_\sigma!}}
\det_{i,a}\left[
\varphi_a(\mathbf r_i^\sigma)
\right].
$$

The prefactor is the conventional normalization for orthonormal orbitals.
It can be omitted in a fixed-$N$ Monte Carlo calculation because it cancels
from normalized expectation values, but the convention must be consistent if
wavefunction ratios between sectors are later used.

For a balanced system with the same occupied orbital list in each spin sector,

$$
\Psi(X)
=D_\uparrow(X_\uparrow)D_\downarrow(X_\downarrow).
$$

The total wavefunction is antisymmetric under exchange of two up coordinates
and separately under exchange of two down coordinates. In first-quantized
notation, exchanging two coordinates of the same spin exchanges two rows of
the corresponding determinant:

$$
D_\uparrow(\ldots,\mathbf r_i,\ldots,\mathbf r_j,\ldots)
=-D_\uparrow(\ldots,\mathbf r_j,\ldots,\mathbf r_i,\ldots).
$$

The probability density satisfies

$$
|\Psi'|^2=|\Psi|^2
$$

after this sign change. Consequently, a sampler that sees only $|\Psi|^2$ can
appear correct even if the signed or complex amplitude is wrong. The exchange
test must therefore inspect $\Psi$ itself.

## 5. Shell filling and exact many-body energies

For balanced populations $N_\uparrow=N_\downarrow=M$,

$$
E_{\mathrm{exact}}
=2\sum_{a\ \mathrm{occupied}}\varepsilon_a.
$$

The first systems are:

| total $N$ | $N_\uparrow,N_\downarrow$ | occupied orbitals per spin | $E_{\mathrm{exact}}$ |
|---:|---:|---|---:|
| 2 | $(1,1)$ | $(0,0)$ | 2 |
| 4 | $(2,2)$ | $(0,0)$ and one of $(1,0),(0,1)$ | 6 |
| 6 | $(3,3)$ | $(0,0),(1,0),(0,1)$ | 10 |

For $N=2$, each spin contributes $1$, so $E=2$.
For $N=4$, each spin contributes $1+2=3$, so $E=6$.
For $N=6$, each spin contributes $1+2+2=5$, so $E=10$.

$N=6$ is the first balanced closed shell: both orbitals in the $q=1$ shell
are occupied for each spin.

### Important $N=4$ caveat

$N=4$ has an open spatial shell. Choosing $(1,0)$ or $(0,1)$ gives the same
energy but different anisotropic densities. A single Cartesian choice has

$$
n_{N=4}(x,y)
=2|\phi_0(x)\phi_0(y)|^2
+2|\phi_1(x)\phi_0(y)|^2
=\frac{2}{\pi}e^{-(x^2+y^2)}(1+2x^2)
$$

if $(1,0)$ is selected. This density is not radial.

Therefore a rotationally resolved radial-density benchmark should use $N=2$
and $N=6$, or should average equally over the degenerate $(1,0)$ and $(0,1)$
choices. The averaged $N=4$ density is

$$
\overline n_{N=4}(r)
=\frac{2}{\pi}e^{-r^2}(1+r^2).
$$

This distinction prevents an apparent density error from being blamed on the
Monte Carlo code when it is actually caused by comparing an anisotropic state
with a radial analytic curve.

## 6. Exact one-body densities

The spin-resolved one-body density is defined by

$$
n_\sigma(\mathbf r)
=\left\langle
\sum_{i=1}^{N_\sigma}
\delta^{(2)}(\mathbf r-\mathbf r_i^\sigma)
\right\rangle,
$$

where $\delta^{(2)}$ is the two-dimensional Dirac delta function. For a
determinant of occupied orbitals,

$$
n_\sigma(\mathbf r)
=\sum_{a\ \mathrm{occupied}}
|\varphi_a(\mathbf r)|^2.
$$

The total density is

$$
n(\mathbf r)=n_\uparrow(\mathbf r)+n_\downarrow(\mathbf r),
\qquad
\int_{\mathbb R^2}d^2r\,n(\mathbf r)=N.
$$

For the rotationally symmetric $N=2$ and $N=6$ choices,

$$
n_{N=2}(r)=\frac{2}{\pi}e^{-r^2},
$$

and

$$
n_{N=6}(r)=\frac{2}{\pi}e^{-r^2}(1+2r^2).
$$

If a radial histogram is used, define it so that

$$
N=\int_0^\infty 2\pi r\,n(r)\,dr.
$$

The factor $2\pi r$ is the radial Jacobian. A histogram whose bins already
represent probability per unit radius must not have this factor inserted a
second time.

## 7. Pair distributions and the exchange hole

There are several conventions for a pair distribution. Use one explicit
definition. A convenient normalized separation probability for spin labels
$\sigma,\tau$ is

$$
P_{\sigma\tau}(r)
=\frac{1}{\mathcal N_{\sigma\tau}}
\left\langle
\sum_{i\in\sigma}\sum_{j\in\tau}^{\prime}
\delta\!\left(r-|\mathbf r_i-\mathbf r_j|\right)
\right\rangle,
$$

where the prime excludes $i=j$ when $\sigma=\tau$, and

$$
\mathcal N_{\sigma\tau}
=
\begin{cases}
N_\sigma N_\tau,&\sigma\neq\tau,\\
N_\sigma(N_\sigma-1),&\sigma=\tau.
\end{cases}
$$

With this convention,

$$
\int_0^\infty P_{\sigma\tau}(r)\,dr=1.
$$

Some literature instead defines a dimensionless $g_{\sigma\tau}(r)$ relative
to an uncorrelated reference density. These are not numerically identical.
The code and report must state whether the plotted object is $P(r)$,
$g(r)$, or a histogram with a radial Jacobian removed.

For same-spin fermions, the determinant vanishes when
$\mathbf r_i=\mathbf r_j$. Thus

$$
P_{\sigma\sigma}(r)\longrightarrow 0
\quad\text{as }r\longrightarrow0,
$$

which is the exchange hole. It is present without interactions. For opposite
spins there is no same-spin Pauli node, so the noninteracting short-distance
behavior is qualitatively different.

## 8. Numerical implementation

### Orbital evaluation

Implement $H_n$, $\phi_n$, $\phi_n'$, and $\phi_n''$ as independently
testable functions. Evaluate the logarithm of the determinant using a stable
LU or QR factorization. A raw determinant followed by `log(abs(det))` can
underflow or lose sign information.

### Kinetic energy

For a determinant $D_\sigma=\det A_\sigma$, with
$A_{ia}=\varphi_a(\mathbf r_i)$, analytic derivatives can be written using
the inverse matrix:

$$
\partial_{\mu_i}\log D_\sigma
=\sum_a(A_\sigma^{-1})_{ai}
\partial_{\mu_i}A_{ia},
$$

where $\mu_i$ is $x_i$ or $y_i$. The second derivative uses both the second
orbital derivative and products of first derivatives. It is acceptable to use
automatic differentiation first, but compare both approaches at fixed
configurations.

### Metropolis sampling

For a particle-by-particle proposal

$$
\mathbf r_i'=\mathbf r_i+\boldsymbol\eta,
\qquad
\boldsymbol\eta\sim\mathcal N(0,\sigma_{\mathrm{move}}^2I_2),
$$

the proposal is symmetric and

$$
A(X\to X')
=\min\left(1,\frac{|\Psi(X')|^2}{|\Psi(X)|^2}\right).
$$

Use log amplitudes:

$$
\log\frac{|\Psi(X')|^2}{|\Psi(X)|^2}
=2\left[\log|\Psi(X')|-\log|\Psi(X)|\right].
$$

## 9. Required tests and expected results

1. **Orbital normalization:** numerical integration gives
   $\int|\varphi_{n_x,n_y}|^2d^2r=1$.
2. **Orbital orthogonality:** different orbital labels integrate to zero.
3. **Fixed-configuration energy:** analytic and automatic-differentiation
   local energies agree.
4. **Exchange sign:** swapping two same-spin coordinates gives $\Psi'=-\Psi$.
5. **Probability invariance:** the same swap leaves $|\Psi|^2$ unchanged.
6. **Node:** approaching equal same-spin coordinates sends the determinant
   toward zero.
7. **Shell sum:** Monte Carlo energy agrees with the table.
8. **Density normalization:** $\int d^2r\,n(\mathbf r)=N$.
9. **Pair-distribution normalization:** the selected $P_{\sigma\tau}(r)$
   integrates to one.

## 10. Common failures

- Using dimensional coordinates in a dimensionless orbital formula.
- Mixing physicists' and probabilists' Hermite polynomials.
- Omitting the factor $e^{-x^2/2}$ or using the 3D orbital normalization.
- Mixing orbital ordering between the analytic benchmark and determinant code.
- Calling $N=4$ a closed shell or comparing its anisotropic density with a
  radial curve without degeneracy averaging.
- Comparing pair distributions with different normalization conventions.
- Testing only $|\Psi|^2$ and never testing the signed or complex amplitude.

## Acceptance criteria

The orbital functions, derivatives, determinants, energies, densities, and
pair distributions each pass their own definitions and normalization tests.
The many-body energy agrees with shell filling within statistical uncertainty.
The code distinguishes exchange effects from interaction-induced correlations,
and the $N=4$ open-shell degeneracy is explicitly documented.
