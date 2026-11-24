# Week 3: interactions and a Jastrow factor

## 0. Finish Stage 1
- Check that your energies agree with the exact values.
- Compare $n(r)$ with the exact result.

## 1. Two-body problem
Opposite spins interact through $V(r)=-V_0\,\mathrm{sech}^2(r/R_0)$, with $R_0=0.1$. The coupling strength is measured by the free-space binding energy $E_B$ (as in Bayha et al.).
- Solve the free-space relative problem, $-\nabla_r^2+V(r)$ for $m=0$. Find $V_0$ for $E_B=0.3,\ 1.0,\ 2.5$.
- At the same $V_0$, add the trap and solve again. This gives the exact $E_2$ and the relative wavefunction $f_0(r)$.

## 2. Interacting VMC for $N$ particles
Add the interaction to your code and multiply the determinant part by $e^J$, with

$$J=\sum_{i\in\uparrow,\,j\in\downarrow}u(r_{ij}),\qquad u(r)=\sum_{k=1}^{8}a_k\,e^{-r^2/2s_k^2},$$

where the widths $s_k$ are fixed and spaced geometrically between $R_0/2$ and $2.5$.
- For $N=4$, the open shell is degenerate. Start with a single determinant, then try the paired state $\sum_{p=x,y}D_\uparrow[\varphi_0,\varphi_p]\,D_\downarrow[\varphi_0,\varphi_p]$, where $D_\sigma[\varphi_a,\varphi_b]$ is the determinant of the spin-$\sigma$ particles in orbitals $\varphi_a,\varphi_b$. Compare the energies.
- With $a_k=0$ and $V_0=0$, you must recover Stage 1 exactly.
- Optimize the $a_k$ by gradient descent. For $N=2$, compare with the exact $E_2$ (to $10^{-3}$), and compare $u$ with $\ln f_0+r^2/4$.
- Then $N=3,4,6$, starting from the $N=2$ parameters.
- Compute the opposite-spin pair distribution $P_{\uparrow\downarrow}(r)$. For $N=2$, compare it with $|f_0(r)|^2$.

## For the meeting
- A plot of $E_N-E_N^{(0)}$ vs $E_B$ for $N=2,3,4,6$.
- A table comparing $E_2$ from VMC and from the exact solution.
- For $N=4$: single determinant vs paired state.
- $P_{\uparrow\downarrow}(r)$ for $N=2$ (with the exact curve) and $N=6$, at the three couplings.