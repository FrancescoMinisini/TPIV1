
import numpy as np
from scipy.sparse import diags
from scipy.linalg import eigh_tridiagonal
from scipy.optimize import brentq
import matplotlib.pyplot as plt

R0 = 0.1


def V_int(r, V0):
    return -V0 / np.cosh(r / R0)**2


# h = 0.001: R0 = 0.1 needs a fine grid, with h = 0.01 the V0 found for a given E_B is ~0.1% too small,
# which shifts the continuum (VMC) energies by up to 1e-2 at E_B = 2.5
def build_kinetic(r_max=12.0, h=0.001):
    n = int(round(r_max / h))
    h = r_max / n
    r = (np.arange(n) + 0.5) * h

    # Second derivative matrix
    main = -2 * np.ones(n)
    main[0] = -1    # f'(0) = 0
    main[-1] = -3   # f(r_max) = 0

    D2 = diags(
        [np.ones(n-1), main, np.ones(n-1)],
        [-1, 0, 1],
        format="csr"
    ) / h**2

    # First derivative matrix
    main = np.zeros(n)
    main[0] = -0.5
    main[-1] = -0.5

    D1 = diags(
        [-0.5*np.ones(n-1), main, 0.5*np.ones(n-1)],
        [-1, 0, 1],
        format="csr"
    ) / h


# I am not sure about this
    # Radial kinetic operator
    K = -D2 - diags(1/r) @ D1
    # This is not symmetric under ordinary matrix transposition and we need it to be symmetric to use the eigsh eigensolver, we could do with another solver but this is stable and perfect, the eigval and vec are unchanged
    
    
    # Symmetrize using g(r) = sqrt(r) f(r) 
    S = diags(np.sqrt(r))
    S_inv = diags(1 / np.sqrt(r))

    K = (S @ K @ S_inv).tocsr()

    return r, h, K


def lowest_state(V0, r, h, K, trapped=False):

    V = V_int(r, V0)

    if trapped:
        V = V + r**2 / 4

    # Hamiltonian = kinetic + potential
    H = K + diags(V)

    # Lowest eigenvalue and eigenvector: H is symmetric tridiagonal, so this is exact and fast (eigsh is very slow on a fine grid)
    E, g = eigh_tridiagonal(H.diagonal(), H.diagonal(1), select="i", select_range=(0, 0))

    # Recover physical radial wavefunction
    #by recovering first eigenvector
    f = g[:, 0] / np.sqrt(r)

    # Normalize in 2D
    f /= np.sqrt(2*np.pi*h*np.sum(r*f**2))

    if f[0] < 0:
        f = -f

    return E[0], f

def find_V0(EB):
    def residual(V0):
        E, _ = lowest_state(V0, r, h, K)
        return E + EB

# Define F(V0) = E_min(V0) + EB, where E_min(V0) is the lowest eigenvalue of the free-space Hamiltonian. Since the binding energy is EB = -E_min(V0), we find the value of V0 for which F(V0) = 0 using root finding.
    return brentq(residual, 20, 120)

# Build matrices once
r, h, K = build_kinetic()

# Solve all three cases
results = {}

for EB in [0.3, 1.0, 2.5]:

    # Free-space calculation
    V0 = find_V0(EB)

    # Untrapped calculation
    E_free, f_free = lowest_state(
        V0, r, h, K, trapped=False
    )

    # Trapped calculation
    E_rel, f_trap = lowest_state(
        V0, r, h, K, trapped=True
    )

    E2 = 1 + E_rel

    results[EB] = (V0, E2, f_free, f_trap)

    print(f"EB={EB}: V0={V0:.6f}, E_free={E_free:.6f}, E2_trapped={E2:.6f}")

# Plot wavefunctions: solid = untrapped, dashed = trapped (same color per EB)
for EB, (V0, E2, f_free, f_trap) in results.items():
    line, = plt.plot(r, f_free, label=f"EB={EB} untrapped")
    plt.plot(r, f_trap, "--", color=line.get_color(), label=f"EB={EB} trapped")

plt.xlim(0, 4)
plt.xlabel("r")
plt.ylabel("f0(r)")
plt.legend()
plt.show()
