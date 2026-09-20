import numpy as np
import matplotlib.pyplot as plt
from functools import partial

try:
    from tqdm.auto import tqdm  # progressbar
except ImportError:
    tqdm = lambda x: x

# analytical solution for the ground state energy of the TFI in 1D with pbc

def ising1d_energy(L, Gamma):
    # Gamma is in units of the interaction strength J
    def Epsilon(k,h):
        eps = 1 + h**2 + 2 * h * np.cos(k)
        return 2 * np.sqrt(eps)

    i = np.arange(L)
    k = np.pi * ( 2 * i + 1)/L
    energy = Epsilon(k,Gamma).sum()
    return -0.5*energy

def err_rel(x, y):
    """relative error"""
    return np.abs((x-y)/y)

N = 16          # number of spins
Ns = 512        # number of samples
N_discard = 128 # how many samples to discard

def err_rel(x, y):
    """relative error"""
    return np.abs((x-y)/y)

# throughout this exercise we work with basis states in the σz basis
# represented by arrays containing the quantum numbers -1 and +1

# the following function generates random basis states (uniformly sampled)

def random_states(N, size=1):
    return np.random.choice(np.array([-1,1]), size=(size, N))

random_states(N, 3)

def logpsi_mf(params, s:np.ndarray):
    """Mean-field Ansatz"""
    # in the used ansatz, the parameters are arranged: theta_2i = phi_i_up, theta_2i+1 = phi_i_down
    up  = params[::2]
    down  = params[1::2]
    phi = np.where(s==1, up, down)
    logpsi = np.sum(np.log(phi), axis=1)
    return logpsi

def random_params_mf(N, stddev=0.1):

    # TODO return random parameters for the mean-field Ansatz above
    # you can use a zero-mean normal distribution with the standard deviation provided
    #
    return None

# test it
x = random_states(N, 5)
params = random_params_mf(N)
logpsi_mf(params, x)

# check the shape
assert logpsi_mf(params, x).shape == (len(x),)

