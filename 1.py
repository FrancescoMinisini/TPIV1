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
np.random.seed(0)


def err_rel(x, y):
    """relative error"""
    return np.abs((x-y)/y)

# throughout this exercise we work with basis states in the σz basis
# represented by arrays containing the quantum numbers -1 and +1

# the following function generates random basis states (uniformly sampled)

def random_states(N, size=1):
    return np.random.choice(np.array([-1,1]), size=(size, N))

s = random_states(N, 3)

def logpsi_mf(params, s):
    """Mean-field Ansatz"""
    # in the used ansatz, the parameters are arranged: theta_2i = phi_i_up, theta_2i+1 = phi_i_down
    up  = params[::2]
    down  = params[1::2]
    # remember: the condition turns into a np.array of bools, and then if true: first argument, else: second argument
    phi = np.where(s==1, up, down)
    logpsi = np.sum(np.log(phi), axis=1)
    return logpsi

def random_params_mf(N, stddev=0.1):
    rand = np.random.normal(loc=0, scale=stddev, size=2*N)
    return rand

params = random_params_mf(N)

def sample_mf(params, size=1):
    up = params[0::2]
    down = params[1::2]
    p_up = np.abs(up)**2 / (np.abs(up)**2 + np.abs(down)**2)
    xi = np.random.random((size, len(up)))
    samples = np.where(xi < p_up, 1, -1)
    return samples

def grad_logpsi_mf(params, s):
    Ns, N = s.shape
    #g has to have the dimernsion (Ns, 2*N) to match the shape of params
    g = np.zeros((Ns,) + params.shape)
    up  = params[::2]
    down  = params[1::2]
    #g and s[1] are both arrays of shape (Ns, N) and up and down are arrays of shape (N,)
    g[:,::2] = np.where(s==1, 1/up, 0)    
    g[:,1::2] = np.where(s==-1, 1/down, 0)
    return g

grad_logpsi_mf(params, x)

assert grad_logpsi_mf(params, x).shape == (len(x),)+params.shape



# test it
x = random_states(N, 5)
params = random_params_mf(N)
logpsi_mf(params, x)

# check the shape
assert logpsi_mf(params, x).shape == (len(x),)

