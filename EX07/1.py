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

# throughout this exercise we work with basis states in the σz basis
# represented by arrays containing the quantum numbers -1 and +1

# the following function generates random basis states (uniformly sampled)

def random_states(N, size=1):
    return np.random.choice(np.array([-1,1]), size=(size, N))

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
    return np.exp(rand)

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
    #tuple concat syntax: (Ns,)+(N,) = (Ns,N)
    g = np.zeros((Ns,) + params.shape)
    
    up  = params[::2]
    down  = params[1::2]
    #g and s[1] are both arrays of shape (Ns, N) and up and down are arrays of shape (N,)
    g[:,::2] = np.where(s==1, 1/up, 0)    
    g[:,1::2] = np.where(s==-1, 1/down, 0)
    return g

# we need a funtion to do the update move to obtain new configurations
# we will use this as propose_fn below

def single_spin_flip(x):
    assert x.ndim == 1
    N, = x.shape
    x = x.copy()
    x[np.random.randint(N)] *= -1
    return x

def sample_direct_mf(logpsi, params, N, N_samples):
    assert logpsi == logpsi_mf
    up = params[::2]
    down = params[1::2]
    p_up = np.abs(up)**2 / (np.abs(up)**2+np.abs(down)**2)
    x = np.random.rand(N_samples, N)
    samples =  np.where(x < p_up, 1, -1)
    return samples


def sample_step(logpsi, params, x, logpsi_x, propose_fn):
    """
    One sampling step of Monte Carlo Markov Chain.
    logpsi: function giving the log wave-funtion of a batch of sample given a set of parametes
            (params,x) -> log(ψ(x))
    params: parameters of the ansatz
    x: sample on which to do a step (shape (N,) )
    logpsi_x: log-value of the wave-function for x 
    propose_fn: update move 
    """

    if logpsi_x is None:
        # for the sampling we work with a single sample
        # but our ansatz only supports batches
        # so we add a dummy batch dimension here
        logpsi_x = logpsi(params, np.expand_dims(x, 0))[0]

    # propose a new state
    x_proposed = propose_fn(x) # this function samples from T(x -> x')
    logpsi_x_proposed = logpsi(params, np.expand_dims(x_proposed, 0))[0]

    # since T(x -> x') = T(x' -> x) for the definition of single_spin_flip => R = |psi'|^2 / |psi|^2
    R = np.exp(2*(logpsi_x_proposed-logpsi_x))
    accept = R > np.random.rand()

    if accept:
        return x_proposed, logpsi_x_proposed
    else:
        return x, logpsi_x

def sample_mcmc(logpsi, params, N, N_samples, N_discard, x0=None, propose_fn=single_spin_flip):
    """
    Monte Carlo Markov Chains sampling. Given an ansatz, it samples randomly N_samples.
    logpsi: function giving the log wave-funtion of a batch of sample given a set of parametes
            (params,x) -> log(ψ(x))
    params: parameters of the ansatz
    N: number of sites
    N_samples: number of samples to generate
    N_discard: number of initial samples to discard (thermalization)
    x0: initial configuration 
        (if None, a random sample is drawn from the Hilbert space)
    propose_fn: update move 
    """

    # Initialization
    if x0 is None:
        x0 = random_states(N, 1)[0]

    x = x0
    logpsi_x = None

    # Thermalization : we don't keep the samples
    for i in range(N_discard):
        x, logpsi_x = sample_step(logpsi, params, x, logpsi_x, propose_fn)

    # MCMC
    samples = []
    for i in range(N_samples):
        x, logpsi_x = sample_step(logpsi, params, x, logpsi_x, propose_fn)
        samples.append(x)

    return np.vstack(samples)

def ising_hamiltonian(x, Γ=1, J=1):
    """
    TFI hamiltonian in 1D with pbc. Given a configuration x compute the all connected x' and matrix elements Hxx' s.t. Hxx' != 0
    """

    # x is again a batch of samples:
    n_samples, n_sites = x.shape

    # there are n_sites + 1 connected states
    n_conn = n_sites + 1

    # intitalize arrays
    x_prime = np.zeros((n_samples, n_conn, n_sites), dtype=x.dtype)
    mels = np.zeros((n_samples, n_conn))

    # states

    # diagonal: we take the first row to be the diagonal where x==x'
    # (this is arbitrary, we could have picked any other order)
    x_prime[:, 0] = x

    # off-diagonal:
    # the remaining rows are the off-diagonal terms

    # compute the all the off-diagonal connected states
    for i in range(n_sites):
       x_prime[:, i+1] = x
       x_prime[:, i+1, i] *= -1

    # you can try to do it without a loop and set all of them directly:
    # x_prime[:, 1:] = ?


    # # connected states
    # x_prime[:, 0] = x

    # x_prime[:, 1:] = x[:, None, :] * (1 - 2*np.eye(n_sites))

    # matrix elements

    # diagonal
    mels[:, 0] = J * np.sum(x * np.roll(x, -1, axis=1), axis=1 )

    # off-diagonal: REMOVE THE +1 FROM N+1
    mels[:, 1:] = -Γ

    return x_prime, mels

def compute_eloc(operator, logpsi, params, x):
    Ns, N = x.shape
    xp, mels = operator(x)
    logpsi_x = logpsi(params, x)
    # logpsi can only take batches of samples (with one single batch dimension)
    # so we have to flatten the input and unflatten the output
    logpsi_xp = logpsi(params, xp.reshape(-1, N)).reshape(xp.shape[:-1])

    eloc = np.sum(mels * np.exp(logpsi_xp - logpsi_x[:, None]), axis=1)
    return eloc

def expect(operator, logpsi, params, x):
    "compute the expectation value of an operator given a batch of samples"
    eloc = compute_eloc(operator, logpsi, params, x)
    E = eloc.mean()
    return E

def expect_and_grad(operator, logpsi, grad_logpsi, params, x):

    eloc = compute_eloc(operator, logpsi, params, x)
    E = eloc.mean()
    Dk = grad_logpsi(params, x)
    grad = 2 * np.real(np.mean( np.conj(Dk) * (eloc - E)[:, None], axis=0 ))
    return E, grad

# we will pass this as opt_fn to the vmc function
def sgd(params, grad, η):
    # one step of gradient descent: theta -> theta - eta * dE/dtheta
    # (returns a new array, the caller's params are left untouched)
    return params - η * grad

def vmc(operator, sample_fn, opt_fn, logpsi, grad_logpsi, params, N, nsteps):

    energies = []

    for i in tqdm(range(nsteps)):

        # sample
        x = sample_fn(logpsi, params, N)
        # estimate energy and gradients
        E, grad = expect_and_grad(operator, logpsi, grad_logpsi, params, x)
        energies.append(E)
        # compute updated parameters
        params = opt_fn(params, grad)

    E = expect(operator, logpsi, params, x)
    energies.append(E)

    return params, energies

# 7.1 j) auto-correlation

def corr_fn_fft(g):
    n = len(g)
    # fluctuations around the mean
    dg = g - np.mean(g)
    # zero-pad to 2n, otherwise the fft gives the circular correlation (the chain wraps around)
    f = np.fft.fft(dg, n=2*n)
    # Wiener-Khinchin: the autocorrelation is the inverse fft of the power spectrum |f|^2
    # c[j] = sum_i dg_i dg_(i+j), a sum over n-j pairs
    c = np.fft.ifft(f * np.conj(f))[:n].real
    c = c / (n - np.arange(n))
    # normalise such that rho[0] = 1
    return c / c[0]

# 7.2) Jastrow

def neighbour_corr(s, d):
    """sum_i s_i s_(i+d) for each sample in s, with pbc"""
    # np.roll(s, -d, axis=1)[:, i] = s[:, (i+d) % N] 
    # is for PERIODIC BOUNDARY CONDITIONSSSS
    return np.sum(s * np.roll(s, -d, axis=1), axis=1)

def logpsi_jastrow_nearest(params, s):
    Ns, N = s.shape
    J1 = params
    return J1 * neighbour_corr(s, 1)

def grad_logpsi_jastrow_nearest(params, s):
    Ns, N = s.shape
    # logpsi is linear in J1, so the derivative is just the sum it multiplies
    return neighbour_corr(s, 1).reshape(Ns, 1)

def random_params_jastrow_nearest(N, stddev=0.1):
    return np.random.normal(0, stddev, size=1)

def logpsi_jastrow_next_nearest(params, s):
    Ns, N = s.shape
    J1 = params[0]
    J2 = params[1]
    return J1 * neighbour_corr(s, 1) + J2 * neighbour_corr(s, 2)

def grad_logpsi_jastrow_next_nearest(params, s):
    Ns, N = s.shape
    # column k is d logpsi / d J_(k+1)
    return np.stack([neighbour_corr(s, 1), neighbour_corr(s, 2)], axis=1)

def random_params_jastrow_next_nearest(N, stddev=0.1):
    return np.random.normal(0, stddev, size=2)


if __name__ == "__main__":
    # the checks for every part of the exercise live in ex07_tests.py,
    # the diagnostic figures in ex07_plots.py
    #   python 1.py               run all quick checks (TODO = not written yet)
    #   python 1.py 1d 1e         run only some of them
    #   python 1.py --list        list the checks, the figures and the VMC runs
    #   python 1.py plots         build every figure (a few minutes)
    #   python 1.py plots --save  ... and write them to plots/ instead of showing them
    #   python 1.py vmc-mf-mcmc   full VMC optimisation with a plot (slow)
    from EX07.ex07_tests import main
    main(globals())

# Its expected that the meanfield ansatz, the lowest energy is not the gs because the problem is the ansatz as you can see in the last 2
# Hamiltonian is Z2 invariant (invert the spins and the Egs is the same energy) thn the prob of param up = down for each and we dont really care wich one, its basically learning noise: if change seed other randomo parameters
# If you know a priori (most of the time its not the case) than its important keep track of the sampling metrics: this time there should be 0 correlation?

# The correlations that we have to look is : 
# - within the single markov chain 
# - within different markov chains

# If you have high: too much computation for the information of the syste
# if its too low, its too good 
# If its too high, you introduce a sweep: you do additional steps skips within the chain (1 every n steps you calculate the observable, gradient etc...)
# furhter states gets you lower correlation

# so the correlation in general is basically an efficiency metric

# The sampling is cheap so if with sweep_size = 1 and its too correlated you can just do it more efficiently by increasing sweep_size (you need to do more sample steps for each evaluation of observables + wavefunction) but its cool because its very cheap the samplings wrt to the evaluations


# Either your ansatz or your sampling method its not working
# We have in this case 2 level of approximations: ansatz and I cant do densly sampling

# Important: implementation of the local operator its important because its a non trivial speedup wrt of the dense exponential states
# If you change the seed or Ns and the result changes its NOT GOOD SAMPLING: we want to push the ansatz to the max 

# You can just use a NN for the estimation of logpsi