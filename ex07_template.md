---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.13.8
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

<h1><center>Computational Quantum Physics - PHYS 463</center></h1>

<p><center> <b>Lecturer:</b> <i>Prof. G. Carleo</i> </center><p>
    
<p><center> <b>Assistants: </b> <i>alessandro.sinibaldi@epfl.ch, linda.mauron@epfl.ch, lorenzo.fioroni@epfl.ch </i> </center><p>


## Exercise 07 - Variational Monte Carlo (VMC)

```python
import numpy as np
import matplotlib.pyplot as plt
from functools import partial

try:
    from tqdm.auto import tqdm  # progressbar
except ImportError:
    tqdm = lambda x: x
```

```python
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
```

```python
def err_rel(x, y):
    """relative error"""
    return np.abs((x-y)/y)
```

```python
N = 16          # number of spins
Ns = 512        # number of samples
N_discard = 128 # how many samples to discard
```

```python
# throughout this exercise we work with basis states in the σz basis
# represented by arrays containing the quantum numbers -1 and +1

# the following function generates random basis states (uniformly sampled)

def random_states(N, size=1):
    return np.random.choice(np.array([-1,1]), size=(size, N))

random_states(N, 3)
```

## Exercise 7.1 : Mean-field Ansatz


### a) Ansatz

```python
def logpsi_mf(params, s):
    """Mean-field Ansatz"""
    Ns, N = s.shape

    # TODO implement the mean field ansatz given params
    # and a batch of states s
    #
    # return ...

def random_params_mf(N, stddev=0.1):

    # TODO return random parameters for the mean-field Ansatz above
    # you can use a zero-mean normal distribution with the standard deviation provided
    #
    # return ...
```

```python
# test it
x = random_states(N, 5)
params = random_params_mf(N)
logpsi_mf(params, x)
```

```python
# check the shape
assert logpsi_mf(params, x).shape == (len(x),)
```

### b) Gradient

```python
def grad_logpsi_mf(params, s):
    Ns, N = s.shape
    g = np.zeros((Ns,) + params.shape)

    # TODO compute the gradient of the mean-field Ansatz
    # for each sample in s
    # ...

    return g
```

```python
grad_logpsi_mf(params, x)
```

```python
assert grad_logpsi_mf(params, x).shape == (len(x),)+params.shape
```

### c) Direct sampling

```python
def sample_direct_mf(logpsi, params, N, N_samples):
    assert logpsi == logpsi_mf

    # TODO implement direct sampling for the mean-field Ansatz
    # generate N_samples samples from the given params
    #
    # samples = ...

    return samples
```

```python
samples_direct = sample_direct_mf(logpsi_mf, params, N, Ns)
samples_direct
```

```python
assert samples_direct.shape == (Ns, N)
```

### d) Monte Carlo Sampling

```python
# we need a funtion to do the update move to obtain new configurations
# we will use this as propose_fn below

def single_spin_flip(x):
    assert x.ndim == 1
    N, = x.shape
    x = x.copy()

    # TODO flip a randomly selected spin in x
    # ...

    return x
```

```python
s = np.array([-1,1,1,-1])

print(s)
print(single_spin_flip(s))
```

```python
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

    # TODO decide whether to accept or reject the proposed state
    # according to the metropolis probability
    #
    # accept = ...

    if accept:
        return x_proposed, logpsi_x_proposed
    else:
        return x, logpsi_x
```

```python
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
```

```python
samples_mcmc = sample_mcmc(logpsi_mf, params, N, Ns, N_discard)
samples_mcmc
```

```python
assert samples_mcmc.shape == (Ns, N)
```

### e) Connected Elements

```python
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

    # TODO compute the all the off-diagonal connected states
    # for i in range(n_sites):
    #    x_prime[:, i+1] = ...

    # you can try to do it without a loop and set all of them directly:
    # x_prime[:, 1:] = ...


    # matrix elements

    # diagonal

    # TODO set the matrix elements corresponding to the diagonal
    # mels[:, 0] = J * ...


    # off-diagonal
    mels[:, 1:] = -Γ

    return x_prime, mels
```

```python
xp, mels = ising_hamiltonian(x, 1, 1)
```

```python
xp
```

```python
mels
```

```python
assert xp.shape == (len(x), N+1, N)
assert mels.shape == (len(x), N+1)
```

### f) Local operator

```python
def compute_eloc(operator, logpsi, params, x):
    Ns, N = x.shape
    xp, mels = operator(x)
    logpsi_x = logpsi(params, x)
    # logpsi can only take batches of samples (with one single batch dimension)
    # so we have to flatten the input and unflatten the output
    logpsi_xp = logpsi(params, xp.reshape(-1, N)).reshape(xp.shape[:-1])

    # TODO compute the local energy
    # eloc = ...

    return eloc
```

```python
eloc = compute_eloc(ising_hamiltonian, logpsi_mf, params, x)
```

```python
assert eloc.shape == (len(x),)
```

### g+h) Stochastic estimates

```python
def expect(operator, logpsi, params, x):
    "compute the expectation value of an operator given a batch of samples"
    eloc = compute_eloc(operator, logpsi, params, x)
    E = eloc.mean()
    return E
```

```python
expect(ising_hamiltonian, logpsi_mf, params, samples_direct)
```

```python
expect(ising_hamiltonian, logpsi_mf, params, samples_mcmc)
```

### i) Gradient descent

```python
def expect_and_grad(operator, logpsi, grad_logpsi, params, x):

    eloc = compute_eloc(operator, logpsi, params, x)
    E = eloc.mean()
    Dk = grad_logpsi(params, x)

    # TODO compute the gradient estimator
    # grad = ...

    return E, grad
```

```python
_, g = expect_and_grad(ising_hamiltonian, logpsi_mf, grad_logpsi_mf, params, x)
assert g.shape == params.shape
```

```python
# we will pass this as opt_fn to the vmc function below
def sgd(params, grad, η):

    # TODO implement one step of gradient descent
    # return ...
```

```python
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
```

#### Mean-field Ansatz with direct sampling

```python


# setup the direct sampler fixing the number of samples
sa = partial(sample_direct_mf, N_samples=Ns)

# set up the optimizer by fixing the learning rate
opt = partial(sgd, η=0.01)

#set up the operator
ha = partial(ising_hamiltonian, Γ=1, J=1)

lpsi = logpsi_mf
glpsi = grad_logpsi_mf
# set up the initial guess for the parameters
par = random_params_mf(N, 0.1)

_, energies = vmc(ha, sa, opt, lpsi, glpsi, par, N, 1000)
plt.figure()
plt.plot(energies)
E0_analytical = ising1d_energy(N, 1)
plt.plot([0, len(energies)], [E0_analytical,]*2, color='k')
plt.xlabel('step')
plt.ylabel(('Energy'))
print('relative error: {:0.2e}'.format(err_rel(E0_analytical, energies[-1])))
plt.show()
```
#### Mean-field with MCMC sampling


```python

sa = partial(sample_mcmc, N_samples=Ns, N_discard=N_discard)
opt = partial(sgd, η=0.01)
ha = partial(ising_hamiltonian, Γ=1, J=1)

lpsi = logpsi_mf
glpsi = grad_logpsi_mf
par = random_params_mf(N, 0.1)

_, energies = vmc(ha, sa, opt, lpsi, glpsi, par, N, 1000)

plt.figure()
plt.plot(energies)
E0_analytical = ising1d_energy(N, 1)
plt.plot([0,  len(energies)], [E0_analytical,]*2, color='k')
plt.xlabel('step')
plt.ylabel(('Energy'))
print('relative error: {:0.2e}'.format(err_rel(E0_analytical, energies[-1])))
plt.show()
```
### j) Auto-correlation time (bonus)

```python
def corr_fn_fft(g):
    n = len(g)

    # TODO compute the correlation function ρ with fast fourier transforms
    # given a sequence of estimates g
    # return ...
```

```python
par_mf = random_params_mf(N, 0.1)
x = sample_mcmc(logpsi_mf, par_mf, N_samples=Ns, N=N, N_discard=N_discard)
eloc = compute_eloc(ising_hamiltonian, logpsi_mf, par_mf, x)
```

```python
ρ = corr_fn_fft(eloc)
plt.figure()
plt.plot(ρ)
plt.show()
```

```python
jcut = np.argmin(ρ > 0)
τ = 0.5 + (ρ[:jcut]).sum()
τ
```

## Exercise 7.2


### a) Nearest-neighbour Jastrow

```python
def logpsi_jastrow_nearest(params, s):
    Ns, N = s.shape
    J1 = params

    # TODO compute nearest-neighbor jastrow wavefunction
    # return ...

def grad_logpsi_jastrow_nearest(params, s):
    Ns, N = s.shape

    # TODO compute the sample-wise gradient of nearest-neighbour jastrow
    # make sure that the gradient has shape (Ns, num_params)
    # where here we have num_marams = 1
    #
    # return ...

def random_params_jastrow_nearest(N, stddev=0.1):
    return np.random.normal(0, stddev, size=1)
```

```python
# check
params_jastrow = random_params_jastrow_nearest(N, 0.1)
assert logpsi_jastrow_nearest(params_jastrow, x).shape == (len(x),)
assert grad_logpsi_jastrow_nearest(params_jastrow, x).shape == (len(x),len(params_jastrow))
```

```python
sa = partial(sample_mcmc, N_samples=Ns, N_discard=N_discard)
opt = partial(sgd, η=0.001) # needs a fairly small learning rate
ha = partial(ising_hamiltonian, Γ=1, J=1)

lpsi = logpsi_jastrow_nearest
glpsi = grad_logpsi_jastrow_nearest
par = random_params_jastrow_nearest(N, 0.1)

_, energies = vmc(ha, sa, opt, lpsi, glpsi, par, N, 300)
plt.figure()
plt.plot(energies)
E0_analytical = ising1d_energy(N, 1)
plt.plot([0,  len(energies)], [E0_analytical,]*2, color='k')
plt.xlabel('step')
plt.ylabel(('Energy'))
print('relative error: {:0.2e}'.format(err_rel(E0_analytical, energies[-1])))
plt.show()
```

### b) Nearest+next-nearest-neighbour Jastrow

```python
def logpsi_jastrow_next_nearest(params, s):
    Ns, N = s.shape
    J1 = params[0]
    J2 = params[1]

    # TODO nearest and next-nearest neighbour jastrow
    # return ...

def grad_logpsi_jastrow_next_nearest(params, s):
    Ns, N = s.shape

    # TODO compute the sample-wise gradient of nearest+next-nearest neighbour jastrow
    # make sure that the gradient has shape (Ns, num_params)
    #
    # return ...

def random_params_jastrow_next_nearest(N, stddev=0.1):

    # TODO
    #
    # return ...
```

```python
sa = partial(sample_mcmc, N_samples=Ns, N_discard=N_discard)
opt = partial(sgd, η=0.001)
ha = partial(ising_hamiltonian, Γ=1, J=1)

lpsi = logpsi_jastrow_next_nearest
glpsi = grad_logpsi_jastrow_next_nearest
par = random_params_jastrow_next_nearest(N, 0.1)

_, energies = vmc(ha, sa, opt, lpsi, glpsi, par, N, 300)

plt.figure()
plt.plot(energies)
E0_analytical = ising1d_energy(N, 1)
plt.plot([0,  len(energies)], [E0_analytical,]*2, color='k')
plt.xlabel('step')
plt.ylabel(('Energy'))
print('relative error: {:0.2e}'.format(err_rel(E0_analytical, energies[-1])))
plt.show()
```

