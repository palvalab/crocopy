import numpy as np
from scipy.stats import rv_continuous

from joblib import Parallel, delayed

from typing import Optional

from ...observables._base import NDArray, get_module

class singleExp_gen(rv_continuous):
    '''Exponential distribution''' 
    def _pdf(self, x: NDArray[float], gamma: float) -> NDArray[float]:         
        return gamma * np.exp(-gamma*x)
    
singleExp = singleExp_gen(name='singleExp', a=0)
    
class biExp_gen(rv_continuous):
    
    ''' Bi-exponential distribution '''
    def _pdf(self, x: NDArray[float], gamma1: float, gamma2: float, delta1: float, delta2: float):
        return (1/(delta1+delta2))*(delta1*gamma1*np.exp(-(gamma1*x)) + delta2*gamma2*np.exp(-(gamma2*x))) 

biExp = biExp_gen(name='biExp', a=0)

def _fit_exponents(data): 
    
    ''' Fit data with MLE'''
    
    gamma1, gamma2, delta1, delta2, _, _ = biExp.fit(data, floc=0, fscale=1)
    gamma, _, _ = singleExp.fit(data, floc=0, fscale=1)
    
    biexp_params = [gamma1, gamma2, delta1, delta2]
    exp_params = gamma
    
    return biexp_params, exp_params

def _exp_pdf(x: NDArray[float], a: NDArray[float], g: NDArray[float], xp=np) -> NDArray[float]:
    return (a/g).reshape(-1,1)*xp.exp(-1.0/g.reshape(-1,1)*x)

def _fit_biexponential_mixture(x: NDArray[float], max_iters: int = 1000, tol: float = 1e-12):
    """
    Fit a 2-component exponential mixture using the EM algorithm.

    The mixture model is:
        f(x) = p * alpha * exp(-alpha * x)
                 + (1 - p) * beta * exp(-beta * x).

    Parameters
    ----------
    x : NDArray[float]
        2D array of signal power, shape (n_channels, n_samples).
        Should be scaled to [0, 1].
    max_iters : int, optional
        Maximum number of EM iterations.
    tol : float, optional
        Convergence tolerance on the log-likelihood.

    Returns
    -------
    NDArray[float]
        Array of shape (2, n_channels) with estimated rates (alpha, beta).
    NDArray[float]
        Mixing weights p for each channel, shape (n_channels,).
    """
    xp = get_module(x) 

    n_chans, n_samp = x.shape

    eps = xp.finfo(x.dtype).tiny if xp.issubdtype(x.dtype, xp.floating) else 1e-12

    p = xp.full(n_chans, 0.8)
    mean = xp.mean(x, axis=-1)
    mean = xp.clip(mean, eps, None)
    alpha = 1.0/mean
    beta = 2.0/mean

    prev_ll = -xp.inf

    for it in range(max_iters):
        # E-step
        # w_1 = p*alpha*exp(-alpha*x)
        # w_2 = (1 - p)*beta*exp(-beta*x)
        # z1 = w1 / (w1 + w2)

        w1 = _exp_pdf(x, p, 1/alpha, xp=xp)
        w2 = _exp_pdf(x, 1-p, 1/beta, xp=xp)
        
        denom = xp.clip(w1 + w2, eps, None)
        z1 = w1 / denom
        z2 = 1 - z1

        # M-step
        # p = sum_z1 / (sum_z1 + sum_z2) or just n_samp
        # alpha = sum(z1) / sum(z1*x)
        # beta = sum(z2) / sum(z2*x)
        sum_z1 = xp.sum(z1, axis=-1)
        sum_z2 = xp.sum(z2, axis=-1)
        
        p = sum_z1 / n_samp
        
        alpha = sum_z1 / xp.sum(z1 * x, axis=-1)
        beta = sum_z2 / xp.sum(z2 * x, axis=-1)

        # check convergence
        ll = xp.sum(xp.log(denom))
        if xp.abs(ll - prev_ll) < tol * xp.abs(ll):
            break

        prev_ll = ll

    return xp.vstack([alpha, beta]), p

def _compute_BiS_channel_mle(data: NDArray[float]) -> float:     
    biexp_params, exp_params = _fit_exponents(data)
    
    gamma1, gamma2, delta1, delta2 = biexp_params
    
    gamma = exp_params
    
    loglike_biexp = np.sum(np.log(biExp.pdf(data, gamma1, gamma2, delta1, delta2)))
    loglike_exp = np.sum(np.log(singleExp.pdf(data, gamma)))

    nSamples = data.shape[0]
    k_biexp = len(biexp_params)  
    k_exp = 1 
    
    BIC_biexp = -2*loglike_biexp + k_biexp*np.log(nSamples)
    BIC_exp = -2*loglike_exp + k_exp*np.log(nSamples)

    deltaBIC = BIC_exp - BIC_biexp
    
    if deltaBIC > 0: 
        BiS = np.log10(deltaBIC)
    else:
        BiS = 0
    
    return BiS

def compute_BiS_mle(data: NDArray[float], context: Optional[Parallel]=None, n_jobs: int=-1) -> NDArray[float]:
    """
    Compute BiS using MLE fits of single- and bi-exponential models.

    Parameters
    ----------
    data : NDArray[float]
        Input power array of shape (n_channels, n_samples).
    context : joblib.Parallel, optional
        Optional joblib parallel context to reuse.
    n_jobs : int, optional
        Number of parallel jobs for MLE fitting.

    Returns
    -------
    NDArray[float]
        BiS scores for each channel.

    Raises
    ------
    RuntimeError
        If called with CuPy data (MLE requires NumPy/SciPy).
    """
    if not(get_module(data) is np):
        raise RuntimeError('Trying to run MLE BiS with CuPy data! Use only EM algorithm with it.')
    
    if (context is None):
        context = Parallel(n_jobs=n_jobs)

    res = context(delayed(_compute_BiS_channel_mle)(chan_data) for chan_data in data)
    res_arr = np.array(res)

    return res_arr

def compute_BiS_em(data: NDArray[float], max_iters: int=1000) -> NDArray[float]:
    """
    Compute Bistability Score (BiS) using Expectation-Maximization algorithm.
    
    Parameters
    ----------
    data : NDArray[float]
        Input data array of shape (n_channels, n_samples)
    max_iters : int, optional
        Maximum number of iterations for the EM algorithm, by default 1000
    
    Returns
    -------
    NDArray[float]
        BiS scores for each channel
        
    Raises
    ------
    ValueError
        If input data is empty, 1D, or contains negative values
    """
    xp = get_module(data)

    if data.ndim != 2:
        raise ValueError("Input data must be a 2D array of shape (n_channels, n_samples)")
        
    if xp.any(data < 0):
        raise ValueError("Input data contains negative values")
    
    n_samples = data.shape[-1]

    exp_params = 1.0/data.mean(axis=-1)
    thetas, alpha = _fit_biexponential_mixture(data, max_iters=max_iters)

    eps = xp.finfo(data.dtype).tiny if xp.issubdtype(data.dtype, xp.floating) else 1e-12
    exp_pdf = _exp_pdf(data, 1.0, 1.0/exp_params, xp=xp)
    biexp_pdf = _exp_pdf(data, alpha, 1.0/thetas[0], xp=xp) + _exp_pdf(data, 1 - alpha, 1.0/thetas[1], xp=xp)

    exp_pdf = xp.clip(exp_pdf, eps, None)
    biexp_pdf = xp.clip(biexp_pdf, eps, None)

    nnlf_exp = xp.sum(xp.log(exp_pdf), axis=-1)
    nnlf_biexp = xp.sum(xp.log(biexp_pdf), axis=-1)
        
    bic_exp = -2*nnlf_exp + xp.log(n_samples)
    bic_biexp = -2*nnlf_biexp + 3*xp.log(n_samples)

    deltaBIC = bic_exp - bic_biexp

    BiS = xp.zeros_like(deltaBIC)

    valid_mask = (deltaBIC > 0) & xp.isfinite(deltaBIC)
    BiS[valid_mask] = xp.log10(deltaBIC[valid_mask])
    
    return BiS

def compute_BiS(data: NDArray[float], method: str='em', is_power: bool=False, is_envelope: bool=False, **kwargs) -> NDArray[float]:
    """
    Compute Bistability Score (BiS) using EM or MLE following Wang et. al. 2023, Journal of Neuroscience.

    Parameters
    ----------
    data : NDArray[float]
        Input data array of shape (n_channels, n_samples).
    method : {'em', 'mle'}, optional
        Estimation method to use.
    is_power : bool, optional
        If True, data is already power.
    is_envelope : bool, optional
        If True, data is an amplitude envelope (power is computed as envelope^2).
    **kwargs
        Additional arguments passed to the selected method.

    Returns
    -------
    NDArray[float]
        BiS scores for each channel.
    """
    xp = get_module(data)

    if is_power:
        data_power = xp.asarray(data)
    elif is_envelope:
        data_power = data**2
    else:
        data_power = xp.abs(data)**2
    
    data_power /= data_power.max(axis=-1, keepdims=True)

    if method == 'mle':
        return compute_BiS_mle(data_power, **kwargs)
    elif method == 'em':
        return compute_BiS_em(data_power, **kwargs)
    else:
        raise ValueError(f'BiS: unknown method {method}! Only "em" and "mle" are available.')
