# -*- coding: utf-8 -*-

import numpy as np
import scipy as sp

from typing import Optional

from ...observables._base import HAS_CUPY, NDArray, get_module, supports_gpu

from ...preprocessing.signal import normalize_signal

if HAS_CUPY:   
    from ...observables.cupy_kernels import get_wpli_kernel
    
from ...observables.numba_kernels import _wpli_numba

def ppc_from_diff(phase_diff: NDArray[complex], min_length: int=0) -> float:
    """
    Compute Pairwise Phase Consistency from phase differences following Vinck et al. (2010) NeuroImage.
    
    PPC is an unbiased estimator of squared phase consistency that is not affected
    by sample size. It is suitable for comparing datasets with different lengths.
    
    Parameters
    ----------
    phase_diff : NDArray[complex]
        Complex-valued array of phase differences.
    min_length : int, optional
        Minimum required length of input array (returns NaN if shorter).
    
    Returns
    -------
    float
        PPC value between -1 and 1.
    """

    xp = get_module(phase_diff)
    
    if len(phase_diff) <= min_length:
        return xp.nan
    
    N = len(phase_diff)
    diff_angle = xp.angle(phase_diff)
    pd_cos = xp.cos(diff_angle)
    pd_sin = xp.sin(diff_angle)
    
    cos_sums = xp.cumsum(pd_cos[::-1])[::-1]
    sin_sums = xp.cumsum(pd_sin[::-1])[::-1]

    r = (pd_cos[:-1]*cos_sums[1:] + pd_sin[:-1]*sin_sums[1:]).sum()
    
    return r * 2 / (N * (N - 1))


def compute_ppc(x: NDArray[complex], y: NDArray[complex], is_normed: bool=False) -> float:
    """
    Compute Pairwise Phase Consistency between two signals.
    
    Parameters
    ----------
    x : NDArray[complex]
        First complex-valued signal.
    y : NDArray[complex]
        Second complex-valued signal.
    is_normed : bool, optional
        If True, assumes inputs are already amplitude normalized.
    
    Returns
    -------
    float
        PPC value between -1 and 1.
    """
    xp = get_module(x)

    if is_normed:
        x_norm = x
        y_norm = y
    else:
        x_norm = normalize_signal(x)
        y_norm = normalize_signal(y)

    phase_diff = x_norm * xp.conjugate(y_norm)
    return ppc_from_diff(phase_diff)

@supports_gpu
def compute_cplv(x: NDArray[complex], y: Optional[NDArray[complex]] = None, is_normed: bool = False, zero_diag: bool = False) -> NDArray[complex]:
    """
    Compute complex Phase Locking Value between all channel pairs. See Palva et.al. 2018 NeuroImage
        
    Parameters
    ----------
    x : NDArray[complex]
        Complex-valued array of shape (n_channels, n_samples).
    y : NDArray[complex], optional
        Optional second input array. If None, computes PLV between channels in ``x``.
    is_normed : bool, optional
        If True, assumes inputs are already normalized.
    zero_diag : bool, optional
        If True, sets diagonal elements to zero.
    
    Returns
    -------
    NDArray[complex]
        Matrix of complex PLVs between channels.
    """
    xp = get_module(x)

    n_ch, n_s = x.shape

    if is_normed:
        x = x
        y = x if y is None else y
    else:
        x = normalize_signal(x)
        y = x if y is None else normalize_signal(y)

    avg_diff = xp.inner(x, xp.conj(y)) / n_s
    
    if zero_diag:
        xp.fill_diagonal(avg_diff, 0)

    return avg_diff

@supports_gpu
def compute_iplv(*args, **kwargs) -> NDArray[complex]:
    return compute_cplv(*args, **kwargs).iplv

@supports_gpu
def compute_wpli(x: NDArray[complex], y: Optional[NDArray[complex]] = None, debias: bool = False, use_numba: bool=True, is_normed: bool = False) -> NDArray[float]:
    """
    Computes weighted Phase-Lag Index between all channels of input data.  
    See (Vinck et al. 2011,  NeuroImage) and (Palva et al. 2018, NeuroImage) for details.
    Calls GPU implementation if the data is on a GPU, otherwise uses Numba or Numpy.
    
    Parameters
    ----------
    x : NDArray[complex]
        Complex-valued array of shape (n_channels, n_samples).
    y : NDArray[complex], optional
        Optional second input array. If None, computes wPLI between channels in ``x``.
    debias : bool, optional
        If True, computes debiased wPLI.
    use_numba : bool, optional
        Whether to use Numba-accelerated CPU kernels.
    is_normed : bool, optional
        Whether the input data is already normalized.
    
    Returns
    -------
    NDArray[float]
        Matrix of wPLI values between all channel pairs.
    """
        
    xp  = get_module(x)
    if is_normed:
        x = x
        y = x if y is None else y 
    else:
        x = normalize_signal(x)
        y = x if y is None else normalize_signal(y)
        
    if HAS_CUPY and not(xp is np):            
        values = _wpli_cupy(x, y, debias)        
    elif use_numba:
        values = _wpli_numba(x, y, debias)
    else:
        values = _wpli_numpy(x, y, debias)

    xp.fill_diagonal(values, 0.0)    
        
    return values

def _wpli_cupy(x: NDArray[complex], y: NDArray[complex], debias: bool = False) -> NDArray[float]:
    """
    Computes weighted Phase-Lag Index, regular or debiased, between all channels of input data. 
    Calls custom CUDA kernels because its not possible to express wPLI in terms of dot product.
    
    Parameters
    ----------
    x, y : NDArray[complex]
        Complex-valued arrays of shape (n_channels, n_samples).
    debias : bool, optional
        Whether to compute debiased wPLI.
    
    Returns
    -------
    NDArray[float]
        Matrix of wPLI values between all channel pairs.
    
    """
    xp = get_module(x)

    _kernel = get_wpli_kernel(x.dtype, debias)

    tile_dim = 16
    res_shape = (x.shape[0], x.shape[0])

    y_conj = xp.conj(y.T)

    outsum = xp.dot(x, y_conj).imag
    outsum_envelope = xp.empty_like(outsum)

    block_size = (tile_dim, tile_dim)
    grid_size = ((res_shape[-1] - 1)//tile_dim + 1,
                 (res_shape[0] - 1)//tile_dim + 1)

    if debias:
        outsum_square = xp.empty_like(outsum_envelope)

        args = (x, y_conj, outsum_envelope, *x.shape, 
                *y_conj.shape, *outsum_envelope.shape, outsum_square)

        _kernel(grid_size, block_size, args)

        outsum = outsum**2 - outsum_square
        outsum_envelope = outsum_envelope**2 - outsum_square
    else:
        args = (x, y_conj, outsum_envelope, *x.shape,
                *y_conj.shape, *outsum_envelope.shape)

        _kernel(grid_size, block_size, args)

    eps = xp.finfo(outsum_envelope.dtype).tiny
    res = xp.full_like(outsum, fill_value=xp.nan)
    out_mask = outsum_envelope > eps
    res[out_mask] = outsum[out_mask] / outsum_envelope[out_mask]
    
    return res

def _wpli_numpy(x: NDArray[complex], y: NDArray[complex], debias: bool = False) -> NDArray[float]:
    xp = get_module(x)

    y_conj = xp.conj(y)

    n_ch = len(x)
    values = xp.zeros([n_ch, n_ch])
    for i in range(n_ch):
        for j in xp.arange(n_ch):
            values[i, j] = _wpli_core(x[i]*y_conj[j], debias)

    return values


def _wpli_core(phase_diff: NDArray[complex], debias: bool=False) -> float:
    csi = np.imag(phase_diff)
    outsum = np.nansum(csi, 0)
    outsumW = np.nansum(np.abs(csi), 0)
    
    if np.abs(outsumW) < 1e-10: 
        return 0.0
        
    if debias:
        outssq = np.nansum(csi**2, 0)
        denom = outsumW**2 - outssq
        wpli = (outsum**2 - outssq) / denom if np.abs(denom) > 0 else 0.0
    else:
        wpli = outsum/outsumW

    return wpli 
