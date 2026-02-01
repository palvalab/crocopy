# -*- coding: utf-8 -*-
"""Signal processing utilities for neural data analysis."""

import numpy as np
import scipy as sp
from typing import Union, Any, Optional

import mne

from ..observables._base import get_module, HAS_CUPY, NDArray

if HAS_CUPY:
    import cupy as cp

    from ._gpu_filter import _filter_routine_gpu
    
def normalize_signal(x: NDArray[complex], eps: float = 1e-10) -> NDArray[complex]:
    """Normalize complex signals to unit magnitude.
    
    Args:
        x: Complex-valued input array
        eps: Small value to avoid division by zero
        
    Returns:
        NDArray[complex]: Normalized signal with unit magnitude
        
    Raises:
        ValueError: If input array is empty
    """
    if x.size == 0:
        raise ValueError("Input array cannot be empty")
        
    xp = get_module(x)
    x_abs = xp.abs(x)
    
    # Avoid copying if possible
    if xp.all(x_abs > 0):
        return x / (x_abs + eps)
    
    x_norm = x.copy()
    mask = (x_abs > 0.0)
    x_norm[mask] /= (x_abs[mask] + eps)
    return x_norm

def orthogonalize_signals(x: NDArray[complex], y: NDArray[complex], is_normed: bool=False):
    # orthogonalized X | Y 
    # following https://www.nature.com/articles/nn.3101
    x_normed = x if is_normed else normalize_signal(x)

    ls = np.imag(y*x_normed.conj())
    rs = 1j*x_normed
    return ls*rs

def time_shift(y: NDArray[Any], shift: Optional[Union[int, NDArray[int]]] = None) -> NDArray:
    """Time-shift signals by specified amounts.
    
    Args:
        y: Input array of shape [channels, samples]
        shift: Integer shift amount or array of shifts per channel
        
    Returns:
        NDArray: Time-shifted signals
        
    Raises:
        ValueError: If shifts are invalid
    """
    if y.size == 0:
        raise ValueError("Input array cannot be empty")
        
    xp = get_module(y)
    n_ch, n_s = y.shape

    if shift is None:
        # Vectorized random shifts
        shifts = xp.random.randint(0, n_s, n_ch)
        indices = xp.arange(n_s)
        shifted_indices = (indices[None, :] + shifts[:, None]) % n_s
        return y[xp.arange(n_ch)[:, None], shifted_indices]
    else:
        # Single shift for all channels
        return xp.roll(y, shift, axis=-1)

def filter_data(x: NDArray[float], sfreq: float, frequency: float, 
                omega: float, n_jobs: Union[str, int], 
                decimate_rate: int = 1, 
                normalize_wavelet: bool = False) -> NDArray[complex]:
    """Filter data using Morlet wavelets.
    
    Args:
        x: Input data of shape [channels, samples]
        sfreq: Sampling frequency in Hz
        frequency: Center frequency for wavelet
        omega: Wavelet width parameter
        n_jobs: Number of parallel jobs or 'cuda' for GPU
        decimate_rate: Decimation factor
        normalize_wavelet: Whether to normalize the wavelet
        
    Returns:
        NDArray[complex]: Filtered data
        
    Raises:
        ValueError: If parameters are invalid
        RuntimeError: If GPU requested but not available
    """
    if x.size == 0:
        raise ValueError("Input array cannot be empty")
    if sfreq <= 0 or frequency <= 0 or omega <= 0:
        raise ValueError("Frequency parameters must be positive")
    if decimate_rate < 1:
        raise ValueError("Decimate rate must be >= 1")

    if n_jobs == 'cuda':
        if not HAS_CUPY:
            raise RuntimeError('GPU filtering requested but CuPy is not available')
        data_gpu = cp.asarray(x)
        return _filter_routine_gpu(data_gpu, sfreq, omega, frequency, normalize_wavelet=normalize_wavelet)
    else:
        return mne.time_frequency.tfr_array_morlet(x[np.newaxis, ...], sfreq, [frequency], omega, decim=decimate_rate, verbose=False, n_jobs=n_jobs).squeeze()