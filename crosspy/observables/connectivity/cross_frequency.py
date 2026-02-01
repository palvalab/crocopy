import numpy as np

from typing import Optional, Tuple

from .._base import get_module, NDArray, HAS_CUPY
from .synchrony import compute_wpli, compute_cplv
from ...preprocessing.signal import normalize_signal, filter_data

def compute_phase_amplitude_coupling(data_low_frequency: NDArray[complex], data_high_frequency: NDArray[complex], low_frequency: float, m: int, n: int=1, sfreq: float=1000.0, omega: float=7.5) -> NDArray[complex]:
    """
    Computes n:m cross-frequency phase-amplitude coupling between all channel pairs from two arrays, using the PLV-based method.
    
    Parameters
    ----------
    data_low_frequency : NDArray[complex]
        Low-frequency complex time series, shape (n_channels, n_samples).
    data_high_frequency : NDArray[complex]
        High-frequency complex time series, shape (n_channels, n_samples).
    low_frequency : float
        Low frequency in Hz used for envelope filtering.
    m : int
        Integer ratio parameter (n:m = low:high).
    n : int, optional
        Integer ratio parameter (n:m = low:high).
    sfreq : float, optional
        Sampling frequency in Hz.
    omega : float, optional
        Morlet wavelet width parameter.
    
    Returns
    -------
    NDArray[complex]
        Phase-amplitude coupling values.
    """
    xp = get_module(data_low_frequency)      
    n_jobs = 'cuda' if (HAS_CUPY and not(xp is np)) else 8         

    hf_envelope = xp.abs(data_high_frequency)
    hf_envelope_filtered = filter_data(hf_envelope, sfreq, low_frequency, omega, n_jobs)

    return compute_cplv(data_low_frequency, hf_envelope_filtered, is_normed=False)

def compute_cross_frequency_synchrony(data_low_frequency: NDArray[complex], data_high_frequency: NDArray[complex], m: int, n: int=1) -> NDArray[complex]:
    """
    Computes n:m cross-frequency phase synchrony (phase-phase coupling) between all channel pairs from two arrays.
    
    Parameters
    ----------
    data_low_frequency : NDArray[complex]
        Low-frequency complex time series, shape (n_channels, n_samples).
    data_high_frequency : NDArray[complex]
        High-frequency complex time series, shape (n_channels, n_samples).
    m : int
        Integer ratio parameter (n:m = low:high).
    n : int, optional
        Integer ratio parameter (n:m = low:high).
    
    Returns
    -------
    NDArray[complex]
        Phase synchrony values.
    """
    xp = get_module(data_low_frequency)
    
    phases = xp.angle(data_low_frequency)
    low_frequency_mapped = xp.exp(1j * phases * m / n)
    
    high_frequency_normed = normalize_signal(data_high_frequency)

    return compute_cplv(low_frequency_mapped, high_frequency_normed, is_normed=True)

def compute_amplitude_phase_synchrony(data: NDArray[complex], low_frequency: float, sfreq: float, omega: float=5.0, ps_method: str='plv', **kwargs) -> NDArray[float]:
    """
    Computes 1:1 synchrony between low-frequency-filtered amplitude envelopes of time series between all channel pairs.
    
    Parameters
    ----------
    data : NDArray[complex]
        Complex time series, shape (n_channels, n_samples).
    low_frequency : float
        Low frequency in Hz used for envelope filtering.
    sfreq : float, optional
        Sampling frequency in Hz.
    omega : float, optional
        Morlet wavelet width parameter.
    ps_method : {'plv', 'iplv', 'wpli'}, optional
        Phase synchrony method to compute.
    **kwargs
        Additional arguments passed to the synchrony functions.
    
    Returns
    -------
    NDArray[float]
        Synchrony values.
    """
    xp = get_module(data)
    n_jobs = 'cuda' if (HAS_CUPY and not(xp is np)) else 8        

    data_envelope = xp.abs(data)
    data_filt = filter_data(data_envelope, sfreq, low_frequency, omega, n_jobs)
    
    ps_method = ps_method.lower()
    
    if ps_method == 'plv':
        res = compute_cplv(data_filt, **kwargs)
    elif ps_method == 'iplv':
        res = xp.imag(compute_cplv(data_filt, **kwargs))
    elif ps_method == 'wpli':
        res = compute_wpli(data_filt, **kwargs)
    else:
        raise ValueError(f"Unknown PS_type: {ps_method}. Must be one of: 'plv', 'iplv', 'wpli'")
    
    return xp.abs(res)
