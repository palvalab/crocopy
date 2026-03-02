#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from typing import Sequence

from ._base import get_module, NDArray, HAS_CUPY
from .phase import compute_instantaneous_frequency
from ..utils.stats import moving_average_fast

if HAS_CUPY:
    from .cupy_kernels import _pac_kernel

def transform_to_cdf(data: NDArray[float], derivative_threshold: float=1e-3, min_consequent: int=3) -> NDArray[float]:
    """
    Transform pAC values into a cumulative density function (CDF).

    Parameters
    ----------
    data : NDArray[float]
        1D vector of pAC values.
    derivative_threshold : float, optional
        Threshold for the first derivative considered as a plateau.
    min_consequent : int, optional
        Minimum number of consecutive samples below the threshold.

    Returns
    -------
    NDArray[float]
        1D vector representing the CDF.
    """
    xp = get_module(data)

    diff = xp.abs(xp.diff(data))
    
    thresholded = xp.convolve((diff <= derivative_threshold).astype(int), xp.ones(min_consequent, dtype=int))
    baseline_idx = xp.argmax(thresholded)
    
    cdf = data.copy()
    cdf[baseline_idx:] = 0
    cdf /= cdf.sum()
    
    return xp.cumsum(cdf)

def get_length_by_cdf(data: NDArray[float], lag_values: NDArray[float], cdf_threshold: float=0.9, derivative_threshold: float=1e-3, interpolate: bool=False, **kwargs) -> float:
    """
    Compute pAC length as the first lag with CDF >= threshold.

    Parameters
    ----------
    data : NDArray[float]
        1D vector of pAC values.
    lag_values : NDArray[float]
        Sequence of lags.
    cdf_threshold : float, optional
        Threshold to compute length.
    derivative_threshold : float, optional
        Threshold used to compute the CDF.
    interpolate : bool, optional
        If True, interpolate CDF to a finer grid before thresholding.

    Returns
    -------
    float
        pAC length in the same units as ``lag_values``.
    """

    xp = get_module(data)

    cdf = transform_to_cdf(data, derivative_threshold=derivative_threshold)
    
    if interpolate:
        lags_interp = xp.linspace(0, lag_values[~0], 1000)
        cdf_interp = xp.interp(lags_interp, lag_values, cdf)
        
        length_idx = xp.argmax(cdf_interp >= cdf_threshold)

        return lags_interp[length_idx]
    
    length_idx = xp.argmax(cdf >= cdf_threshold)

    return lag_values[length_idx]


def get_length_by_mean(data: NDArray[float], lower_index: int=0, upper_index: int=50, **kwargs) -> float:
    """
    Compute mean pAC value over a fixed index range.

    Parameters
    ----------
    data : NDArray[float]
        pAC values array.
    lower_index : int, optional
        Lower index (inclusive).
    upper_index : int, optional
        Upper index (exclusive).

    Returns
    -------
    float
        Mean pAC value over the specified range.
    """

    return data[..., lower_index:upper_index].mean(axis=-1)


def compute_phase_autocorrelation(data: NDArray[complex], sfreq: float, lags_cycles: Sequence[float]=None, 
                                  method: str='lifetime',  is_normed=False, return_lifetime: bool=True, **kwargs) -> NDArray[float]:
    """
    Compute phase autocorrelation (pACF) or its lifetime metric.

    Parameters
    ----------
    data : NDArray[complex]
        Complex time series of shape (n_channels, n_samples).
    sfreq : float
        Sampling frequency in Hz.
    lags_cycles : Sequence[float], optional
        Lags in cycles to evaluate. Defaults to 0..20 in steps of 0.1.
    method : {'lifetime', 'mean'}, optional
        Aggregation method for lifetime computation.
    is_normed : bool, optional
        If True, assumes data are already normalized to unit magnitude.
    return_lifetime : bool, optional
        If True, return lifetime values; otherwise return pAC values for all lags.
    **kwargs
        Additional arguments passed to lifetime aggregation functions.

    Returns
    -------
    NDArray[float]
        Lifetime values per channel or pAC values per channel and lag.
    """
    xp = get_module(data)

    if (lags_cycles is None):
        lags_cycles = np.arange(0, 20.1, 0.1)

    data_if = compute_instantaneous_frequency(data, sampling_rate=sfreq).mean(axis=-1)
    data_normed = data if is_normed else data / xp.abs(data)
    data_conj = xp.conj(data_normed)
    phase_diff = xp.empty_like(data)

    pac_vals = xp.zeros((data.shape[0], len(lags_cycles)))
    
    for lag_idx, lag in enumerate(lags_cycles):
        lag_samples = xp.rint(lag*sfreq/data_if).astype(int)
        
        if xp is np:
            data_shift = xp.roll(data_conj, lag_samples, axis=-1)
            xp.multiply(data_normed, data_shift, out=phase_diff)
        else: 
            # in theory, we dont need to check for cupy becausy not numpy -> always cupy
            _pac_kernel(data_normed, data_conj, lag_samples, data.shape[-1], phase_diff)


        pac_vals[..., lag_idx] = xp.abs(xp.nanmean(phase_diff, axis=-1))

    if not(xp is np):
        pac_vals = pac_vals.get()
    
    if return_lifetime:
        pac_final = np.zeros(data.shape[0])

        if method == 'lifetime':
            pac_agg_method = get_length_by_cdf
        elif method == 'mean':
            pac_agg_method = get_length_by_mean
        else:
            raise RuntimeError(f'Cannot find {method} for pACF lifetime computation!')
        
        for i in range(pac_vals.shape[0]):
            pac_final[i] = pac_agg_method(pac_vals[i], lag_values=lags_cycles)

    else:
        pac_final = pac_vals

    return pac_final


def compute_tfr_pacf(data: NDArray[complex], sfreq: float, lags_cycles: Sequence[float]=None, window_size: float=3.0, is_normed=False, **kwargs) -> NDArray[float]:
    """
    Compute time-frequency resolved phase autocorrelation.

    Parameters
    ----------
    data : NDArray[complex]
        Complex time series of shape (n_channels, n_samples).
    sfreq : float
        Sampling frequency in Hz.
    lags_cycles : Sequence[float], optional
        Lags in cycles to evaluate. Defaults to 1..2 in steps of 0.05.
    window_size : float, optional
        Window size in cycles used for moving average.
    is_normed : bool, optional
        If True, assumes data are already normalized to unit magnitude.
    **kwargs
        Additional arguments passed to the moving average function.

    Returns
    -------
    NDArray[float]
        Phase similarity values of shape (n_channels, n_samples).
    """
    xp = get_module(data)

    if (lags_cycles is None):
        lags_cycles = np.arange(2, 4, 0.05)

    data_if = compute_instantaneous_frequency(data, sampling_rate=sfreq).mean(axis=-1)
    data_normed = data if is_normed else data / xp.abs(data)
    data_conj = xp.conj(data_normed)
    phase_diff = xp.empty_like(data)
    phase_similarity = xp.zeros(data.shape, dtype=float)

    for lag_idx, lag in enumerate(lags_cycles):
        lag_samples = xp.rint(lag*sfreq/data_if).astype(int)
        
        if xp is np:
            data_shift = xp.roll(data_conj, lag_samples, axis=-1)
            xp.multiply(data_normed, data_shift, out=phase_diff)
        else:
            _pac_kernel(data_normed, data_conj, lag_samples, data.shape[-1], phase_diff)

        phase_diff[xp.isnan(phase_diff)] = 0.0

        window_size_samples = int(xp.rint(window_size*sfreq/data_if).mean())
        phase_similarity += xp.abs(moving_average_fast(phase_diff, window_size_samples))
    
    phase_similarity /= len(lags_cycles)

    return phase_similarity
