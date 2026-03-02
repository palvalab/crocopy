from typing import Tuple, Sequence
from collections import namedtuple

import numpy as np
import numba

import mne

import powerlaw

AvalancheScalingResult = namedtuple('AvalancheScalingResult', ['lengths_exponent', 'sizes_exponent', 'branching_ratio'])

@numba.jit(nopython=True)
def _avalanche_peak_detection(channel_values, std_threshold: float=3.0, sign_invariant: bool=True):
    channel_values_abs = np.abs(channel_values) if sign_invariant else channel_values
    channel_mask = (channel_values_abs > std_threshold)

    res = np.zeros_like(channel_mask)
    n_chans, n_ts = channel_mask.shape

    for i in range(n_chans):
        peak_value = 0.0
        avalanche_peak_idx = 0
        for j in range(n_ts):
            if channel_mask[i, j] and (channel_values_abs[i, j] > peak_value):
                peak_value = channel_values_abs[i, j]
                avalanche_peak_idx = j
            elif (peak_value > 0):
                res[i, avalanche_peak_idx] = True
                peak_value = 0
        if peak_value > 0:
            res[i, avalanche_peak_idx] = True

    return res

def detect_avalanche_peaks(data, std_threshold: float=3.0, bin_size: int=8, sign_invariant: bool=True, sfreq: int=1000, l_freq: int=1, h_freq: int=120, filt_n_jobs: int=8):
    """
    Detect avalanche peaks from multichannel data.

    Parameters
    ----------
    data : array-like
        Multichannel time series data of shape (n_channels, n_samples).
    std_threshold : float, optional
        Threshold in standard deviations for peak detection.
    bin_size : int, optional
        Size of time bins in samples.
    sign_invariant : bool, optional
        If True, use absolute value for threshold detection.
    sfreq : int, optional
        Sampling frequency in Hz.
    l_freq : int, optional
        Low-pass filter cutoff in Hz.
    h_freq : int, optional
        High-pass filter cutoff in Hz.
    filt_n_jobs : int, optional
        Number of parallel jobs for filtering.

    Returns
    -------
    ndarray
        Boolean array of detected peaks with shape (n_channels, n_samples).
    """
    data_filt = mne.filter.filter_data(data, sfreq, l_freq=l_freq, h_freq=h_freq, verbose=False, n_jobs=filt_n_jobs)
    data_std = data_filt.std(axis=-1, keepdims=True)
    data_normed = (data_filt - data_filt.mean(axis=-1, keepdims=True)) / np.where(data_std > 0, data_std, 1.0)


    avalanche_peaks = _avalanche_peak_detection(data_normed, std_threshold, sign_invariant=sign_invariant)

    return avalanche_peaks

def compute_avalanche_properties(data, std_threshold: float=3.0, bin_size: int=8, sign_invariant: bool=True, sfreq: int=1000, l_freq: int=1, h_freq: int=120, filt_n_jobs: int=8) -> Tuple[Sequence[int], Sequence[int]]:
    """
    Compute avalanche sizes and durations from multichannel data.
    
    Parameters
    ----------
    data : array-like
        Multichannel time series data [n_channels, n_samples]
    std_threshold : float
        Threshold in standard deviations for peak detection
    bin_size : int
        Size of time bins in samples
    sign_invariant : bool
        If True, use absolute value for threshold detection
    sfreq : int
        Sampling frequency in Hz
    l_freq : int
        Low-pass filter cutoff
    h_freq : int
        High-pass filter cutoff
    filt_n_jobs : int
        Number of parallel jobs for filtering
    
    Returns
    -------
    avalanche_sizes : array
        Array of avalanche sizes (number of channels active per bin)
    avalanche_lengths : list
        List of avalanche durations (number of consecutive bins)
    """
    avalanche_peaks = detect_avalanche_peaks(data, std_threshold, bin_size, sign_invariant, sfreq, l_freq, h_freq, filt_n_jobs)
    
    # Compute number of bins and trim data
    n_bins = avalanche_peaks.shape[-1] // bin_size
    avalanche_peaks_trimmed = avalanche_peaks[..., :n_bins * bin_size]
    avalanche_peaks_binned = avalanche_peaks_trimmed.reshape(*avalanche_peaks.shape[:-1], n_bins, bin_size)

    avalanches_binned = avalanche_peaks_binned.any(axis=-1)

    avalanche_sizes = avalanches_binned.sum(axis=0)
    nonzero_mask = (avalanche_sizes > 0)
    padded = np.concatenate([[False], nonzero_mask, [False]])
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    avalanche_lengths = list(edges[1::2] - edges[::2])

    return avalanche_sizes, avalanche_lengths

def compute_avalanche_metrics(data, std_threshold: float=3.0, bin_size: int=8, sign_invariant: bool=True, sfreq: int=1000, l_freq: int=1, h_freq: int=120, filt_n_jobs: int=8) -> AvalancheScalingResult:
    """
    Compute avalanche scaling exponents and branching ratio.

    Parameters
    ----------
    data : array-like
        Multichannel time series data of shape (n_channels, n_samples).
    std_threshold : float, optional
        Threshold in standard deviations for peak detection.
    bin_size : int, optional
        Size of time bins in samples.
    sign_invariant : bool, optional
        If True, use absolute value for threshold detection.
    sfreq : int, optional
        Sampling frequency in Hz.
    l_freq : int, optional
        Low-pass filter cutoff in Hz.
    h_freq : int, optional
        High-pass filter cutoff in Hz.
    filt_n_jobs : int, optional
        Number of parallel jobs for filtering.

    Returns
    -------
    tuple
        (size_scaling_exponent, length_scaling_exponent, branching_ratio).
    """
    avalanche_sizes, avalanche_lengths = compute_avalanche_properties(data, std_threshold, bin_size, sign_invariant, sfreq, l_freq, h_freq, filt_n_jobs)

    sizes_powerlaw_fit = powerlaw.Fit(avalanche_sizes, discrete=True)
    lengths_powerlaw_fit = powerlaw.Fit(avalanche_lengths, discrete=True)

    size_scaling_exponent = sizes_powerlaw_fit.alpha
    length_scaling_exponent = lengths_powerlaw_fit.alpha
    branching_ratio = estimate_branching_ratio(avalanche_sizes)

    return AvalancheScalingResult(lengths_exponent=length_scaling_exponent, sizes_exponent=size_scaling_exponent, branching_ratio=branching_ratio)

@numba.jit(nopython=True)
def estimate_branching_ratio(avalanche_sizes: Sequence[int]) -> list[float]:
    """
    Estimate branching ratio from avalanche sizes.

    Parameters
    ----------
    avalanche_sizes : array-like
        Sequence of avalanche sizes per bin.

    Returns
    -------
    list
        Estimated branching ratios for each avalanche onset.
    """
    res = list()
    for i in range(1, len(avalanche_sizes) - 1):
        if (avalanche_sizes[i-1] == 0) and (avalanche_sizes[i] > 0):
            p = avalanche_sizes[i+1]/avalanche_sizes[i]
            res.append(p)
    
    return res

def compute_normalized_count(data, is_avalanches: bool=False, bin_size: int=8, **kwargs) -> np.ndarray[float]:
    """
    Compute normalized avalanche transition counts between channels.
    
    Parameters
    ----------
    data : array-like
        Either raw multichannel data or pre-computed avalanche binary matrix
    is_avalanches : bool
        If True, data is already avalanche binary matrix. If False, compute it.
    bin_size : int
        Size of time bins in samples (used if is_avalanches=False)
    **kwargs
        Additional arguments passed to detect_avalanche_peaks
    
    Returns
    -------
    res : ndarray
        Matrix of normalized transition counts [n_channels, n_channels]
    """
    if is_avalanches:
        X = data
    else:
        # Compute avalanche binary matrix
        avalanche_peaks = detect_avalanche_peaks(data, bin_size=bin_size, **kwargs)
        n_bins = avalanche_peaks.shape[-1] // bin_size
        avalanche_peaks_trimmed = avalanche_peaks[..., :n_bins * bin_size]
        avalanche_peaks_binned = avalanche_peaks_trimmed.reshape(*avalanche_peaks.shape[:-1], n_bins, bin_size)
        X = avalanche_peaks_binned.any(axis=-1)

    na = X.sum(axis=0)
    X_valid = X[..., na > 0]
    na = na[na > 0]

    n_chans, n_ts = X_valid.shape

    res = np.zeros((n_chans, n_chans))

    for i in range(n_chans):
        for j in range(n_chans):
            res[i,j] = (X_valid[i, :-1]*X_valid[j, 1:]/na[:-1]).sum()/X_valid.shape[-1]
    
    return res

