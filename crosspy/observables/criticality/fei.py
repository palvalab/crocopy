import numpy as np
import scipy as sp
import scipy.signal

from collections import namedtuple

from joblib import Parallel, delayed

from typing import Tuple, Sequence, Optional

from ...observables._base import HAS_CUPY, NDArray, get_module

if HAS_CUPY:
    from ...observables.cupy_wrappers import cupy_detrend
    
def compute_fei(data: NDArray[float], window_size: int, overlap: float=0.75, force_gpu: bool=False) -> NDArray[float]:
    """
    Compute fEI for a given 2D signal envelope.

    Parameters
    ----------
    data : NDArray[float]
        2D array of size (n_channels, n_samples), typically a signal envelope.
    window_size : int
        Window size in samples used to split a signal.
    overlap : float, optional
        Fractional overlap between consecutive windows in [0, 1).
    force_gpu : bool, optional
        If True, force GPU computation when CuPy is available.

    Returns
    -------
    NDArray[float]
        1D array of fEI values of shape (n_channels,).

    TODO
    -------
    Robust regression with outlier detection
    """
    if window_size <= 0:
        raise ValueError('fEI: window size must be positive')

    if not(0 <= overlap < 1):
        raise ValueError('fEI: overlap must in in [0, 1)')

    xp = get_module(data, force_gpu)

    data_device = xp.asarray(data)

    n_chans, n_ts = data_device.shape
    step_size = max(1, int(round(window_size * (1 - overlap))))

    # split data on windows and compute average in each window
    # data_windowed = split_on_windows(data_device, window_size, step_size)
    data_windowed = xp.lib.stride_tricks.sliding_window_view(data_device, window_shape=window_size, axis=-1)[:, ::step_size]
    data_win_averaged = data_windowed.mean(axis=-1)

    # compute signal profile, split it on windows and normalize it on average signal in each window
    signal_profile = xp.cumsum(data_device - xp.mean(data_device, axis=-1, keepdims=True), axis=-1)
    # profile_windowed = split_on_windows(signal_profile, window_size, step_size) / data_win_averaged[...,None]
    profile_windowed = xp.lib.stride_tricks.sliding_window_view(signal_profile, window_shape=window_size, axis=-1)[:, ::step_size]
    profile_windowed /= data_win_averaged[..., None]

    # detrend each window. By default detrend works with the last axis and we already have N_chans  x N_windows x window_size tensor.
    profile_detrended = sp.signal.detrend(profile_windowed) if xp is np else cupy_detrend(profile_windowed)

    fluctuations = xp.std(profile_detrended, axis=-1, ddof=1)

    full_corr = xp.corrcoef(fluctuations, data_win_averaged)
    # corrcoeff returns a 2*N_chans x 2*N_chans because it concatenates both arguments and compute pairwise correlations
    # because we need just correlation of the same rows and because in concat array the second matrix starts from N_chans index
    # we need to extract cells with coords (0,N_chans), (1,N_chans+1) ... or diagonal elements of [:n_chans,n_chans:] subarray.
    # Why not compuate correlation manually using base definitions: numeric stability.
    ei = 1 - xp.diag(full_corr[:n_chans,n_chans:])
    
    return ei
