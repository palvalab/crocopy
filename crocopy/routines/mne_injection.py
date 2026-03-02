from typing import Sequence

import mne
import numpy as np

from .._base import NDArray, HAS_CUPY, get_module
from ..observables.criticality.lrtc import compute_dfa
from ..observables.connectivity.synchrony import compute_wpli, compute_cplv
from ..observables.connectivity.amplitude_correlations import compute_cc, compute_occ
from ..preprocessing.signal import filter_data

if HAS_CUPY:
    import cupy as cp

_CONNECTIVITY_METHODS_MAP = {'cplv': lambda x: np.abs(compute_cplv(x)),
                             'iplv': lambda x: compute_cplv(x).imag,
                             'wpli': compute_wpli,
                             'cc': compute_cc,
                             'occ': compute_occ}

def _compute_dfa_wrapper(self, frequencies: Sequence[float], omega: float=5.0,
                         min_window_size: float=10.0, max_window_size: float=0.3, n_windows: int=30, 
                         force_gpu: bool=False) -> NDArray[float]:
    """
    Compute Detrended Fluctuation Analysis (DFA) for a given range of frequencies.
    
    This function performs DFA analysis for each input frequency.
    
    Args:
        frequencies: Sequence of frequencies to analyze in Hz
        omega: Parameter for the Morlet wavelets used in filtering, default=7.5
        min_window_size: Minimum window size in number of cycles of the target frequency, default=10.0
        max_window_size: Maximum window size as a fraction of the total data length, default=0.3
        n_windows: Number of logarithmically spaced window sizes to use, default=30
        force_gpu: Whether to use GPU acceleration if available, default=False
    
    Returns:
        NDArray[float]: Array of shape [n_frequencies, n_channels] containing DFA exponents
                       for each frequency and channel

    """

    data = self.get_data()

    if force_gpu and HAS_CUPY:
        data = cp.asarray(data)
        n_jobs = 'cuda'
    else:
        n_jobs = 8

    res = np.zeros((len(frequencies), data.shape[0]))

    for freq_idx, freq in enumerate(frequencies):
        frequency_window_sizes = np.geomspace(min_window_size*self.info['sfreq']/freq, max_window_size*data.shape[-1], n_windows).astype(int)

        data_filtered = filter_data(data, self.info['sfreq'], freq, n_jobs=n_jobs, omega=omega)
        data_envelope = np.abs(data_filtered)
        res[freq_idx] = compute_dfa(data_envelope, frequency_window_sizes).dfa_values

    return res

def _compute_connectivity_wrapper(self, frequencies: Sequence[float], method: str, omega: float=5.0,
                         force_gpu: bool=False, **method_kwargs) -> NDArray[float]:
    """
    Compute Detrended Fluctuation Analysis (DFA) for a given range of frequencies.
    
    This function performs DFA analysis for each input frequency.
    
    Args:
        frequencies: Sequence of frequencies to analyze in Hz
        omega: Parameter for the Morlet wavelets used in filtering, default=7.5
        min_window_size: Minimum window size in number of cycles of the target frequency, default=10.0
        max_window_size: Maximum window size as a fraction of the total data length, default=0.3
        n_windows: Number of logarithmically spaced window sizes to use, default=30
        force_gpu: Whether to use GPU acceleration if available, default=False
    
    Returns:
        NDArray[float]: Array of shape [n_frequencies, n_channels] containing DFA exponents
                       for each frequency and channel

    """

    data = self.get_data()

    if force_gpu and HAS_CUPY:
        data = cp.asarray(data)
        n_jobs = 'cuda'
    else:
        n_jobs = 8

    if not(method in _CONNECTIVITY_METHODS_MAP):
        raise RuntimeError(f'compute connectivity: unknown method {method}! only {_CONNECTIVITY_METHODS_MAP.keys()} are supported')

    method_func = _CONNECTIVITY_METHODS_MAP[method]

    n_chans = data.shape[0]
    res = np.zeros((len(frequencies), n_chans, n_chans))

    for freq_idx, freq in enumerate(frequencies):
        data_filtered = filter_data(data, self.info['sfreq'], freq, n_jobs=n_jobs, omega=omega)
        method_res = method_func(data_filtered, **method_kwargs)

        if not(get_module(method_res)) is np:
            method_res = method_res.get()

        res[freq_idx] = method_res

    return res

setattr(mne.io.BaseRaw, 'compute_dfa', _compute_dfa_wrapper)
setattr(mne.io.BaseRaw, 'compute_connectivity', _compute_connectivity_wrapper)