import cupy as cp
import cupyx.scipy.signal

import scipy as sp
import numpy as np

import mne

from typing import Optional

from ..observables._base import NDArray

def get_morlet_wavelet(sampling_rate: float, frequency: float, omega: float, normalize: bool=False) -> NDArray[complex]:
    win = cp.array(mne.time_frequency.morlet(sampling_rate, [frequency], omega)[0])

    if normalize:
        win /= cp.abs(win).sum()

    return win

def _filter_routine_gpu(data_gpu: NDArray[float], sfreq: float, omega: float, morlet_frequency: float, data_buffer: Optional[NDArray[complex]]=None, normalize_wavelet: bool=False) -> NDArray[complex]:
    '''
    Implementation for filtering data with a Morlet wavelet in cupy.  
    Filters data at given frequency with Morlet wavelet,
    using either cupy or mne implementation.
    INPUT:
        x:           Cupy ndarray of shape [channels x samples]. 
        sfreq:       Data sampling frequency.
        frequency:   Center frequency for the Morlet wavelet.
        data_buffer: Optional; array into which to write the output.
    '''
    n_chans, n_ts = data_gpu.shape

    win = get_morlet_wavelet(sfreq, morlet_frequency, omega, normalize=normalize_wavelet)
    
    if data_buffer is None:
        data_preprocessed = cp.zeros_like(data_gpu, dtype=cp.complex64)
    else:
        data_preprocessed = data_buffer
    
    for i in range(n_chans):
        data_preprocessed[i] = cupyx.scipy.signal.convolve(data_gpu[i], win, mode='same', method='auto')
            
    return data_preprocessed