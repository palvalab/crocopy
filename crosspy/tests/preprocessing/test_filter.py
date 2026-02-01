import numpy as np

import pytest

from crosspy.observables._base import HAS_CUPY
from crosspy.preprocessing.signal import filter_data

@pytest.mark.skipif(not HAS_CUPY, reason="CuPy not installed")
def test_gpu_filter_working():
    import cupy as cp

    filter_freq = 10.0
    sfreq = 1000.0
    omega = 5.0
    
    test_data_gpu = cp.random.normal(size=(10, 100000))
    test_data_filt = filter_data(test_data_gpu, sfreq=sfreq, frequency=filter_freq, omega=omega, n_jobs='cuda')

    assert(np.isfinite(test_data_filt).all())

@pytest.mark.skipif(not HAS_CUPY, reason="CuPy not installed")
def test_gpu_filter_sanity():
    import cupy as cp
    import cupyx.scipy.signal

    filter_freq = 10.0
    sfreq = 1000.0
    omega = 5.0
    
    test_data_gpu = cp.random.normal(size=(10, 100000))
    test_data_filt = filter_data(test_data_gpu, sfreq=sfreq, frequency=filter_freq, omega=omega, n_jobs='cuda')

    psd_freqs, psd_vals = cupyx.scipy.signal.welch(test_data_filt.real, fs=1000, nperseg=256*16)

    psd_freq_diff = float(psd_freqs[1] - psd_freqs[0])

    peak_indices = psd_vals.argmax(axis=-1)
    peak_freqs = psd_freqs[peak_indices].get()
    peak_freq_dist = np.abs(peak_freqs - filter_freq)

    assert((peak_freq_dist <= 5*psd_freq_diff).all())

@pytest.mark.skipif(not HAS_CUPY, reason="CuPy not installed")
def test_gpu_filter_compare_with_mne():
    import cupy as cp
    import cupyx.scipy.signal
    import mne

    sfreq = 1000.0
    filter_freq = 10.0
    omega = 5.0

    test_data = np.random.normal(size=(10, 10000))
    test_data_filt_cp = filter_data(test_data, sfreq=sfreq, frequency=filter_freq, omega=omega, n_jobs='cuda', normalize_wavelet=False).get()
    test_data_filt_mne = mne.time_frequency.tfr_array_morlet(test_data[None], sfreq=sfreq, freqs=[filter_freq], n_cycles=omega).squeeze()

    np.testing.assert_allclose(test_data_filt_cp, test_data_filt_mne, atol=1e-4)
