from pathlib import Path

import mne
import numpy as np
import pytest

from crocopy.utils.surrogates import compute_surrogate_statistic
from crocopy.observables._base import HAS_CUPY

import crocopy.routines.mne_injection

def test_dfa_wrapper():
    """
    Test that the DFA wrapper works correctly.
    """
    data = np.random.normal(0, 1, size=(10, 100000))
    ch_names = [f'ch{i}' for i in range(data.shape[0])]
    ch_types = ['eeg']*data.shape[0]
    data_info = mne.create_info(ch_names, 1000.0, ch_types)
    raw = mne.io.RawArray(data, data_info, verbose=False)

    frequencies = np.geomspace(1, 50, 25)

    dfa = raw.compute_dfa(frequencies)

    assert dfa.shape == (len(frequencies), data.shape[0])

@pytest.mark.skipif(not HAS_CUPY, reason="CuPy not installed")
def test_dfa_wrapper_gpu():
    """
    Test that the DFA wrapper works correctly.
    """
    data = np.random.normal(0, 1, size=(10, 100000))
    ch_names = [f'ch{i}' for i in range(data.shape[0])]
    ch_types = ['eeg']*data.shape[0]
    data_info = mne.create_info(ch_names, 1000.0, ch_types)
    raw = mne.io.RawArray(data, data_info, verbose=False)

    frequencies = np.geomspace(1, 50, 25)

    dfa = raw.compute_dfa(frequencies, force_gpu=True)

    assert dfa.shape == (len(frequencies), data.shape[0])