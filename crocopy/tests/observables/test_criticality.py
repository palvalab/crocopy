import glob
import os
import pickle
from pathlib import Path

import numpy as np
import pytest

from crocopy.observables.criticality.bistability import compute_BiS
from crocopy.observables.criticality.lrtc import compute_dfa
from crocopy.observables._base import HAS_CUPY

ROOT_PATH = Path(os.path.abspath(os.path.dirname(__file__))) / '..' / '..' / '..'
TEST_DATA_PATH = ROOT_PATH / 'data' / 'test_data'
BISTABILITY_TEST_DATA_PATH =  TEST_DATA_PATH / 'bistability'
DFA_TEST_DATA_PATH =  TEST_DATA_PATH / 'dfa'

def test_BiS_correctness():
    """
    Test that the function works correctly.
    Values are taken from the original implementation by Sheng Wang, author of the paper.
    Wang et.al. 2023
    https://onlinelibrary.wiley.com/doi/pdfdirect/10.1111/epi.17996
    """
    bis_test_files = glob.glob(str(BISTABILITY_TEST_DATA_PATH / '*.pickle'))
    for fpath in bis_test_files:
        test_data = pickle.load(open(fpath, 'rb'))

        output = compute_BiS(test_data['data'], is_envelope=True)
        np.testing.assert_allclose(output, test_data['expected_result'], atol=1e-4)


@pytest.mark.skipif(not HAS_CUPY, reason="CuPy not installed")
def test_BiS_different_devices():
    """Test cupy and numpy BiS outputs are the same."""
    import cupy as cp
    
    bis_test_files = glob.glob(str(BISTABILITY_TEST_DATA_PATH / '*.pickle'))
    for fpath in bis_test_files:
        test_data = pickle.load(open(fpath, 'rb'))

        cp_array = cp.array(test_data['data'])  
        np_array = test_data['data']

        cp_output = compute_BiS(cp_array, is_envelope=True).get()
        np_output = compute_BiS(np_array, is_envelope=True)

        np.testing.assert_allclose(cp_output, np_output, atol=1e-4)

def test_BiS_different_methods():    
    bis_test_files = glob.glob(str(BISTABILITY_TEST_DATA_PATH / '*.pickle'))
    for fpath in bis_test_files:
        test_data = pickle.load(open(fpath, 'rb'))

        em_output = compute_BiS(test_data['data'], method='em', is_envelope=True)
        mle_output = compute_BiS(test_data['data'], method='mle', is_envelope=True)

        np.testing.assert_allclose(em_output, mle_output, atol=1e-3)

def test_dfa_runnable():
    test_data = np.random.random(size=(10,100000))

    dfa_window_sizes = np.geomspace(100, test_data.shape[-1]//3, 30)

    dfa_res = compute_dfa(test_data, dfa_window_sizes)

    for field_name in ['fluctuation', 'r_squared', 'dfa_values']:
        assert(hasattr(dfa_res, field_name))
        
        field_values = getattr(dfa_res, field_name)
        
        assert(np.isfinite(field_values).all())

def test_dfa_correctness():
    central_frequency = 10.0
    sfreq = 200
    samples_per_cycle = sfreq / central_frequency

    dfa_test_files = glob.glob(str(DFA_TEST_DATA_PATH / '*.pickle'))
    for fpath in dfa_test_files:
        test_data = pickle.load(open(fpath, 'rb'))

        dfa_window_sizes = np.geomspace(int(10*samples_per_cycle), test_data['data'].shape[-1]//4, 30)
        signal = test_data['data'][None]

        dfa_res = compute_dfa(signal, dfa_window_sizes)
        np.testing.assert_allclose(dfa_res.dfa_values[0], test_data['expected_result'], atol=1e-4)

@pytest.mark.skipif(not HAS_CUPY, reason="CuPy not installed")
def test_dfa_different_devices():
    central_frequency = 10.0
    sfreq = 200
    samples_per_cycle = sfreq / central_frequency

    dfa_test_files = glob.glob(str(DFA_TEST_DATA_PATH / '*.pickle'))
    for fpath in dfa_test_files:
        test_data = pickle.load(open(fpath, 'rb'))

        dfa_window_sizes = np.geomspace(int(10*samples_per_cycle), test_data['data'].shape[-1]//4, 30)
        signal = test_data['data'][None]

        dfa_cpu_res = compute_dfa(signal, dfa_window_sizes, method='fft', force_gpu=False)
        dfa_gpu_res = compute_dfa(signal, dfa_window_sizes, method='fft', force_gpu=True)

        np.testing.assert_allclose(dfa_cpu_res.dfa_values[0], dfa_gpu_res.dfa_values[0], atol=1e-4)