from pathlib import Path

import numpy as np
import pytest

from crosspy.utils.surrogates import compute_surrogate_statistic
from crosspy.observables._base import HAS_CUPY
import crosspy

def test_different_methods():
    """
    Test that the function works correctly for different methods.
    """
    N_chans = 50
    N_ts = 100000
    methods = ['random_phase', 'time_shift', 'noise']
    
    data = np.exp(1j*np.random.normal(0, 1, size=(N_chans, N_ts)))
    
    for method in methods:
        output = compute_surrogate_statistic(data, method, observable_function='plv')
        
        assert output.shape == (N_chans, N_chans)

def test_different_observable_functions():
    """
    Test that the function works correctly for different observable functions.
    """
    N_chans = 50
    N_ts = 100000

    data = np.exp(1j*np.random.normal(0, 1, size=(N_chans, N_ts)))

    observable_functions = [{'name': 'plv', 'shape': (N_chans, N_chans), 'kwargs': {}}, 
                            {'name': 'wpli', 'shape': (N_chans, N_chans), 'kwargs': {'debias': True}}, 
                            {'name': 'dfa', 'shape': (N_chans,), 'kwargs': {'window_lengths': np.geomspace(100, 20000, 30)}}]
    
    for func_info in observable_functions:
        output = compute_surrogate_statistic(data, method='time_shift', observable_function=func_info['name'], **func_info['kwargs'])
        
        assert output.shape == func_info['shape']

@pytest.mark.skipif(not HAS_CUPY, reason="CuPy not installed")
def test_surrogates_different_devices():
    """
    Test that the function works correctly for GPU
    """
    import cupy as cp
    N_chans = 50
    N_ts = 100000

    data = cp.exp(1j*cp.random.normal(0, 1, size=(N_chans, N_ts)))

    methods = ['random_phase', 'time_shift', 'noise']
    observable_functions = [{'name': 'plv', 'shape': (N_chans, N_chans), 'kwargs': {}, 'same_device': True}, 
                            {'name': 'wpli', 'shape': (N_chans, N_chans), 'kwargs': {'debias': True}, 'same_device': True}, 
                            {'name': 'dfa', 'shape': (N_chans,), 'kwargs': {'window_lengths': np.geomspace(100, 20000, 30)}, 'same_device': False}]
    
    for method in methods:
        for func_info in observable_functions:
            output = compute_surrogate_statistic(data, method=method, observable_function=func_info['name'], **func_info['kwargs'])
            
            assert output.shape == func_info['shape']
            if func_info['same_device']:
                assert output.device == data.device
