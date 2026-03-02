import glob
import os

import numpy as np
import pytest

from crocopy.observables.connectivity.amplitude_correlations import compute_cc, compute_occ
from crocopy.observables._base import HAS_CUPY

@pytest.mark.skip
def _generate_random_data(n_chans=10, n_ts=10000, method='random'):
    np.random.seed(42)

    random_phases = np.random.uniform(-np.pi, np.pi, size=(10,10000))
    random_envelopes = np.random.uniform(0, 1, size=random_phases.shape)

    test_data = np.exp(1j*random_phases)*random_envelopes

    return test_data

def test_cc_correctness():
    test_data = _generate_random_data()

    res_crocopy = compute_cc(test_data)
    res_numpy = np.corrcoef(np.abs(test_data))

    np.testing.assert_allclose(res_crocopy, res_numpy, atol=1e-4)

@pytest.mark.skipif(not HAS_CUPY, reason="CuPy not installed")
def test_occ_runnable_gpu():
    import cupy as cp
    
    test_data_gpu = cp.array(_generate_random_data(n_chans=100, n_ts=1000000))

    res_gpu = compute_occ(test_data_gpu).get()

    assert(np.isfinite(res_gpu).all())

@pytest.mark.skipif(not HAS_CUPY, reason="CuPy not installed")
def test_occ():
    import cupy as cp
    
    test_data_cpu = _generate_random_data()
    test_data_gpu = cp.array(test_data_cpu)

    res_cpu = compute_occ(test_data_cpu)
    res_gpu = compute_occ(test_data_gpu).get()

    np.testing.assert_allclose(res_cpu, res_gpu, atol=1e-4)

@pytest.mark.skipif(not HAS_CUPY, reason="CuPy not installed")
def test_occ_different_dtypes():
    import cupy as cp
    
    test_data_cpu = _generate_random_data()
    test_data_gpu_32 = cp.array(test_data_cpu).astype(np.complex128)
    test_data_gpu_64 = test_data_gpu_32.astype(np.complex64)

    res_gpu_32 = compute_occ(test_data_gpu_32).get()
    res_gpu_64 = compute_occ(test_data_gpu_64).get()

    np.testing.assert_allclose(res_gpu_32, res_gpu_64, atol=1e-4)


@pytest.mark.skipif(not HAS_CUPY, reason="CuPy not installed")
def test_occ_slices():
    import cupy as cp
    
    test_data_cpu = _generate_random_data()
    test_data_gpu = cp.array(test_data_cpu)

    test_data_gpu_slice = test_data_gpu[:, 1000:10000]
    test_data_gpu_copy = test_data_gpu[:, 1000:10000].copy()

    res_gpu_slice = compute_occ(test_data_gpu_slice).get()
    res_gpu_copy = compute_occ(test_data_gpu_copy).get()

    np.testing.assert_allclose(res_gpu_slice, res_gpu_copy, atol=1e-4)

@pytest.mark.skipif(not HAS_CUPY, reason="CuPy not installed")
def test_cc_cupy():
    """Test cupy and numpy BiS outputs are the same."""
    import cupy as cp
    
    test_data_cpu = _generate_random_data()
    test_data_gpu = cp.array(test_data_cpu)

    res_cpu = compute_cc(test_data_cpu)
    res_gpu = compute_cc(test_data_gpu).get()

    np.testing.assert_allclose(res_cpu, res_gpu, atol=1e-4)