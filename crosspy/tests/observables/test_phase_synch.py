import numpy as np
import pytest

from crosspy.observables.connectivity.synchrony import compute_wpli, compute_cplv
from crosspy.observables._base import HAS_CUPY

@pytest.mark.skip
def _generate_random_data(n_chans=2, n_ts=10000, method='random'):
    np.random.seed(42)

    random_phases = np.random.uniform(-np.pi, np.pi, size=(n_chans,10000))
    random_envelopes = np.random.uniform(0, 1, size=random_phases.shape)

    test_data = np.exp(1j*random_phases)*random_envelopes

    return test_data

@pytest.mark.skipif(not HAS_CUPY, reason="CuPy not installed")
def test_wpli_runnable_gpu():
    import cupy as cp
    
    test_data_gpu = cp.array(_generate_random_data(n_chans=100, n_ts=1000000))

    res_gpu = compute_wpli(test_data_gpu).get()

    assert(np.isfinite(res_gpu).all())


@pytest.mark.skipif(not HAS_CUPY, reason="CuPy not installed")
def test_wpli_sanity():
    import cupy as cp
    
    test_data_cpu = _generate_random_data()
    test_data_gpu = cp.array(test_data_cpu)

    res_cpu = compute_wpli(test_data_cpu)
    res_cpu_numba = compute_wpli(test_data_cpu, use_numba=True)
    res_gpu = compute_wpli(test_data_gpu).get()

    np.testing.assert_allclose(res_cpu, res_gpu, atol=1e-4)
    np.testing.assert_allclose(res_cpu_numba, res_gpu, atol=1e-4)
    np.testing.assert_allclose(res_cpu, res_cpu_numba, atol=1e-4)

@pytest.mark.skipif(not HAS_CUPY, reason="CuPy not installed")
def test_wpli_slice():
    import cupy as cp
    
    test_data_cpu = _generate_random_data()
    test_data_gpu = cp.array(test_data_cpu)

    test_data_gpu_slice = test_data_gpu[:, 1000:10000]
    test_data_gpu_copy = test_data_gpu[:, 1000:10000].copy()

    res_gpu_slice = compute_wpli(test_data_gpu_slice).get()
    res_gpu_copy = compute_wpli(test_data_gpu_copy).get()

    np.testing.assert_allclose(res_gpu_slice, res_gpu_copy, atol=1e-4)

@pytest.mark.skipif(not HAS_CUPY, reason="CuPy not installed")
def test_wpli_sanity_debias():
    import cupy as cp
    
    test_data_cpu = _generate_random_data()
    test_data_gpu = cp.array(test_data_cpu)

    res_cpu = compute_wpli(test_data_cpu, debias=True)
    res_gpu = compute_wpli(test_data_gpu, debias=True).get()

    np.testing.assert_allclose(res_cpu, res_gpu, atol=1e-4)

@pytest.mark.skipif(not HAS_CUPY, reason="CuPy not installed")
def test_plv_sanity():
    import cupy as cp
    
    test_data_cpu = _generate_random_data()
    test_data_gpu = cp.array(test_data_cpu)

    res_cpu = compute_cplv(test_data_cpu)
    res_gpu = compute_cplv(test_data_gpu).get()

    np.testing.assert_allclose(res_cpu, res_gpu, atol=1e-4)