import numpy as np
import scipy as sp

from ...observables._base import NDArray, get_module, supports_gpu, HAS_CUPY

from typing import Optional

from ...preprocessing.signal import orthogonalize_signals

if HAS_CUPY:   
    from ...observables.cupy_kernels import  _occ_cupy

from ...observables.numba_kernels import _occ_numba

@supports_gpu
def compute_cc(x: NDArray[float | complex], y: Optional[NDArray[float | complex]] = None, zero_diag: bool=False, is_envelope=False) -> NDArray[float]:
    """
    Computes correlation coefficients between signal envelopes.
    
    Parameters
    ----------
    x : NDArray[float | complex]
        Input array of shape (n_channels, n_samples).
    y : NDArray[float | complex], optional
        Optional second input array. If None, computes CC between channels in ``x``.
    zero_diag : bool, optional
        If True, sets diagonal elements to zero.
    is_envelope : bool, optional
        If True, assumes inputs are already envelopes.
    
    Returns:
    -------
    NDArray[float]
        Correlation coefficient matrix of shape (n_channels, n_channels).
    """

    xp  = get_module(x)

    n_ch = x.shape[0]

    x_envelope = x if is_envelope else xp.abs(x)

    if y is None:
        y_envelope = x_envelope
    else:
        y_envelope = y if is_envelope else xp.abs(y)

    res = xp.corrcoef(x_envelope, y_envelope)[n_ch:,:n_ch] 

    return res

def _occ_numpy(x: np.ndarray[complex]) -> np.ndarray[float]:
    """
    Numba kernel for the orthogonalized correlation coefficient
    """
    res = np.zeros((x.shape[0], x.shape[0]))

    for i in range(x.shape[0]):
        for j in range(x.shape[0]):
            yo = orthogonalize_signals(x[i], x[j])

            res[i,j] = sp.stats.pearsonr(np.abs(x[i]), np.abs(yo))[0]
    
    return res


def compute_occ(x: NDArray[complex], force_gpu: bool=False, use_numba: bool=False) -> NDArray[float]:
    """
    Computes orthogonalized correlation coefficient between all channels of input data.
    Implementation follows Hipp et al. 2012 (https://www.nature.com/articles/nn.3101).
    
    Parameters
    ----------
    x : NDArray[complex]
        Complex-valued array of shape (n_channels, n_samples).
    force_gpu : bool, optional
        Whether to try using GPU acceleration via CuPy.
    use_numba : bool, optional
        Whether to use Numba-accelerated CPU kernels.
        
    Returns
    -------
    NDArray[float]
        Matrix of OCC values between all channel pairs.
        
    Raises
    ------
    EnvironmentError
        If ``force_gpu=True`` but CuPy is not installed.
    """
    if not(np.iscomplexobj(x)):
        raise TypeError('Data should contain complex values!')

    xp = get_module(x, force_gpu)

    x_device = xp.asarray(x)

    if HAS_CUPY and not(xp is np):            
        res = _occ_cupy(x_device)        
    elif use_numba:
        res = _occ_numba(x_device)
    else:
        res = _occ_numpy(x_device)

    xp.fill_diagonal(res, 0.0)  

    return res
