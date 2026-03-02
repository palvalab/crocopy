import numba
import numpy as np

from ..observables._base import NDArray

@numba.njit(parallel=True)
def _occ_numba(S: NDArray[complex]) -> NDArray[float]:
    """
    Compute orthogonalized correlation coefficient using Numba acceleration.
    
    Implements the method from Hipp et al. (2012) for removing artificial 
    correlations due to volume conduction.
    
    Args:
        S: Complex-valued array of shape [channels x samples]
    
    Returns:
        np.ndarray: Matrix of orthogonalized correlation coefficients
    """        
    n_signals, n_samples = S.shape
    corr = np.zeros((n_signals, n_signals), dtype=S.real.dtype)

    for i in numba.prange(n_signals):
        x = S[i]
        x_abs_arr = np.empty(n_samples, dtype=S.real.dtype)
        x_normed_conj_arr = np.empty(n_samples, dtype=S.dtype)
        x_normed_j_arr = np.empty(n_samples, dtype=S.dtype)
        sumX = 0.0
        sumXX = 0.0

        for k in range(n_samples):
            x_abs = np.abs(x[k]) + 1e-14
            x_abs_arr[k] = x_abs
            x_normed = x[k] / x_abs
            x_normed_conj_arr[k] = np.conj(x_normed)
            x_normed_j_arr[k] = 1j * x_normed
            sumX += x_abs
            sumXX += x_abs * x_abs

        for j in range(n_signals):
            y = S[j]

            sumY = 0.0
            sumYY = 0.0
            sumXY = 0.0

            for k in range(n_samples):
                x_abs = x_abs_arr[k]

                #    ls = imag(y * conj(x_normed))
                #    rs = 1j * x_normed
                ls = np.imag(y[k] * x_normed_conj_arr[k])
                rs = x_normed_j_arr[k]
                
                y_orth = ls * rs

                Y = np.abs(y_orth)

                sumY  += Y
                sumYY += Y * Y
                sumXY += x_abs * Y

            # Pearson correlation
            num = sumXY - (sumX * sumY / n_samples)
            den = np.sqrt(
                (sumXX - (sumX*sumX / n_samples)) *
                (sumYY - (sumY*sumY / n_samples))
            )
            r = 0.0
            if den > 1e-14:
                r = num / den

            corr[i, j] = r

    return corr

def _wpli_numba(x: NDArray[complex], y: NDArray[complex], debias: bool = False) -> NDArray[float]:
    y_conj = np.conj(y) 
    
    outsum = np.inner(x, y_conj).imag
    outsum_envelope = _inner_with_abs_numba(x, y_conj)
    
    if debias:
        outsum_square = _inner_with_square_numba(x, y_conj)
        
        outsum = outsum**2 - outsum_square
        outsum_envelope = outsum_envelope**2 - outsum_square
        
    eps = np.finfo(outsum_envelope.dtype).tiny
    res = np.zeros_like(outsum_envelope)
    out_mask = (outsum_envelope > eps)
    res[out_mask] = outsum[out_mask] / outsum_envelope[out_mask]
    
    return res

@numba.njit(parallel=True, fastmath=True)
def _inner_with_abs_numba(x: NDArray[complex], y: NDArray[complex]) -> NDArray[float]:
    n_ch, n_ts = x.shape
    res = np.zeros((n_ch, n_ch), dtype=x.real.dtype)
    
    for i in numba.prange(n_ch):
        for j in range(n_ch):
            for k in range(n_ts):
                imag = (x[i, k]*y[j, k]).imag
                res[i,j] += np.abs(imag)
    
    return res

@numba.njit(parallel=True, fastmath=True)
def _inner_with_square_numba(x: NDArray[complex], y: NDArray[complex]) -> NDArray[float]:
    n_ch, n_ts = x.shape
    res = np.zeros((n_ch, n_ch), dtype=x.real.dtype)
    
    for i in numba.prange(n_ch):
        for j in range(n_ch):
            for k in range(n_ts):
                imag = (x[i, k]*y[j, k]).imag
                res[i,j] += np.square(imag)
    
    return res
