import functools

from typing import Optional, Tuple

import numpy as np
import statsmodels.api as sm

from scipy.stats import rv_continuous

from ..observables._base import NDArray, get_module

def moving_average_fast(x: NDArray, window_size: int, 
                      res_buf: Optional[NDArray] = None, 
                      cumsum_buf: Optional[NDArray] = None) -> NDArray:
    """
    Compute moving average using cumulative sum for efficiency.
    
    Args:
        x: Input array of shape [..., n_samples]
        window_size: Size of the moving average window
        res_buf: Optional pre-allocated buffer for result
        cumsum_buf: Optional pre-allocated buffer for cumulative sum
    
    Returns:
        NDArray: Moving average of same shape as input
        
    Raises:
        ValueError: If window_size <= 0 or window_size > array length
    """
    xp = get_module(x)
    
    if window_size <= 0:
        raise ValueError("Window size must be positive")
    if window_size > x.shape[-1]:
        raise ValueError("Window size cannot be larger than array length")
        
    # Handle trivial case
    if window_size == 1:
        return x.copy() if res_buf is None else xp.copyto(res_buf, x)

    is_odd = (window_size % 2) == 1
    w_half = window_size // 2

    # Prepare output buffer
    if res_buf is None:
        res = xp.empty_like(x)
    else:
        res = res_buf

    # Prepare cumsum buffer
    if cumsum_buf is None:
        x_cumsum = xp.empty_like(x)
    else:
        x_cumsum = cumsum_buf

    # Compute cumulative sum
    xp.cumsum(x, axis=-1, out=x_cumsum)

    # Central part using vectorized operation
    start = w_half + (0 if is_odd else 1)
    end = -w_half + (1 if is_odd else 0)
    xp.subtract(x_cumsum[..., window_size:], 
               x_cumsum[..., :-window_size], 
               out=res[..., start:end])

    # Handle head (first w_half points)
    res[..., :w_half+1] = x_cumsum[..., :w_half+1]

    # Handle tail (last w_half points)
    if is_odd:
        xp.subtract(x_cumsum[..., -1:], 
                   x_cumsum[..., -window_size:-1], 
                   out=res[..., -w_half:])
    else:
        xp.subtract(x_cumsum[..., -1:], 
                   x_cumsum[..., -(window_size+1):-1], 
                   out=res[..., -w_half:])

    # Normalize
    # res /= xp.minimum(xp.arange(1, x.shape[-1] + 1), window_size).reshape((1,) * (x.ndim-1) + (-1))
    res /= window_size
    
    return res

def icc(ratings: NDArray[float], model: str="oneway", icc_type: str="agreement") -> float:
    """
        Intraclass correlation metric.
    :param ratings: should be number array of shape N_judges X n_ratings (e.g. n_channels x n_timestamps)
    :param model: either oneway or twoway
    :param icc_type: either "consistency" or "agreement"
    :return: float value of correlation
    """
    possible_models = ("oneway", "twoway")
    possible_types = ("consistency", "agreement")

    if not (model in possible_models):
        raise RuntimeError("Model should be one of the types: {}".format(possible_models))

    if not (icc_type in possible_types):
        raise RuntimeError("Type should be one of the values: {}".format(possible_types))

    nr, ns = ratings.shape

    SStotal = np.var(ratings, ddof=1) * (ns * nr - 1)
    MSr = np.var(np.mean(ratings, axis=0), ddof=1) * nr
    MSw = np.sum(np.var(ratings, axis=0, ddof=1) / ns)
    MSc = np.var(np.mean(ratings, axis=1), ddof=1) * ns
    MSe = (SStotal - MSr * (ns - 1) - MSc * (nr - 1)) / ((ns - 1) * (nr - 1))

    if model == "oneway":
        return (MSr - MSw) / (MSr + (nr - 1) * MSw)
    else:
        if icc_type == "consistency":
            return (MSr - MSe) / (MSr + (nr - 1) * MSe)
        else:
            return (MSr - MSe) / (MSr + (nr - 1) * MSe + (nr / ns) * (MSc - MSe))

###
# Taken from SO
# https://stackoverflow.com/questions/70179307/why-is-sklearn-r-squared-different-from-that-of-statsmodels-when-fit-intercept-f 
###
def rsquared(y_true, y_pred, fit_intercept=True):
    """
    Compute R-squared consistent with statsmodels definitions.

    Parameters
    ----------
    y_true : array-like
        True response values.
    y_pred : array-like
        Predicted response values.
    fit_intercept : bool, optional
        If True, use the centered formulation.

    Returns
    -------
    float
        R-squared value.
    """
    if fit_intercept:
        return 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)
    else:
        return 1 - np.sum((y_true - y_pred) ** 2) / np.sum(y_true ** 2)



def cfc_sig_test(CFC: NDArray[float], CFC_surr: NDArray[float], z: float) -> Tuple[NDArray[float], NDArray[float], NDArray[float]]:
    """ 
    Finds significant interactions for local and inter-areal CFC. 
    An interaction CFC_ij is deemed significant if CFC_ij > mean(CFC_surr) * z.
    
    Args:
        CFC: Numpy array, shape [channels x channels] 
             CFC interaction matrix where local CFC is on the diagonal.
        CFC_surr: Surrogate CFC interaction matrix.
        z: z-value to use for determining significance.
    
    Returns:
        tuple containing:
            - CFC_sig: Matrix containing all significant inter-areal connections, diagonal is 0.
            - CFC_local: 1D Array containing all significant local CFC interactions.
            - CFC_sig_comb: Matrix containing all significant inter-areal and local interactions.    
    """
    n_ch = len(CFC)
    CFC_local = np.diag(CFC)
    CFC_surr_local = np.diag(CFC_surr)
    
    # Create mask for non-diagonal elements
    dummy1 = CFC_surr.copy()
    np.fill_diagonal(dummy1, np.nan)
    
    # Calculate significant connections
    CFC_sig = CFC * (CFC > z * np.nanmean(dummy1))
    np.fill_diagonal(CFC_sig, 0)
    
    CFC_sig_local = CFC_local * (CFC_local > z * np.nanmean(CFC_surr_local))
    
    # Combine local and inter-areal significant connections
    CFC_sig_comb = CFC_sig.copy()
    np.fill_diagonal(CFC_sig_comb, CFC_sig_local)
    
    return CFC_sig, CFC_sig_local, CFC_sig_comb


def cfc_spurious_correction(CF_sig: NDArray[float], PS_sig_LF: NDArray[float], PS_sig_HF: NDArray[float]) -> NDArray[bool]:
    """
    Correct for spurious cross-frequency coupling.
    
    Args:
        CF_sig: Numpy array of size [N_ch x N_ch]
               The interaction matrix of significant CFC (including local CFC).
        PS_sig_LF: Numpy ndarray of size [N_ch x N_ch]
                  The interaction matrix of significant PS at LF.
        PS_sig_HF: Numpy ndarray of size [N_ch x N_ch]
                  The interaction matrix of significant PS at HF.
    
    Returns:
        NDArray[bool]: The matrix of spurious-corrected significant CFC.
    """
    N_ch = len(CF_sig)
    data_sig_local = CF_sig.diagonal()
    
    HF_local = np.tile(data_sig_local, (N_ch, 1))
    LF_local = HF_local.T
    
    tri_mask = ((PS_sig_LF == 0) + (HF_local == 0)) * ((PS_sig_HF == 0) + (LF_local == 0))
    np.fill_diagonal(tri_mask, 0)
    
    return tri_mask * CF_sig
