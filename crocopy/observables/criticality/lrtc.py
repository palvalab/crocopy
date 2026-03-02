import numpy as np

import statsmodels.api as sm

import functools

from collections import namedtuple
from typing import Tuple, Sequence

from ...observables._base import NDArray, get_module
from ...utils.stats import rsquared

DFAResultType = namedtuple('DFAResult', ['fluctuation', 'r_squared', 'dfa_values', 'intercept_values'])

class TukeyWeightedNorm(sm.robust.norms.TukeyBiweight):
    ''' 
    Custom Norm Class using Tukey's biweight (bisquare).
    '''
    
    def __init__(self, weights: NDArray[float], c: float=4.685, **kwargs):
        super().__init__(**kwargs)
        self.weights_vector = np.array(weights)
        self.flag = 0
        self.c = c
        
    def set_weights(self, weights: NDArray[float]):
        self.weights_vector = weights
        self.flag = 0

    def weights(self, z: NDArray[float]) -> NDArray[float]:
        """
            Instead of weights equal to one return custom
        INPUT:
            z : 1D array
        OUTPUT:
            weights: ndarray
        """
        if self.flag == 0:
            self.flag = 1
            return self.weights_vector.copy()
        else:
            subset = self._subset(z)
            return (1 - (z / self.c)**2)**2 * subset

def _fit_dfa_exponent(window_lengths: NDArray[float], fluct: NDArray[float], weighting: str, N_samp: int, fitting: str='Tukey', min_valid_fraction: float=0.75) -> Tuple[NDArray[float], NDArray[float], NDArray[float]]:
    """
    Fit DFA exponents across channels using linear or robust regression.

    Parameters
    ----------
    window_lengths : NDArray[float]
        Window lengths in samples.
    fluct : NDArray[float]
        Fluctuation function values of shape (n_channels, n_windows).
    weighting : {'sq1ox', '1ox'}
        Weighting scheme for regression.
    N_samp : int
        Number of samples in the original time series.
    fitting : {'Tukey', 'weighted', 'linfit'}, optional
        Regression method.
    min_valid_fraction:
        Minimum fraction of valid (not nan) windows

    Returns
    -------
    NDArray[float]
        DFA exponents for each channel.
    NDArray[float]
        R-squared values for each channel.
    """
    match weighting:
        case 'sq1ox':
            sigma = np.sqrt(window_lengths/N_samp)
        case '1ox':
            sigma = window_lengths/N_samp
        case _:
            raise RuntimeError(f'Weighting {weighting} is not available!')

    match fitting:
        case 'Tukey':
            orig_weights = sigma
            model = functools.partial(sm.RLM, M=TukeyWeightedNorm(weights=orig_weights, c=4.685))
        case 'weighted':
            orig_weights = 1.0/(sigma**2)
            model = functools.partial(sm.WLS, weights=orig_weights)
        case 'linfit':
            model = functools.partial(sm.OLS)
        case _:
            raise RuntimeError(f'Fitting {fitting} is not available!')
        
    n_ch = fluct.shape[0]

    fluct_log = np.log2(fluct)
    x = sm.tools.add_constant(np.log2(window_lengths))

    dfa_values = np.full(n_ch, np.nan)
    r_squared_values = np.full(n_ch, np.nan)    
    intercept_values = np.full(n_ch, np.nan)

    for chan_idx, chan_fluct in enumerate(fluct_log):
        mask = np.isfinite(chan_fluct)

        if mask.mean() < min_valid_fraction:
            continue

        if fitting == 'Tukey':
            model.keywords['M'].set_weights(orig_weights[mask])
        elif fitting == 'weighted':
            model.keywords['weights'] = orig_weights[mask]

        mdl_fit = model(chan_fluct[mask], x[mask]).fit()

        dfa_values[chan_idx] = mdl_fit.params[1]
        intercept_values[chan_idx] = mdl_fit.params[0]

        if hasattr(mdl_fit, 'rsquared'):
            r_squared_values[chan_idx] = mdl_fit.rsquared
        else:
            predicted = mdl_fit.predict(x)
            chan_rs = rsquared(chan_fluct[mask], predicted[mask])
            r_squared_values[chan_idx] = chan_rs

    return dfa_values, r_squared_values, intercept_values

def _calc_rms(x: NDArray[float], window_size: int, max_nan_frac: float=0.2, overlap: float=0.0) -> NDArray[float]:
    """
    Windowed RMS with linear detrending, vectorized over windows and signals,
    NaN-aware, and supporting window overlap.

    Parameters
    ----------
    x : 2D array (NumPy or CuPy)
        Integrated signals, shape (n_signals, N).
    window_size : int
        Window length in samples.
    max_nan_frac : float, optional
        Maximum allowed fraction of NaNs in a window. Windows with a higher
        fraction are treated as invalid (RMS -> NaN).
    overlap : float in [0, 1], optional
        Fraction of overlap between consecutive windows:
            0.0 -> non-overlapping windows (step = scale)
            1.0 -> sliding windows (step = 1)

    Returns
    -------
    rms : xp.ndarray
        RMS values per window, shape (n_signals, n_windows).
        Invalid windows are filled with NaN.
    """
    xp = get_module(x)
    x = xp.asarray(x)

    n_signals, N = x.shape

    step = max(1, int(round(window_size * (1.0 - overlap))))
    n_windows = 1 + (N - window_size) // step

    win_offsets = xp.arange(window_size)[None]
    start_idxs = xp.arange(n_windows) * step
    idx = start_idxs[:, None] + win_offsets

    K = n_signals * n_windows
    X_flat = x[:, idx].reshape(K, window_size)

    valid_mask = ~xp.isnan(X_flat)
    valid_counts = valid_mask.sum(axis=1).astype(xp.float64) 

    min_points = max(2, int((1.0 - max_nan_frac) * window_size))
    valid_enough = valid_counts >= min_points

    scale_ax = xp.arange(window_size, dtype=x.dtype)

    # For sums, treat NaNs as 0 and use mask for counts
    weights_valid = valid_mask.astype(xp.float64)  
    Y = xp.where(valid_mask, X_flat, 0.0)     

    # regression sums
    # sum over y 
    sum_y  = Y.sum(axis=1)       
    # sum over x                           
    sum_x  = (weights_valid * scale_ax[None]).sum(axis=1)  
    # sum over x & y  
    sum_xy = (Y * scale_ax[None]).sum(axis=1)    
    # sum over x ^ 2
    sum_x2 = (weights_valid * (scale_ax[None]**2)).sum(axis=1) 

    den = valid_counts * sum_x2 - sum_x**2
    good_reg = (den != 0)
    good = (valid_enough & good_reg)

    # Initialize RMS as NaN for all windows
    rms_flat = xp.full(K, xp.nan, dtype=x.dtype)

    if int(good.sum()) > 0:
        M_good = valid_mask[good]
        n_good = valid_counts[good]
        sum_y_good  = sum_y[good]
        sum_x_good  = sum_x[good]
        sum_xy_good = sum_xy[good]
        den_good    = den[good]

        # regression coefficients: y = a + b x
        b = (n_good * sum_xy_good - sum_x_good * sum_y_good) / den_good
        a = (sum_y_good - b * sum_x_good) / n_good

        yfit_good = a[:, None] + b[:, None] * scale_ax[None, :]

        residuals_good = xp.where(M_good, X_flat[good] - yfit_good, 0.0)
        sum_res2_good = (residuals_good**2).sum(axis=1)

        rms_good = xp.sqrt(sum_res2_good / n_good)
        rms_flat[good] = rms_good


    rms = rms_flat.reshape(n_signals, n_windows)
    return rms

def _compute_dfa_rms(data, win_lengths, max_nan_frac=0.2, overlap=0.5):
    """
    DFA fluctuation function for multiple signals (rows), NumPy/CuPy, NaN-aware,
    with overlapping windows.

    Parameters
    ----------
    data : 2D array (NumPy or CuPy)
        Amplitude time series, shape (n_signals, N).
        Each row is one signal.
    win_lengths : 1D array-like
        Window lengths in samples.
    max_nan_frac : float, optional
        Maximum allowed fraction of NaNs per window. Windows with more NaNs
        do not contribute (their RMS is NaN).
    overlap : float in [0, 1], optional
        Fraction of overlap between consecutive windows.

    Returns
    -------
    fluct : xp.ndarray
        Fluctuation function F(s) per signal:
        shape (n_signals, n_scales).
    slope : xp.ndarray
        Slope
    """
    xp = get_module(data)
    data = xp.asarray(data)

    if not(0 <= overlap < 1):
        raise ValueError(f"DFA: overlap argument must be in [0, 1), it is {overlap} now")

    if not(0 <= max_nan_frac < 1):
        raise ValueError(f"DFA: max_nan_frac argument must be in [0, 1), it is {max_nan_frac} now")

    n_signals, N = data.shape
    n_windows = len(win_lengths)

    # demean data
    data -= xp.nanmean(data, axis=-1, keepdims=True)

    # prevent NaNs from breaking cumsum: treat them as 0 increments,
    # then put NaNs back after the cumsum
    nan_mask = xp.isnan(data)
    y = xp.nancumsum(data, axis=-1)
    y = xp.where(nan_mask, xp.nan, y)

    fluct = xp.zeros((n_signals, n_windows), dtype=data.dtype)

    for window_idx, window_size in enumerate(win_lengths):
        window_size = int(window_size)
        if window_size <= 0 or window_size > N:
            fluct[..., window_idx] = xp.nan
            continue

        rms = _calc_rms(y, window_size=window_size, max_nan_frac=max_nan_frac, overlap=overlap)

        # DFA fluctuation: sqrt( mean(rms^2 over windows))
        rms_sq = rms**2
        fluct[..., window_idx] = xp.sqrt(xp.nanmean(rms_sq, axis=1))

    return fluct


def _compute_dfa_fft(data_orig: NDArray[float], win_lengths: Sequence[int]) -> Tuple[NDArray[float], NDArray[float]]:
    """
    Compute DFA using FFT-based method (Nolte 2019 Sci Rep).

    Parameters
    ----------
    data_orig : NDArray[float]
        Amplitude time series, shape (n_channels, n_samples).
    win_lengths : Sequence[int]
        Window lengths in samples.

    Returns
    -------
    NDArray[float]
        Fluctuation function values.
    NDArray[float]
        Slopes.
    """
    xp = get_module(data_orig)

    data = xp.asarray(data_orig).copy()
    win_arr = xp.asarray(win_lengths)
    
    data -= data.mean(axis=-1, keepdims=True)
    data_fft = xp.fft.fft(data)

    n_chans, n_ts = data.shape
    is_odd = n_ts % 2 == 1

    nx = (n_ts + 1)//2 if is_odd else n_ts//2 + 1
    data_power = 2*xp.abs(data_fft[:, 1:nx])**2

    # fir for Niquist limit
    if is_odd == False:
        data_power[:,~0] /= 2
        
    ff = xp.arange(1, nx)
    g_sin = xp.sin(xp.pi*ff/n_ts)
    
    hsin = xp.sin(xp.pi*xp.outer(win_arr, ff)/n_ts)
    hcos = xp.cos(xp.pi*xp.outer(win_arr, ff)/n_ts)

    hx = 1 - hsin/xp.outer(win_arr, g_sin)
    h = (hx / (2*g_sin.reshape(1, -1)))**2

    f2 = xp.inner(data_power, h)

    fluct = xp.sqrt(f2)/n_ts

    return fluct

def compute_dfa(data: NDArray[float | complex], window_lengths: Sequence[int], method: str='fft', min_valid_windows_fraction: float=0.75,
            force_gpu: bool=False, fitting: str='Tukey', weighting: str='sq1ox', **method_kwargs) -> DFAResultType:    
    """
    Compute DFA with windowed RMS or FFT-based method.

    Parameters
    ----------
    data : NDArray[float | complex]
        2D array of shape (n_channels, n_samples). If complex, amplitude is used.
    window_lengths : Sequence[int]
        Window lengths in samples.
    method : {'fft', 'rms'}, optional
        DFA method to use.
    force_gpu : bool, optional
        If True, input np.array is converted to CuPy in the function.
    fitting : {'linfit', 'Tukey', 'weighted'}, optional
        Regression method for exponent fitting.
    weighting : {'sq1ox', '1ox'}, optional
        Weighting scheme for regression.
    **method_kwargs
        Additional keyword arguments passed to the DFA method.

    Returns
    -------
    DFAResultType
        Named tuple with fields: fluctuation, r_squared, dfa_values, residuals.
    """

    xp = get_module(data, force_gpu)

    if xp.iscomplexobj(data):
        data = xp.abs(data)
    
    allowed_methods = ('fft','rms' )
    if not(method in allowed_methods):
        raise ValueError('Method {} is not allowed! Only {} are available'.format(method, ','.join(allowed_methods)))

    allowed_weightings = ('sq1ox', '1ox')
    if not(weighting in allowed_weightings):
        raise ValueError('Weighting {} is not allowed! Only {} are available'.format(weighting, ','.join(allowed_weightings)))

    if method == 'rms':
        fluct =  _compute_dfa_rms(data, window_lengths, **method_kwargs)
    elif method == 'fft':
        fluct =  _compute_dfa_fft(data, window_lengths, **method_kwargs)
        
    if not(xp is np):
        fluct = xp.asnumpy(fluct)
    
    dfa_values, r_squared_values, intercept_values = _fit_dfa_exponent(window_lengths, fluct, weighting=weighting, N_samp=data.shape[-1], 
                                                                        min_valid_fraction=min_valid_windows_fraction, fitting=fitting)

    res = DFAResultType(fluctuation=fluct, r_squared=r_squared_values, dfa_values=dfa_values, intercept_values=intercept_values)

    return res
