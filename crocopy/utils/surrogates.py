import numpy as np
import scipy as sp

from typing import Callable, Sequence

from .._base import NDArray, get_module
from ..observables.criticality.lrtc import compute_dfa
from ..observables.connectivity.synchrony import compute_cplv, compute_wpli


def _shuffle_phase(signal: NDArray[float]) -> NDArray[float]:
    """
        Shuffle phase of given signal.
    :param sig: 1d numpy array of signal. Should be real valued
    :return: 1d numpy array of the same type and shape as input. Has the same magnitude as original one but shuffled phase.
    """
    xp = get_module(signal)

    fs = xp.fft.fft(signal)
    mag = xp.abs(fs)
    phase = xp.angle(fs)

    xp.random.shuffle(phase)

    fs_shuffled = mag * xp.exp(1j * phase)
    fs_reconstructed = xp.fft.ifft(fs_shuffled)

    return xp.real(fs_reconstructed)

def _generate_single_channel_iaaft(signal: NDArray[float], max_iter: int=100, tol: float=1e-8, patience: int=5) -> NDArray[float]:
    xp = get_module(signal)

    n_ts = signal.shape[0]

    amplitude_target = xp.abs(xp.fft.rfft(signal))
    signal_sorted = xp.sort(signal)

    surr = xp.random.permutation(signal)
    surr_spectra = xp.fft.rfft(surr)

    eps = xp.finfo(signal.dtype).tiny
    best_err = xp.inf
    stall = 0

    norm_target = xp.linalg.norm(amplitude_target) + eps

    for _ in range(max_iter):
        phase = surr_spectra / xp.maximum(xp.abs(surr_spectra), eps)
        y = xp.fft.irfft(amplitude_target * phase, n=n_ts)

        order = xp.argsort(y)
        surr = xp.empty_like(y)
        surr[order] = signal_sorted

        surr_spectra = xp.fft.rfft(surr)

        error_normed = float(xp.linalg.norm(xp.abs(surr_spectra) - amplitude_target) / norm_target)

        if error_normed < tol:
            break

        if error_normed < best_err * (1 - 1e-12):
            best_err = error_normed
            stall = 0
        else:
            # no improvement in score
            stall += 1
            if stall >= patience:
                break

    return surr

def compute_surrogate_statistic(data: NDArray[float], method: str, observable_function: Callable | str, **kwargs) -> NDArray[float]:
    """
    Compute a surrogate statistic for a given data array using a specified method.

    Args:
        data: The data array to compute the surrogate statistic for.
        method: The method to use to compute the surrogate statistic.
        observable_function: Callable or string name of the observable to compute
            ('plv', 'wpli', or 'dfa').
        **kwargs: Additional keyword arguments to pass to the observable function.
    """

    if method == 'random_phase':
        surr_data = _create_random_phase_surrogates(data)
    elif method == 'iaaft':
        surr_data = _create_iaaft_surrogates(data)
    elif method == 'time_shift':
        surr_data = _create_time_shift_surrogates(data)
    elif method == 'noise':
        surr_data = _create_noise_surrogates(data)
    else:
        raise ValueError(f"Invalid method: {method}")

    if type(observable_function) is str:
        if observable_function == 'plv':
            observable_function = compute_cplv
        elif observable_function == 'wpli':
            observable_function = compute_wpli
        elif observable_function == 'dfa':
            observable_function = lambda x, **kw: compute_dfa(x, **kw)[2]
        else:
            raise ValueError(f"Invalid statistic: {observable_function}")

    surr_stat = observable_function(surr_data, **kwargs)

    return surr_stat

def _create_random_phase_surrogates(data: NDArray[float]) -> NDArray[float]:
    """
    Create random phase surrogates for a given data array.
    """
    xp = get_module(data)
    res = xp.empty_like(data)

    for i in range(res.shape[0]):
        res[i] = _shuffle_phase(data[i])  # was incorrectly calling itself recursively

    return res

def _create_iaaft_surrogates(data: NDArray[float]) -> NDArray[float]:
    xp = get_module(data)
    res = xp.empty_like(data)

    for i in range(res.shape[0]):
        res[i] = _generate_single_channel_iaaft(data[i])

    return res

def _create_time_shift_surrogates(data: NDArray[float], shift: int | Sequence[int]=None) -> NDArray[float]:
    """
    Create time shift surrogates for a given data array.
    """
    xp = get_module(data)
    res = data.copy()

    if (shift is None):
        shift = xp.random.randint(0, res.shape[1], size=res.shape[0])

    if (type(shift) is int):
        shift = [shift] * res.shape[0]

    for i in range(res.shape[0]):
        channel_shift = shift[i]
        res[i] = xp.roll(res[i], channel_shift)

    return res

def _create_noise_surrogates(data: NDArray[float]) -> NDArray[float]:
    """
    Create noise surrogates for a given data array.
    """
    xp = get_module(data)
    res = xp.random.normal(size=data.shape)

    return res
