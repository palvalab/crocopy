import numpy as np

from ._base import NDArray

def compute_instantaneous_frequency(data: NDArray[complex], sampling_rate: float, lag: int=1) -> NDArray[float]:
    """
        Computes instantaneous frequency of given data.
    :param data: 2d matrix of analog (complex) values. Should have N_chans x N_time_stamps size
    :param sampling_rate: sampling rate of the data
    :return: 2d matrix of instantaneous frequency. Has N_chans x N_time_stamps - 2 if method is 'slope',
             N_chans x N_time_stamps - 1 otherwise
    """

    return sampling_rate*np.angle(data[..., lag:]*np.conj(data[...,:-lag]))/(2*np.pi*lag)

