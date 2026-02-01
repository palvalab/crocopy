import numpy as np

import copy

from joblib import Parallel

import tqdm

from typing import Optional, Callable, Any, Sequence

from ..observables._base import supports_gpu, get_module, NDArray, HAS_CUPY
from ..preprocessing.signal import filter_data
from ..observables.criticality.lrtc import compute_dfa
from ..observables.criticality.fei import compute_fei
from ..observables.criticality.bistability import compute_BiS_em

if HAS_CUPY:
    import cupy as cp

def _try_interfere_return_rank(function: Callable) -> int:
    try:
        return_arg = function.__annotations__['return'] # has return typing at all

        if("Array" in str(return_arg)): # follows nptyping
            return len(return_arg.__args__[0].prepared_args)
        else:
            return None
    except:
        return None

def _try_interfere_return_type(function: Callable) -> int:
    try:
        return_arg = function.__annotations__['return'] # has return typing at all

        if("Array" in str(return_arg)): # follows nptyping
            return return_arg.__args__[1]
        else:
            return None
    except:
        return None

def _convert_size_to_samples(window_size: int, window_type: str, frequency: float, sampling_rate: int) -> float:
    match window_type:
        case 'samples':
            return window_size
        case 'seconds':
            return int(window_size*sampling_rate)
        case 'cycles':
            return int(sampling_rate/frequency*window_size)
        case _:
            raise RuntimeError(f'Uknown window type: {window_type}')
        
@supports_gpu
def dfa_wrapper(data: NDArray[complex], **kwargs) -> NDArray[float]:
    xp = get_module(data)

    data_envelope = xp.abs(data)
    return compute_dfa(data_envelope, **kwargs).dfa_values

@supports_gpu
def bis_wrapper(data: NDArray[complex], *args, **kwargs) -> NDArray[float]:
    xp = get_module(data)

    data_power = xp.abs(data)**2
    data_power /= data_power.max(axis=-1, keepdims=True)
    
    return compute_BiS_em(data_power, *args, **kwargs)

@supports_gpu
def fei_wrapper(data: NDArray[complex], *args, **kwargs) -> NDArray[float]:
    xp = get_module(data)

    data_envelope = xp.abs(data)
    return compute_fei(data_envelope, *args, **kwargs)

def _compute_windowed_function_raw(data: NDArray[complex], function: Callable, window_size: int, window_step: int, function_rank: int, function_dtype: np.dtype, function_arguments: dict) -> NDArray[Any]:
    xp = get_module(data)

    n_chans = data.shape[0]

    data_windowed = xp.lib.stride_tricks.sliding_window_view(data, (n_chans, window_size))[0, ::window_step]

    res_shape = [data_windowed.shape[0]] + [n_chans]*function_rank
    res = xp.zeros(res_shape, dtype=function_dtype)

    for window_idx, window_data in enumerate(data_windowed):
        res[window_idx] = xp.array(function(window_data, **function_arguments), copy=False)

    return res

def _prepare_arguments_for_frequency(frequency: float, sampling_rate: int, **kwargs) -> dict:
    res = copy.deepcopy(kwargs)

    if ('window_lengths' in res):
        min_len, max_len_samples = res['window_lengths']
        min_len_samples = _convert_size_to_samples(min_len, 'cycles', frequency, sampling_rate)

        res['window_lengths'] = np.geomspace(min_len_samples, max_len_samples, 40)

    if ('window_size' in res):
        window_size_samples = _convert_size_to_samples(res['window_size'], 'cycles', frequency, sampling_rate)
        res['window_size'] = window_size_samples
    
    return res


def _interfere_function_rank_dtype(data: NDArray[float], function: Callable) -> Sequence[int, np.dtype]:
     # simulate "filtering", we dont care about an actual results, just need some  non-empty data
    data_dummy = np.random.uniform(size=(10, 10000)).astype(complex)
    dummy_function_output = function(data_dummy)

    func_rank = len(dummy_function_output.shape)
    func_dtype = dummy_function_output.dtype

    return func_rank, func_dtype


def compute_windowed_function(data: NDArray[float], function: Callable, sampling_rate: int, frequencies: Sequence[float],
                               window_size: int, window_step: Optional[int]=None, window_type: str='seconds', 
                               omega: float=7.5,
                               function_rank: Optional[int]=None, function_dtype: Optional[np.dtype]=None,
                               function_arguments=dict(),
                               use_tqdm: bool=True) -> NDArray[Any]:
    """

    
    """
    if (function_rank is None) or (function_dtype is None):
        function_rank, function_dtype = _interfere_function_rank_dtype(data, function) 

    if (function_rank is None):
        raise RuntimeError('You must provide a rank of the function!')
    
    if (function_dtype is None):
        raise RuntimeError('You must provide a return dtype of the function !')

    if (window_step is None):
        window_step = window_size

    if HAS_CUPY:
        n_filt_jobs = 'cuda'
    else:
        n_filt_jobs = 16

    res = list()

    for frequency_idx, frequency in enumerate(tqdm.tqdm(frequencies, leave=False, disable=not(use_tqdm))):
        data_filt = filter_data(data, sfreq=sampling_rate, frequency=frequency, omega=omega, n_jobs=n_filt_jobs)

        if not(type(data_filt) is np.ndarray):
            data_filt = data_filt.get()
        
        frequency_window_size = _convert_size_to_samples(window_size, window_type, frequency, sampling_rate)
        frequency_window_step = _convert_size_to_samples(window_step, window_type, frequency, sampling_rate)

        prepared_function_kwargs = _prepare_arguments_for_frequency(frequency=frequency, sampling_rate=sampling_rate, **function_arguments)

        if hasattr(function, 'supports_multiprocessing') and (type(data_filt) is np.ndarray):
            prepared_function_kwargs['context'] = Parallel(n_jobs=function_arguments['n_jobs'], backend='loky')

        frequency_res = _compute_windowed_function_raw(data_filt, function, 
                                                       frequency_window_size, frequency_window_step, 
                                                       function_rank, function_dtype, function_arguments=prepared_function_kwargs)
        
        if not(type(data_filt) is np.ndarray):
            del data_filt
            mempool = cp.get_default_memory_pool()
            mempool.free_all_blocks()

            frequency_res = frequency_res.get()

        res.append(frequency_res)

    if window_type in ['seconds', 'samples']:
        res = np.array(res)
    
    return res