import numpy as np
import numpy.typing 

from typing import Union, Any, TypeVar, TypeAlias
T = TypeVar("DType", bound=np.generic)

try:    
    import cupy as cp

    HAS_CUPY = True

    NDArray: TypeAlias = np.typing.NDArray[T] | cp.typing.NDArray[T]
except:
    HAS_CUPY = False

    NDArray: TypeAlias = np.typing.NDArray[T]

def get_module(arr: NDArray[Any], force_gpu: bool=False):
    if force_gpu:
        if HAS_CUPY:
            xp = cp
        else:
            raise RuntimeError('"force_gpu=True" while cupy is not installed')
    else:
        if HAS_CUPY:
            xp = cp.get_array_module(arr)
        else:
            xp = np
    
    return xp

def _add_support_field(*, field_name: str):
    """Decorator to mark function support for GPU and/or multiprocessing.
    
    Args:
        field_name: The name of the field to add to the function
        
    Returns:
        Decorated function with appropriate support flags
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        
        setattr(wrapper, field_name, True)
        return wrapper
    return decorator

def supports_gpu(func):
    return _add_support_field(field_name='gpu')(func)

def supports_multiprocessing(func):
    return _add_support_field(field_name='multiprocessing')(func)