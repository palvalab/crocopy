import numpy as np
import cupy as cp
import cupyx
import cupyx.scipy.signal

import itertools

# # # # #
#  
# Some of the functions from cupy do not support out parameter 
# While I want to have to -> Need to copy-paste it
#  
# # # # #

def cupy_roll_with_buffer(a, shift, axis=None, out=None):
    """Roll array elements along a given axis.
    Elements that roll beyond the last position are re-introduced at the first.
    Args:
        a (~cupy.ndarray): Array to be rolled.
        shift (int or tuple of int): The number of places by which elements are
            shifted. If a tuple, then `axis` must be a tuple of the same size,
            and each of the given axes is shifted by the corresponding number.
            If an int while `axis` is a tuple of ints, then the same value is
            used for all given axes.
        axis (int or tuple of int or None): The axis along which elements are
            shifted. By default, the array is flattened before shifting, after
            which the original shape is restored.
    Returns:
        ~cupy.ndarray: Output array.
    .. seealso:: :func:`numpy.roll`
    """
    if axis is None:
        return cp.roll(a.ravel(), shift, 0).reshape(a.shape)
    else:
        broadcasted = np.broadcast(shift, axis)
        if broadcasted.nd > 1:
            raise ValueError(
                '\'shift\' and \'axis\' should be scalars or 1D sequences')
        shifts = {ax: 0 for ax in range(a.ndim)}
        for sh, ax in broadcasted:
            shifts[ax] += sh

        rolls = [((slice(None), slice(None)),)] * a.ndim
        for ax, offset in shifts.items():
            offset %= a.shape[ax] or 1  # If `a` is empty, nothing matters.
            if offset:
                # (original, result), (original, result)
                rolls[ax] = ((slice(None, -offset), slice(offset, None)),
                             (slice(-offset, None), slice(None, offset)))

        if out is None:
            result = cp.empty_like(a)
        else:
            result = out

        for indices in itertools.product(*rolls):
            arr_index, res_index = zip(*indices)
            result[res_index] = a[arr_index]

        return result


def _prod(iterable):
    """
        Stolen from cusignal.filter
    """
    product = 1
    for x in iterable:
        product *= x
    return product


def cupy_detrend(data, axis=-1, type="linear", bp=0, overwrite_data=False):
    """
        Copied from cusignal + fixed a bug with NotImplementedError with cp.r_ (swapped it to np.r_)
    """
    if type not in ["linear", "l", "constant", "c"]:
        raise ValueError("Trend type must be 'linear' or 'constant'.")
    data = cp.asarray(data)
    dtype = data.dtype.char
    if dtype not in "dfDF":
        dtype = "d"
    if type in ["constant", "c"]:
        ret = data - cp.expand_dims(cp.mean(data, axis), axis)
        return ret
    else:
        dshape = data.shape
        N = dshape[axis]
        bp = cp.sort(cp.unique(np.r_[0, bp, N]))
        if cp.any(bp > N):
            raise ValueError(
                "Breakpoints must be less than length of \
                data along given axis."
            )
        Nreg = len(bp) - 1
        # Restructure data so that axis is along first dimension and
        #  all other dimensions are collapsed into second dimension
        rnk = len(dshape)
        if axis < 0:
            axis = axis + rnk
            
        newdims = np.r_[axis, 0:axis, axis + 1 : rnk]
        newdata = cp.reshape(
            cp.transpose(data, tuple(newdims)), (N, _prod(dshape) // N)
        )
        if not overwrite_data:
            newdata = newdata.copy()  # make sure we have a copy
        if newdata.dtype.char not in "dfDF":
            newdata = newdata.astype(dtype)
        # Find leastsq fit and remove it for each piece
        for m in range(Nreg):
            Npts = int(bp[m + 1] - bp[m])
            A = cp.ones((Npts, 2), dtype)
            A[:, 0] = cp.arange(1, Npts + 1) * 1.0 / Npts
            sl = slice(bp[m], bp[m + 1])
            coef, resids, rank, s = cp.linalg.lstsq(A, newdata[sl])
            newdata[sl] = newdata[sl] - cp.dot(A, coef)
        # Put data back in original shape.
        tdshape = cp.take(cp.asarray(dshape), cp.asarray(newdims), 0)
        ret = cp.reshape(newdata, tuple(cp.asnumpy(tdshape)))
        vals = list(range(1, rnk))
        olddims = vals[:axis] + [0] + vals[axis:]
        ret = cp.transpose(ret, tuple(cp.asnumpy(olddims)))
        return ret
    
def cp_sliding_window_view(x, window_shape, axis=None, *,
                        subok=False, writeable=False):

    window_shape = (tuple(window_shape)
                    if np.iterable(window_shape)
                    else (window_shape,))

    # writeable is not supported:
    if writeable:
        raise NotImplementedError("Writeable views are not supported.")

    # first convert input to array, possibly keeping subclass
    x = cp.array(x, copy=False, subok=subok)

    window_shape_array = cp.array(window_shape)
    for dim in window_shape_array:
        if dim < 0:
            raise ValueError('`window_shape` cannot contain negative values')

    if axis is None:
        axis = tuple(range(x.ndim))
        if len(window_shape) != len(axis):
            raise ValueError(f'Since axis is `None`, must provide '
                             f'window_shape for all dimensions of `x`; '
                             f'got {len(window_shape)} window_shape elements '
                             f'and `x.ndim` is {x.ndim}.')
    else:
        axis = cp._core.internal._normalize_axis_indices(axis, x.ndim)
        if len(window_shape) != len(axis):
            raise ValueError(f'Must provide matching length window_shape and '
                             f'axis; got {len(window_shape)} window_shape '
                             f'elements and {len(axis)} axes elements.')

    out_strides = x.strides + tuple(x.strides[ax] for ax in axis)

    # note: same axis can be windowed repeatedly
    x_shape_trimmed = list(x.shape)
    for ax, dim in zip(axis, window_shape):
        if x_shape_trimmed[ax] < dim:
            raise ValueError(
                'window shape cannot be larger than input array shape')
        x_shape_trimmed[ax] -= dim - 1
    out_shape = tuple(x_shape_trimmed) + window_shape
    return cp.lib.stride_tricks.as_strided(x, strides=out_strides, shape=out_shape)


# [obsolete]
# some people use older version of cupy, so lets inject necessary functions to ensure that functionality is here
if not(hasattr(cp.lib.stride_tricks, 'sliding_window_view')):
    cp.lib.stride_tricks.sliding_window_view = cp_sliding_window_view

if not(hasattr(cupyx.scipy.signal, 'detrend')):
    cupyx.scipy.signal.detrend = cupy_detrend