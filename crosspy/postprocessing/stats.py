from numpy import ndarray, percentile, squeeze
from numpy.random import randint


# this should be made more generic
# hence, 1. data should be reararndged to have axis as 0-axis
# bootstraps should be creted dynamically because data can have whatever shape
# resulting ci should be computed as 2 x original shape but the dimension alogn which we operated
def bootstrap_ci(data: ndarray, n=100, axis=0, percentiles=(2.5, 97.5))->tuple:

    bootstraps = ndarray((n, data.shape[1]), dtype=data.dtype)

    for idx in range(n):
        perm_indices = randint(0, data.shape[0], n)
        bootstraps[idx, :] = data[perm_indices ,:].mean(axis=0)

    ci = squeeze(percentile(bootstraps, percentiles, keepdims=True, axis=0))
    return ci[0, :], ci[1, :]
