import numpy as np
import scipy as sp

from ..observables.connectivity.synchrony import compute_cplv

class SurrogateMixerSamePair(object):
    def __init__(self, data, metric=compute_cplv):
        self.data = data / np.abs(data).mean(axis=1).reshape((-1, 1))
        self.metric = metric

    def get_pair(self, target_metric, pair_index=None):
        if pair_index is None:
            pair_index = self._generate_random_indexes()

        optim_res = sp.optimize.minimize_scalar(self._optimize_mixing, bounds=(0.0, 1.0), method='Bounded',
                                                args=(
                                                self.data[pair_index[0]], self.data[pair_index[1]], target_metric))

        return self._mix_channels(optim_res.x, self.data[pair_index[0]], self.data[pair_index[1]])

    def _generate_random_indexes(self):
        x_idx = np.random.randint(self.data.shape[0])
        y_idx = np.random.randint(self.data.shape[0])

        while x_idx == y_idx:
            y_idx = np.random.randint(self.data.shape[0])

        return x_idx, y_idx

    def _optimize_mixing(self, mc, x_ch, y_ch, target_metric):
        x_mix, y_mix = self._mix_channels(mc, x_ch, y_ch)
        mix_metric = self.metric(x_mix, y_mix)

        return (mix_metric - target_metric) ** 2

    def _mix_channels(self, mc, x_ch, y_ch):
        x_mix = x_ch + mc * y_ch
        y_mix = y_ch + mc * x_ch

        return x_mix, y_mix
