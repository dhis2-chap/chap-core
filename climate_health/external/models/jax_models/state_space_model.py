import logging

import jax
import numpy as np
from matplotlib import pyplot as plt
from .hmc import sample as nuts_sample
from scipy.special import logit

from climate_health.datatypes import ClimateHealthTimeSeries, HealthData, ClimateData


class SimpleSampler:
    def __init__(self, key, log_prob: callable, sample_func: callable, param_names: list[str], n_states=None, n_warmup_samples=1000, n_samples=100):
        self._log_prob = log_prob
        self._param_names = param_names
        self._param_samples = None
        self._sample_key = key
        self._sample_func = sample_func
        self._n_states = n_states
        self._n_warmup_samples = n_warmup_samples
        self._n_samples = n_samples

    @classmethod
    def from_model(cls, model, key, *args, **kwargs) -> 'SimpleSampler':
        return cls(key, model.lp_func, model.sampler, model.param_names, model.n_states, *args, **kwargs)

    @property
    def n_samples(self):
        return len(self._param_samples[list(self._param_samples)[0]])

    def train(self, time_series: ClimateHealthTimeSeries, init_values=None) -> HealthData:
        T = len(time_series) + 1
        if self._n_states is None:
            init_diffs = np.random.normal(0, 1, (T - 1))
        else:
            init_diffs = np.random.normal(logit(0.05), 1, (T - 1, self._n_states)).reshape((T - 1, self._n_states))
        init_params = {param_name: np.random.normal(0, 10) for param_name in self._param_names}
        init_dict = {'logits_array': init_diffs} | init_params
        if init_values is not None:
            for key, value in init_values.items():
                init_dict[key] = value

        lp = self._log_prob(time_series.disease_cases, time_series.mean_temperature)
        self._sample_key, key = jax.random.split(self._sample_key)
        self._param_samples = nuts_sample(lp, key, init_dict, self._n_samples, self._n_warmup_samples)
        last_params = {param_name: self._param_samples[param_name][-1] for param_name in self._param_samples}
        last_grad = jax.grad(lp)(last_params)
        if any(np.any(np.isnan(value)) for value in last_grad.values()):
            lp(last_params)
            logging.warning('Nans in gradient')
        np.savez('params.npz', **self._param_samples)

    def sample(self, climate_data: ClimateData) -> HealthData:
        T = len(climate_data) + 1
        self._sample_key, key1, key2 = jax.random.split(self._sample_key, 3)

        param_nr = jax.random.randint(key2, (1,), 0, self.n_samples)[0]
        params = {param_name: self._param_samples[param_name][param_nr] for param_name in self._param_names}
        return self._sample_func(T, key1, params, climate_data.mean_temperature)

    def diagnostic_plot(self, real_params):
        plt.subplots(nrows=len(self._param_names), ncols=1)
        for i, param_name in enumerate(self._param_names):
            plt.subplot(len(self._param_names), 1, i + 1)
            plt.plot(self._param_samples[param_name], label='sampled')
            plt.axhline(real_params[param_name], color='red', label='real')
            plt.title(param_name)
            plt.legend()
        return plt.gcf()


