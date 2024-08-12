from datetime import datetime

import numpy as np
from flax import linen as nn
import jax.numpy as jnp
import optax
from flax.training import train_state

from climate_health.datatypes import ClimateHealthTimeSeries, FullData, SummaryStatistics
import jax

from climate_health.external.models.flax_models.rnn_model import RNNModel
from climate_health.external.models.jax_models.model_spec import PoissonSkipNaN
from climate_health.spatio_temporal_data.temporal_dataclass import SpatioTemporalDict


def l2_regularization(params, scale=1.0):
    return sum(jnp.sum(jnp.square(p)) for p in jax.tree_util.tree_leaves(params)) * scale


def year_position_from_datetime(dt: datetime) -> float:
    day = dt.timetuple().tm_yday
    return day / 365

class TrainState(train_state.TrainState):
  key: jax.Array


# @register_model("flax_model")
class FlaxModel:
    model: nn.Module = RNNModel()

    def __init__(self, rng_key: jax.random.PRNGKey = jax.random.PRNGKey(100), n_iter: int = 2000):
        self.rng_key = rng_key
        self.n_iter = n_iter
        self._losses = []
        self._params = None
        self._saved_x = None
        self._validation_x = None
        self._validation_y = None
        self._model = None
        self._n_locations = None

    @property
    def model(self):
        if self._model is None:
            self._model = RNNModel(n_locations= self._saved_x.shape[0])
        return self._model

    def set_validation_data(self, data: SpatioTemporalDict[FullData]):
        x, y = self._get_series(data)
        self._validation_x = x
        self._validation_y = y

    def _get_series(self, data: SpatioTemporalDict[FullData]):
        x = []
        y = []
        for series in data.values():
            year_position = [year_position_from_datetime(period.start_timestamp.date) for period in series.time_period]
            x.append(np.array(
                (series.rainfall, series.mean_temperature, series.population, year_position)).T)  # type: ignore
            if hasattr(series, 'disease_cases'):
                y.append(series.disease_cases)

        return np.array(x), np.array(y)

    def _loss(self, y_pred, y_true):
        return jnp.mean(self.loss_func(y_pred, y_true))

    def loss_func(self, y_pred, y_true):
        return -PoissonSkipNaN(jnp.exp(y_pred.ravel())).log_prob(y_true.ravel())

    def get_validation_y(self, params):
        x = np.concatenate([self._saved_x, (self._validation_x - self._mu) / self._std], axis=1)
        y_pred = self.model.apply(params, x)
        return y_pred[:, self._saved_x.shape[1]:]

    def train(self, data: SpatioTemporalDict[ClimateHealthTimeSeries]):
        x, y = self._get_series(data)
        self._mu = np.mean(x, axis=(0, 1))
        self._std = np.std(x, axis=(0, 1))
        x = (x - self._mu) / self._std
        self._saved_x = x
        params = self.model.init(self.rng_key, x, training=False)
        print(params)
        y_pred = self.model.apply(params, x)
        assert y_pred.shape == x.shape[:-1] + (1,)
        assert np.all(np.isfinite(y_pred))
        assert np.all(~np.isnan(y_pred)), y_pred

        #val_grad = jax.value_and_grad(logprob_func)(params)

        #solver = optax.adam(learning_rate=0.003)
        #opt_state = solver.init(params)
        # f = jax.value_and_grad(logprob_func)
        dropout_key = jax.random.PRNGKey(40)

        state = TrainState.create(
            apply_fn=self.model.apply,
            params=params,
            tx=optax.adam(1e-3),
            key=dropout_key
        )

        @jax.jit
        def train_step(state, dropout_key):
            dropout_train_key = jax.random.fold_in(key=dropout_key, data=state.step)
            def loss_func(params):
                eta = state.apply_fn(params, x, training=True, rngs={'dropout': dropout_train_key})
                return self._loss(eta, y) + l2_regularization(params)
                #return self._loss(state.apply_fn({'params': params}, x, training=True, rngs={'dropout': dropout_train_key}), y) + l2_regularization(params)
            #logprob_func = lambda params: self._loss(state.apply_fn({'params': params}, x, training=True, rngs={'dropout': dropout_train_key}), y) + l2_regularization(params)
            grad_func = jax.value_and_grad(loss_func)
            loss, grad = grad_func(state.params)
            state = state.apply_gradients(grads=grad)
            return state




        for i in range(self.n_iter):
            #loss, grad = f(params)
            #updates, opt_state = solver.update(grad, opt_state)
            #params = optax.apply_updates(params, updates)
            state = train_step(state, dropout_key)
            if i % 10 == 0:
                if self._validation_x is not None:
                    val_loss = self._loss(self.get_validation_y(state.params), self._validation_y)
                    print(f"Validation Loss: {val_loss}")
                #self._losses.append(loss)
            #self._losses.append(loss)

        self._params = state.params

    def forecast(self, data: SpatioTemporalDict[FullData], n_samples, forecast_delta):
        print('Forecasting with params:', self._params)
        x, y = self._get_series(data)
        x = (x - self._mu) / self._std
        full_x = jnp.concatenate([self._saved_x, x], axis=1)
        print(full_x)
        full_y_pred = np.exp(self.model.apply(self._params, full_x))
        y_pred = full_y_pred[:, self._saved_x.shape[1]:]
        print(y_pred)
        time_period = next(iter(data.values())).time_period
        return SpatioTemporalDict(
            {key: SummaryStatistics(time_period, *[row.ravel()] * 7)
             for key, row in zip(data.keys(), y_pred)})

    def diagnose(self):
        import matplotlib.pyplot as plt
        plt.plot(self._losses)
        plt.show()

    def predict(self, data: SpatioTemporalDict[FullData]):
        x, y = self._get_series(data)
        return np.exp(self.model.apply(self._params, x))
