import flax.linen as nn

from climate_health.datatypes import ClimateHealthTimeSeries
from climate_health.external.models.jax_models import jax
from climate_health.spatio_temporal_data.temporal_dataclass import SpatioTemporalDict


class RNNModel(nn.Module):
    n_hidden: int = 4
    pre_hidden: int = 4

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=5)(x)
        x = nn.Dense(features=self.pre_hidden)(x)
        gru = nn.SimpleCell(features=self.n_hidden)
        #initial_carry = self.param('initial_carry', gru.initialize_carry, (1, self.n_hidden))
        x = nn.RNN(gru)(x)
        x = nn.Dense(features=4)(x)
        x = nn.Dense(features=1)(x)
        return x


