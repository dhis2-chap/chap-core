import flax.linen as nn
import jax.numpy as jnp
from climate_health.datatypes import ClimateHealthTimeSeries
from climate_health.external.models.jax_models import jax
from climate_health.spatio_temporal_data.temporal_dataclass import SpatioTemporalDict


class RNNModel(nn.Module):
    n_hidden: int = 4
    pre_hidden: int = 4
    n_locations: int = 1
    embedding_dim: int = 3

    @nn.compact
    def __call__(self, x, training=False):
        loc = nn.Embed(num_embeddings=self.n_locations, features=self.embedding_dim)(jnp.arange(self.n_locations))
        loc = jnp.repeat(loc[:, None, :], x.shape[1], axis=1)
        x = jnp.concatenate([x, loc], axis=-1) # batch x embedding_dim
        x = nn.Dense(features=4)(x)
        x = nn.Dense(features=self.pre_hidden)(x)
        x = nn.Dropout(rate=0.5, deterministic=not training)(x)
        gru = nn.SimpleCell(features=self.n_hidden)
        #initial_carry = self.param('initial_carry', gru.initialize_carry, (1, self.n_hidden))
        x = nn.RNN(gru)(x)
        x = nn.Dense(features=4)(x)
        x = nn.Dense(features=1)(x)
        return x


