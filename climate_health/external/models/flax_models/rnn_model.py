import flax.linen as nn
import jax.numpy as jnp



class RNNModel(nn.Module):
    n_hidden: int = 4
    pre_hidden: int = 4
    n_locations: int = 1
    embedding_dim: int = 4
    output_dim: int = 1

    @nn.compact
    def __call__(self, x, training=False):
        dropout_rate = 0.2
        loc = nn.Embed(num_embeddings=self.n_locations, features=self.embedding_dim)(jnp.arange(self.n_locations))
        loc = jnp.repeat(loc[:, None, :], x.shape[1], axis=1)
        x = jnp.concatenate([x, loc], axis=-1) # batch x embedding_dim
        layers = [4]
        for i in range(len(layers)):
            x = nn.Dense(features=layers[i])(x)
            x = nn.relu(x)
        x = nn.Dropout(rate=dropout_rate, deterministic=not training)(x)
        x = nn.Dense(features=self.pre_hidden)(x)
        x = nn.Dropout(rate=dropout_rate, deterministic=not training)(x)
        gru = nn.SimpleCell(features=self.n_hidden)
        x = nn.RNN(gru)(x)
        x = nn.Dense(features=6)(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.output_dim)(x)
        return x




class ARModel(nn.Module):
    n_locations: int = 1
    prediction_length: int = 3

    def __call__(self, x, training=False):
        rnn = RNNModel(n_locations=self.n_locations)

        x = rnn(x[:, :-self.prediction_length], training=training)


