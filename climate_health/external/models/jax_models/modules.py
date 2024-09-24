import equinox as eqx
import jax.numpy as np  # type: ignore


class LogisticTransform(eqx.Module):
    a: float
    b: float
    c: float
    d: float

    def __call__(self, x):
        return self.a / (1 + self.b * np.exp(-self.c * x)) + self.d


def test_logistic_transform():
    result = LogisticTransform(1.0, 2.0, 3.0, 4.0)(10.0)
    print(result)
