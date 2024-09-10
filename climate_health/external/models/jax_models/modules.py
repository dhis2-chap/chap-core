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
    result = LogisticTransform(1., 2., 3., 4.)(10.)
    print(result)
