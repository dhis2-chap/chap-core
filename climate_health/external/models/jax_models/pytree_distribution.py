from jax import Array

from .jax import jax


class IsPyTreeDistribution:
    def sample(self, shape: tuple[int]) -> jax.PyTree:

    def log_prob(self, value: jax.PyTree) -> float | Array[float]:
        ...

def test_tree_distribution():
