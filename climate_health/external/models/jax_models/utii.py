import dataclasses

import numpy as np

from climate_health.external.models.jax_models.model_spec import Normal, LogNormal
from climate_health.external.models.jax_models.protoype_annotated_spec import Positive
from .jax import jnp, jax, tree_util

state_or_param = lambda f: tree_util.register_pytree_node_class(dataclasses.dataclass(f, frozen=True))


def index_tree(tree, index):
    if isinstance(tree, jax.Array):
        return tree[index]
    elif hasattr(tree, 'items'):
        return {key: index_tree(value, index) for key, value in tree.items()}
    elif isinstance(tree, PydanticTree):
        return tree.__class__.tree_unflatten(None, (index_tree(value, index) for value in tree.tree_flatten()[0]))
    elif isinstance(tree, tuple):
        return tuple(index_tree(value, index) for value in tree)
    else:
        return index_tree(next(iter(tree)), index)


def get_normal_prior(field):
    if field.type == float:
        mu = 0
        if field.default is not None:
            mu = field.default
        return Normal(mu, 10)
    if field.type == Positive:
        mu = 1
        if field.default is not None:
            mu = field.default
        return LogNormal(np.log(mu), 10)


def tree_sample(tree, key, shape=()):
    if isinstance(tree, dict):
        keys = jax.random.split(key, len(tree))
        return {name: tree_sample(obj, key, shape=shape) for key, (name, obj) in zip(keys, tree.items())}
    else:
        return tree.sample(key, shape=shape)


class PydanticTree:
    def tree_flatten(self):
        obj = self
        ret = tuple(getattr(obj, field.name) for field in dataclasses.fields(obj))
        # ret = ({field.name: getattr(obj, field.name) for field in dataclasses.fields(obj)}, None)
        return ret, None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

    def sample(self, key, shape=()):
        obj = self
        d = {field.name: getattr(obj, field.name) for field in dataclasses.fields(obj)}

        return self.__class__(
            **{name: obj.sample(key, shape=shape) if hasattr(obj, 'sample') else obj for key, (name, obj) in
               zip(jax.random.split(key, len(d)), d.items())})

    def log_prob(self, value: 'PydanticTree'):
        return sum(getattr(self, field.name).log_prob(getattr(value, field.name))
                   for field in dataclasses.fields(self)
                   if hasattr(getattr(self, field.name), 'log_prob'))


def log_prob(dist, value):
    if isinstance(dist, tuple):
        return sum(log_prob(d, v) for d, v in zip(dist, value))
    return dist.log_prob(value)


def get_state_transform(params):
    if isinstance(params, tuple):
        T, f, inv_f = zip(*[get_state_transform(p) for p in params])
        return tuple(T), lambda x: tuple(f_i(x_i) for f_i, x_i in zip(f, x)), lambda x: tuple(
            inv_f_i(x_i) for inv_f_i, x_i in zip(inv_f, x))

    new_fields = []

    converters = []
    inv_converters = []
    identity = lambda x: x
    for field in dataclasses.fields(params):
        if field.type == Positive:
            converters.append(jnp.exp)
            inv_converters.append(jnp.log)
            default = Normal(np.log(field.default), 1.)
        elif field.type == np.ndarray:
            converters.append(identity)
            inv_converters.append(identity)
            default = Normal(np.array(field.default), 10., ndim=1)
        elif issubclass(field.type, PydanticTree):
            T, f, inv_f = get_state_transform(field.type)
            converters.append(f)
            default = T()
        else:
            converters.append(identity)
            inv_converters.append(identity)
            default = Normal(field.default, 10.)
        new_fields.append((field.name, float, default))
    new_class = dataclasses.make_dataclass('T_' + params.__name__, new_fields, bases=(PydanticTree,), frozen=True)
    tree_util.register_pytree_node_class(new_class)

    def f(transformed: new_class) -> params:
        return params.tree_unflatten(None, tuple(
            converter(val) for converter, val in
            zip(converters, transformed.tree_flatten()[0])))

    def inv_f(params: params) -> new_class:
        inv_flat = tuple(inv_converter(val) for inv_converter, val in zip(inv_converters, params.tree_flatten()[0]))
        return new_class.tree_unflatten(None, inv_flat)

    return new_class, f, inv_f
