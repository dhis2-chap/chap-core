# from chap_core.external.models.jax_models.deterministic_seir_model import PydanticTree
# from chap_core.time_period import dataclasses
# from .jax import jnp
#
#
# def get_categorical_transform(cls: object) -> object:
#     new_fields = [(field.name, float, jnp.log(field.default))
#                   for field in dataclasses.fields(cls)]
#     new_class = dataclasses.make_dataclass('T_' + cls.__name__, new_fields, bases=(PydanticTree,), frozen=True)
#
#     def f(x: cls) -> new_class:
#         values = x.tree_flatten()[0]
#         return new_class.tree_unflatten(None, [jnp.log(value) for value in values])
#         return new_class(*[(jnp.log(value)) for value in values])
#
#     def inv_f(x: new_class) -> cls:
#         values = x.tree_flatten()
#         new_values = [jnp.exp(value) for value in values]
#         s = sum(new_values)
#         return cls.tree_unflatten(None, [value / s for value in new_values])
#
#     return new_class, f, inv_f
