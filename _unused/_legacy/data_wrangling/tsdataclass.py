import inspect
from typing import Literal, Union

import numpy as np

type_conversion = [(int, np.ndarray),
                   (float, np.ndarray)]


def get_all_annotations(cls):
    annotations = {}
    for base in reversed(cls.__mro__):
        annotations.update(inspect.get_annotations(base))
    return annotations


def pivot_type(cls, value_name="value", index_name="data_element"):
    annotations = get_all_annotations(cls)
    literal_type = Literal[*annotations.keys()]
    value_type = Union[*annotations.values()]
    return type(f"Pivoted{cls.__name__}", (object,),
                {"__annotations__": {index_name: literal_type, value_name: value_type}})


def tsdataclass(cls):
    annotations = get_all_annotations(cls)
    print(annotations)


def test_pivot_type():
    class MyClass:
        rainfall: float
        temperature: float
        population: int

    pivoted = pivot_type(MyClass)
    assert inspect.get_annotations(pivoted) == {"data_element": Literal["rainfall", "temperature", "population"],
                                                'value': Union[int, float]}


def test():
    class A:
        a: int
        b: float

    class B(A):
        c: int

    tsdataclass(B)
