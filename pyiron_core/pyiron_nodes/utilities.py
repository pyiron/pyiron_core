from pyiron_workflow import as_function_node
import numpy as np

# workflow does not understand this
# from typing import Any


@as_function_node("range")
def Range(start: int, stop: int, step: int):
    return list(range(start, stop, step))


@as_function_node("linspace")
def Linspace(start: float | int, stop: float | int, num: int) -> np.ndarray:
    return np.linspace(start, stop, num)


@as_function_node("value")
def Index(values: list, index: int):
    return values[index]


@as_function_node("slice")
def Slice(values: list, start: int = 0, stop: int = -1, step: int = 1) -> list:
    return values[start:stop:step]


@as_function_node("list")
def Prepend(x, xs: list | None = None) -> list:
    if xs is None:
        xs = []
    return [x, *xs]


@as_function_node("list")
def Append(xs: list, x) -> list:
    if xs is None:
        xs = []
    return [*xs, x]


@as_function_node("list")
def List5(x1, x2=None, x3=None, x4=None, x5=None) -> list:
    return [x for x in (x1, x2, x3, x4, x5) if x is not None]


@as_function_node("df")
def ReadDataFrame(filename: str, compression: str = None):
    import pandas as pd

    return pd.read_pickle(filename, compression=compression)


@as_function_node("item")
def GetItem(obj, index: int | str):
    return obj[index]
