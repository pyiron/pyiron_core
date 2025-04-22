"""
For mathematical operations.
"""

from __future__ import annotations

import numpy as np

from pyiron_workflow import as_function_node


@as_function_node("linspace")
def Linspace(
    x_min: float = 0,
    x_max: float = 0,
    num_points: int = 50,
    endpoint: bool = True,
):
    return np.linspace(x_min, x_max, num_points, endpoint=endpoint)


@as_function_node("arange")
def Arange(
    start: int = 0,
    stop: int = 0,
    step: int = 1,
):
    return np.arange(start, stop, step)


@as_function_node("range")
def Range(
    start: int = 0,
    stop: int = 0,
    step: int = 1,
):
    return list(range(start, stop, step))


@as_function_node("sin")
def Sin(x: list | np.ndarray | float | int = 0):
    return np.sin(x)


@as_function_node("cos")
def Cos(x: list | np.ndarray | float | int):
    return np.cos(x)


@as_function_node("tan")
def Tan(x: list | np.ndarray | float | int):
    return np.tan(x)


@as_function_node("arcsin")
def Arcsin(x: list | np.ndarray | float | int):
    arcsin = np.arcsin(x)
    return arcsin


@as_function_node("arccos")
def Arccos(x: list | np.ndarray | float | int):
    return np.arccos(x)


@as_function_node("arctan")
def Arctan(x: list | np.ndarray | float | int):
    return np.arctan(x)


@as_function_node("arctan2")
def Arctan2(x: list | np.ndarray | float | int, y: list | np.ndarray | float | int):
    return np.arctan2(y, x)


@as_function_node("divide")
def Divide(x: list | np.ndarray | float | int, y: list | np.ndarray | float | int):
    return np.divide(x, y)


@as_function_node("add")
def Add(x: list | np.ndarray | float | int, y: list | np.ndarray | float | int):
    return np.add(x, y)


@as_function_node("subtract")
def Subtract(x: list | np.ndarray | float | int, y: list | np.ndarray | float | int):
    return np.subtract(x, y)


@as_function_node("multiply")
def npMultiply(x: list | np.ndarray | float | int, y: list | np.ndarray | float | int):
    return np.multiply(x, y)


@as_function_node("multiply")
def Multiply(x: any, y: any):
    return x * y
