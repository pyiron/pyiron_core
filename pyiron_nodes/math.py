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


@as_function_node("shape")
def Shape(x: list | np.ndarray | float | int):
    """
    Get the shape of an array or a list.
    """
    return np.shape(x) if isinstance(x, (list, np.ndarray)) else None


@as_function_node
def SVD(x: np.ndarray, full_matrices: bool = False):
    """
    Perform Singular Value Decomposition (SVD) on a 2D array.
    """
    if not isinstance(x, np.ndarray) and x.ndim == 2:
        raise ValueError("Input must be a 2D numpy array.")

    u, s, vh = np.linalg.svd(x, full_matrices=full_matrices)
    return u, s, vh


@as_function_node
def SVDComponents(
    matrix: np.ndarray, full_matrices: bool = False, i_min: int = 0, i_max: int = None
):
    """
    Perform Singular Value Decomposition (SVD) and return specified components.

    Parameters:
    - matrix: 2D numpy array to decompose.
    - full_matrices: If True, U and Vh are of shape (M, M) and (N, N), respectively.
    - i_min: Minimum index of the singular values to return.
    - i_max: Maximum index of the singular values to return (exclusive).

    Returns:
    - Tuple containing U, S, Vh matrices with specified singular values.
    """
    u, s, vh = np.linalg.svd(matrix, full_matrices=full_matrices)
    if i_max is None:
        i_max = len(s)

    svd_mat = np.zeros(matrix.shape)
    norm_list = []
    for i in range(i_min, i_max):
        svd_mat += np.outer(u.T[i], vh[i]) * s[i]
        norm_list.append(np.linalg.norm(svd_mat - matrix))

    return svd_mat, norm_list


@as_function_node("array")
def PseudoInverse(matrix: np.ndarray, rcond: float = 1e-15, hermitian: bool = False):
    return np.linalg.pinv(matrix, rcond, hermitian)


@as_function_node("array")
def DotProduct(a: np.ndarray, b: np.ndarray, store: bool = False):
    return np.dot(a, b)


@as_function_node("array")
def Transpose(a: np.ndarray):
    return np.transpose(a)


@as_function_node("result")
def aAddBC(a, b: float, c):
    return a + b * c


@as_function_node
def Identity(x: float):
    return x
