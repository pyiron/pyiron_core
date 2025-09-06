"""
For mathematical operations.
"""

from typing import Optional

import numpy as np

from pyiron_core.pyiron_workflow import as_function_node


@as_function_node("linspace")
def Linspace(
    x_min: float = 0,
    x_max: float = 1,
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


@as_function_node
def Sum(
    x: list | np.ndarray,
    axis: Optional[int] = None,
    keepdims: bool = False,
):
    """
    Calculate the sum of elements along a given axis in an array or a list.
    """
    sum = np.sum(x, axis=axis, keepdims=keepdims)
    return sum


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


@as_function_node
def Mean(numbers: list | np.ndarray | float | int):
    """
    Calculate the mean of a list or numpy array of numbers.
    """
    mean = np.mean(numbers)
    return mean


@as_function_node
def Array(data, dtype: Optional[str] = None):
    """
    Convert input data to a numpy array.

    Parameters:
        data (list or np.ndarray): Input data to be converted.

    Returns:
        np.ndarray: Numpy array representation of the input data.
    """

    array = np.asarray(data)
    if dtype is not None:
        array = np.asarray(data, dtype=dtype)
    return array


@as_function_node
def Reshape(data: np.ndarray, new_shape: str = "(1, -1)"):
    """
    Reshape a numpy array to a new shape.

    Parameters:
        data (np.ndarray): Input data to be reshaped.
        new_shape (tuple[int, ...]): New shape for the array.

    Returns:
        np.ndarray: Reshaped numpy array.
    """
    if isinstance(new_shape, str):
        # Convert string representation of shape to tuple
        new_shape = tuple(int(dim) for dim in new_shape.strip("()").split(","))

    reshaped_data = np.reshape(data, new_shape)
    return reshaped_data


@as_function_node
def WeightedHistogram(
    data,
    bins: int = 50,
    weighting: str = "linear",
    bin_centers: Optional[np.ndarray] = None,
):
    """
    Compute a weighted histogram for a list of floats using linear, quadratic, or cubic weights
    to share a value between multiple nearest bins.

    Parameters:
        data (list or np.ndarray): Input data (list of floats).
        bins (int): Number of bins for the histogram (ignored if bin_centers is provided).
        weighting (str): Type of weighting ("linear", "quadratic", "cubic").
        bin_centers (np.ndarray, optional): Predefined bin centers. If None, bin centers are computed.

    Returns:
        bin_centers (np.ndarray): Centers of the histogram bins.
        weighted_histogram (np.ndarray): Weighted histogram values.
    """
    # Compute bin centers if not provided
    if bin_centers is None:
        bin_edges = np.linspace(min(data), max(data), bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Initialize the weighted histogram
    weighted_histogram = np.zeros_like(bin_centers)

    # Determine the number of bins to consider based on weighting type
    if weighting == "linear":
        num_bins_to_consider = 2
    elif weighting == "quadratic":
        num_bins_to_consider = 3
    elif weighting == "cubic":
        num_bins_to_consider = 4
    else:
        raise ValueError(
            "Invalid weighting type. Use 'linear', 'quadratic', or 'cubic'."
        )

    # Apply weighting for each data point
    for x in data:
        # Find the nearest bins
        distances = np.abs(bin_centers - x)
        nearest_bins = np.argsort(distances)[
            :num_bins_to_consider
        ]  # Indices of the nearest bins

        # Compute weights based on distance to the bin centers
        total_distance = np.sum(distances[nearest_bins])
        if weighting == "linear":
            weights = 1 - distances[nearest_bins] / total_distance
        elif weighting == "quadratic":
            weights = (1 - distances[nearest_bins] / total_distance) ** 2
        elif weighting == "cubic":
            weights = (1 - distances[nearest_bins] / total_distance) ** 3

        # Normalize weights so their sum is 1
        weights /= np.sum(weights)

        # Add weighted contributions to the histogram
        for i, bin_idx in enumerate(nearest_bins):
            weighted_histogram[bin_idx] += weights[i]

    return weighted_histogram, bin_centers


@as_function_node
def LinearBin(data, bin_centers):
    """
    Smoothly bins data points over neighboring bin centers using linear interpolation.

    Parameters:
        data (array-like): The data points to bin.
        bin_centers (array-like): The centers of the bins.

    Returns:
        np.ndarray: Array of binned counts, linear spline interpolated.
    """
    data = np.asarray(data)
    bin_centers = np.asarray(bin_centers)
    counts = np.zeros_like(bin_centers, dtype=float)

    x_min = np.min(bin_centers)
    dx = np.mean(np.diff(bin_centers))
    steps = len(bin_centers)

    for x0 in data:
        i_left = int((x0 - x_min) / dx)

        # ensure i_left is within bounds
        if i_left < 0:
            continue
        elif i_left >= steps - 1:
            continue

        w = (x0 - bin_centers[i_left]) / dx
        counts[i_left] += 1 - w
        counts[i_left + 1] += w

    return counts, bin_centers


def b_spline_basis(i, k, t, x):
    """
    Compute the B-spline basis function N_i,k at x.

    Parameters:
    - i: index of the basis function
    - k: degree of the spline (order = k + 1)
    - t: knot vector (non-decreasing sequence)
    - x: position(s) to evaluate the basis function

    Returns:
    - values of N_i,k(x)
    """
    if k == 0:
        # zero degree basis function
        return np.where((x >= t[i]) & (x < t[i + 1]), 1.0, 0.0)
    else:
        denom1 = t[i + k] - t[i]
        denom2 = t[i + k + 1] - t[i + 1]

        term1 = 0
        if denom1 != 0:
            term1 = (x - t[i]) / denom1 * b_spline_basis(i, k - 1, t, x)

        term2 = 0
        if denom2 != 0:
            term2 = (t[i + k + 1] - x) / denom2 * b_spline_basis(i + 1, k - 1, t, x)

        return term1 + term2


def b_spline_basis_derivative(i, k, t, x):
    if k == 0:
        # The derivative of degree 0 basis is zero everywhere
        return np.zeros_like(x)
    else:
        denom1 = t[i + k] - t[i]
        denom2 = t[i + k + 1] - t[i + 1]

        term1 = 0
        if denom1 != 0:
            term1 = k / denom1 * b_spline_basis(i, k - 1, t, x)

        term2 = 0
        if denom2 != 0:
            term2 = k / denom2 * b_spline_basis(i + 1, k - 1, t, x)

        return term2 - term1


@as_function_node
def BSpline(
    x0_vals: np.ndarray,
    x_min: float = 0,
    x_max: float = 1,
    steps: int = 10,
    degree: int = 3,
):
    """
    Create a B-spline descriptor for the given range and degree.

    Parameters:
    - x0_vals: array of x values where the B-spline basis will be evaluated
    - x_min: minimum x value
    - x_max: maximum x value
    - degree: degree of the B-spline

    Returns:
    - A function that evaluates the B-spline basis at given x0 values.
    """
    import numpy as np

    knots, dx = np.linspace(
        x_min - degree * 0, x_max + degree * 0, steps, retstep=True
    )  # Create a uniform knot vector
    x_vals = np.linspace(x_min, x_max, steps)
    y_vals = np.zeros(steps)
    y_vals_derivative = np.zeros(steps)

    # Evaluate the B-spline basis at each x0 value
    x_shift = degree * dx / 2 + 3 / 2 * dx
    for x0 in x0_vals:
        y_vals += b_spline_basis(
            1, degree, knots, x_vals - x0 + np.min(knots) + x_shift
        )
        y_vals_derivative += b_spline_basis_derivative(
            1, degree, knots, x_vals - x0 + np.min(knots) + x_shift
        )

    return y_vals, y_vals_derivative, x_vals
