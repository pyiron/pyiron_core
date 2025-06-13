"""
For graphical representations of data.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from pyiron_workflow import as_function_node


@as_function_node("fig")
def PlotDataFrame(df: pd.DataFrame, x: Optional[list | np.ndarray] = None):
    from matplotlib import pyplot as plt

    fig, ax = plt.subplots()
    plot = df.plot(x=x, ax=ax)
    return plt.show()


@as_function_node("fig")
def PlotDataFrameXY(df: pd.DataFrame, x: Optional[list | np.ndarray] = None):
    from matplotlib import pyplot as plt

    # Check if dataframe has only two columns and x parameter is not provided.
    if df.shape[1] == 2 and x is None:
        columns = df.columns
        x = columns[0]  # First column for x-axis.
        y = columns[1]  # Second column for y-axis.
        x_label, y_label = x, y
    else:
        x_label = x if isinstance(x, str) else "x label not defined"
        y_label = "y label not defined"

    fig, ax = plt.subplots()
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    df.plot(x=x, y=y, ax=ax)

    return plt.show()


@as_function_node("fig")
def Scatter(
    x: Optional[list | np.ndarray] = None, y: Optional[list | np.ndarray] = None
):
    from matplotlib import pyplot as plt

    plt.scatter(x, y)
    return plt.show()


@as_function_node("fig")
def LinearFittingCurve(
    x: Optional[list | np.ndarray] = None, y: Optional[list | np.ndarray] = None
):
    from matplotlib import pyplot as plt
    import numpy as np

    rms = np.sqrt(np.var(x - y))
    print(f"RMS: {rms}")
    x_ideal = np.linspace(min(x), max(x), 100)
    y_ideal = np.poly1d(np.polyfit(x, y, 1))(x_ideal)
    plt.plot(x_ideal, x_ideal, "--", label="Ideal")
    plt.plot(x_ideal, y_ideal, label="Fitted")
    plt.scatter(x, y)
    plt.legend
    return plt.show()


@as_function_node("fig")
def ShowArray(mat: Optional[np.ndarray], aspect_ratio: float=None):
    from matplotlib import pyplot as plt

    if aspect_ratio is not None:   
        plt.imshow(mat, aspect=aspect_ratio)
    plt.imshow(mat)
    return plt.show()


@as_function_node("fig")
def Histogram(x: Optional[list | np.ndarray], bins: int = 50):
    from matplotlib import pyplot as plt

    plt.hist(x, bins=bins)
    return plt.show()


@as_function_node("figure")
def Plot(
    y: Optional[list | np.ndarray | pd.core.series.Series],
    x: Optional[list | np.ndarray | pd.core.series.Series] = None,
    axis: Optional[object] = None,
    title: Optional[str] = "",
    color: Optional[str] = "b",
    symbol: Optional[str] = "o",
    legend_label: Optional[str] = "",
    log_x: bool = False,
    log_y: bool = False,

):
    from matplotlib import pyplot as plt

    # If x is not provided, generate a default sequence
    x = np.arange(len(y)) if x is None else x

    if axis is None:
        axis = plt
        axis.title = title
        axis.plot(x, y, color=color, marker=symbol, label=legend_label)
        if log_x:
            axis.xscale("log")
        if log_y:
            axis.yscale("log")
        figure = axis.show()

    else:
        axis.set_title(title)  # Set the title of the plot
        axis.plot(x, y, color=color, marker=symbol, label=legend_label)
        figure = axis

    return figure


@as_function_node("linspace")
def Linspace(
    start: Optional[int | float] = 0.0,
    stop: Optional[int | float] = 1.0,
    num: Optional[int] = 50,
    endpoint: Optional[bool] = True,
):
    from numpy import linspace

    return linspace(start, stop, num, endpoint=endpoint)


@as_function_node("mean")
def Mean(numbers: list | np.ndarray | float | int):
    return np.mean(numbers)


@as_function_node("axes")
def Subplot(
    nrows: Optional[int] = 1,
    ncols: Optional[int] = 1,
    sharex: Optional[bool] = False,
    sharey: Optional[bool] = False,
):
    from matplotlib import pyplot as plt

    _, axes = plt.subplots(nrows, ncols, sharex=sharex, sharey=sharey)
    return axes


@as_function_node("axis")
def Title(axis: Optional[object] = None, title: Optional[str] = ""):
    from matplotlib import pyplot as plt

    if axis is None:
        axis = plt

    return axis.set_title(title)
