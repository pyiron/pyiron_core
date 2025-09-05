"""
For graphical representations of data.
"""

from __future__ import annotations

from typing import Optional, Literal

import numpy as np
import pandas as pd

from pyiron_workflow import as_function_node


@as_function_node("fig")
def PlotDataFrame(df: pd.DataFrame, x: Optional[list | np.ndarray] = None):
    from matplotlib import pyplot as plt

    fig, ax = plt.subplots()
    df.plot(x=x, ax=ax)  # df.plot returns an Axes
    return fig


@as_function_node("fig")
def PlotDataFrameXY(df: pd.DataFrame, x: Optional[list | np.ndarray] = None):
    from matplotlib import pyplot as plt

    # Default labels in case not deduced
    x_label = "x label not defined"
    y_label = "y label not defined"

    # Check if dataframe has only two columns and x parameter is not provided.
    if df.shape[1] == 2 and x is None:
        columns = df.columns
        x = columns[0]  # First column for x-axis.
        y = columns[1]  # Second column for y-axis.
        x_label, y_label = x, y
    else:
        y = None
        if isinstance(x, str):
            x_label = x

    fig, ax = plt.subplots()
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    df.plot(x=x, y=y, ax=ax)
    return fig


@as_function_node("fig")
def Scatter(
    x: Optional[list | np.ndarray] = None, y: Optional[list | np.ndarray] = None
):
    from matplotlib import pyplot as plt

    fig, ax = plt.subplots()
    ax.scatter(x, y)
    return fig


@as_function_node("fig")
def LinearFittingCurve(
    x: Optional[list | np.ndarray] = None, y: Optional[list | np.ndarray] = None
):
    from matplotlib import pyplot as plt
    import numpy as np

    rms = np.sqrt(np.var(x - y))
    correlation_coefficient = np.corrcoef(x, y)[0, 1]
    print(f"Correlation Coefficient: {correlation_coefficient}")
    print(f"RMS: {rms}")

    x_ideal = np.linspace(min(x), max(x), 100)
    y_ideal = np.poly1d(np.polyfit(x, y, 1))(x_ideal)

    fig, ax = plt.subplots()
    ax.plot(x_ideal, x_ideal, "--", label="Ideal")
    ax.plot(x_ideal, y_ideal, label="Fitted")
    ax.scatter(x, y)
    ax.legend()
    return fig


@as_function_node("fig")
def ShowArray(mat: Optional[np.ndarray], aspect_ratio: float = None):
    from matplotlib import pyplot as plt

    fig, ax = plt.subplots()
    if aspect_ratio is not None:
        ax.imshow(mat, aspect=aspect_ratio)
    else:
        ax.imshow(mat)
    return fig


@as_function_node("fig")
def Histogram(x: Optional[list | np.ndarray], bins: int = 50):
    from matplotlib import pyplot as plt

    fig, ax = plt.subplots()
    ax.hist(x, bins=bins)
    return fig


@as_function_node("figure")
def Plot(
    y: Optional[list | np.ndarray | pd.core.series.Series],
    x: Optional[list | np.ndarray | pd.core.series.Series] = None,
    axis: Optional[object] = None,
    title: Optional[str] = "",
    color: Literal[
        "b", "g", "r", "c", "m", "y", "k", "w",
        "blue", "green", "red", "cyan", "magenta", "yellow", "black", "white"
    ] = "b",
    symbol: Literal[
        ".", ",", "o", "v", "^", "<", ">", "1", "2", "3", "4",
        "s", "p", "*", "h", "H", "+", "x", "X", "D", "d", "|", "_"
    ] = "o",
    legend_label: Optional[str] = "",
    log_x: bool = False,
    log_y: bool = False,
):
    import numpy as np
    import pandas as pd
    from matplotlib import pyplot as plt

    # If x is not provided, generate a default sequence
    x = np.arange(len(y)) if x is None else x

    if axis is None:
        fig, ax = plt.subplots()
    else:
        # assume axis is an Axes object passed in
        ax = axis
        fig = ax.figure

    # Plot data
    ax.plot(x, y, color=color, marker=symbol, label=legend_label)

    # Log scales if needed
    if log_x:
        ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")

    # Set title if provided
    if title:
        ax.set_title(title)

    # Add legend if label provided
    if legend_label:
        ax.legend()

    # Return the Figure so caller can display it
    return fig


@as_function_node("figure")
def MultiPlot(
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
    import numpy as np
    from matplotlib import pyplot as plt

    if axis is None:
        fig, ax = plt.subplots()
    else:
        ax = axis
        fig = ax.figure

    # Handle multi-series plotting
    if isinstance(y, (list, tuple)) and not isinstance(y, np.ndarray):
        for i, yy in enumerate(y):
            xx = (
                np.arange(len(yy))
                if x is None
                else (x[i] if isinstance(x, (list, tuple)) else x)
            )
            ax.plot(xx, yy, color=color, marker=symbol, label=legend_label)
    else:
        xx = np.arange(len(y)) if x is None else x
        ax.plot(xx, y, color=color, marker=symbol, label=legend_label)

    if log_x:
        ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")

    if title:
        ax.set_title(title)

    if legend_label:
        ax.legend()

    return fig


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
    import numpy as np

    return np.mean(numbers)


@as_function_node("axes")
def Subplot(
    nrows: Optional[int] = 1,
    ncols: Optional[int] = 1,
    sharex: Optional[bool] = False,
    sharey: Optional[bool] = False,
):
    from matplotlib import pyplot as plt

    fig, axes = plt.subplots(nrows, ncols, sharex=sharex, sharey=sharey)
    # Return both figure and axes so caller can decide which to use
    return axes


@as_function_node("axis")
def Title(axis: Optional[object] = None, title: Optional[str] = ""):
    from matplotlib import pyplot as plt

    if axis is None:
        # if None, create a new figure+axes for standalone title
        fig, ax = plt.subplots()
    else:
        ax = axis
        fig = ax.figure
    ax.set_title(title)
    return ax
