"""
For graphical representations of data.
"""

from __future__ import annotations

from typing import Literal, Optional

import numpy as np
import pandas as pd

from pyiron_core.pyiron_workflow import as_function_node


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
    import numpy as np
    from matplotlib import pyplot as plt

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
        "b",
        "g",
        "r",
        "c",
        "m",
        "y",
        "k",
        "w",
        "blue",
        "green",
        "red",
        "cyan",
        "magenta",
        "yellow",
        "black",
        "white",
    ] = "b",
    symbol: Literal[
        ".",
        ",",
        "o",
        "v",
        "^",
        "<",
        ">",
        "1",
        "2",
        "3",
        "4",
        "s",
        "p",
        "*",
        "h",
        "H",
        "+",
        "x",
        "X",
        "D",
        "d",
        "|",
        "_",
    ] = "o",
    legend_label: Optional[str] = "",
    log_x: bool = False,
    log_y: bool = False,
):
    import numpy as np
    from matplotlib import pyplot as plt

    print("plotting: ", len(np.shape(y)))
    if len(np.shape(y)) == 1:
        # If x is not provided, generate a default sequence
        x = np.arange(len(y)) if x is None else x
    else:
        x = None

    if axis is None:
        fig, ax = plt.subplots()
    else:
        # assume axis is an Axes object passed in
        ax = axis
        fig = ax.figure

    # Plot data
    if x is None:
        ax.plot(y, color=color, marker=symbol, label=legend_label)
    else:
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
    ax.set_title(title)
    return ax


@as_function_node
def AnalyseConvergencePlot(
    y: "np.ndarray",
    x: Optional["np.ndarray"] = None,
    i_start: int = 0,
    smoothen: bool = False,
    window_size: int = 10,
    mode: Literal[
        "reflect", "mirror", "constant", "nearest", "wrap", "grid-constant"
    ] = "nearest",
    ref_to_min: bool = False,
):
    """
    Plot the convergence of a scalar series ``y`` (e.g. energy, force norm)
    as a function of an optional abscissa ``x`` (iteration number, time,
    etc.).  The function is intended to be used as a pyiron node, therefore
    it returns a single Matplotlib ``Figure`` object.

    Parameters
    ----------
    y : np.ndarray
        1‑D array containing the quantity to be analysed.
    x : np.ndarray | None, optional
        Optional abscissa.  If ``None`` a simple integer index
        ``np.arange(len(y))`` is used.
    i_start : int, default ``0``
        Index from which the plot should start – useful to discard the
        initial equilibration part of a trajectory.
    smoothen : bool, default ``False``
        If ``True`` a simple moving‑average (window = 5) is applied to ``y``
        before plotting.  This helps visualise noisy data.
    ref_to_min : bool, default ``False``
        If ``True`` the plotted values are shifted by the minimum value in the
        displayed range (i.e. ``y - min(y)``).  This is handy when one wants to
        see how quickly the series approaches its lowest value.

    Returns
    -------
    Figure
        Matplotlib figure containing the convergence plot.
    """
    # ------------------------------------------------------------------
    # Imports – kept inside the node so that the node can be imported
    # without pulling heavy optional dependencies.
    # ------------------------------------------------------------------
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.ndimage import uniform_filter1d

    # ------------------------------------------------------------------
    # Convert inputs to NumPy arrays and handle the optional x‑axis.
    # ------------------------------------------------------------------
    y = np.asarray(y).ravel()
    if x is None:
        x = np.arange(y.size)
    else:
        x = np.asarray(x).ravel()

    # ------------------------------------------------------------------
    # Slice the data from the requested start index.
    # ------------------------------------------------------------------
    y_plot = y[i_start:]
    x_plot = x[i_start:]

    # ------------------------------------------------------------------
    # Optional smoothing (simple moving average with a fixed window).
    # ------------------------------------------------------------------
    if smoothen:
        y_plot = uniform_filter1d(y_plot, size=window_size, mode=mode)

    # ------------------------------------------------------------------
    # Optional reference to the minimum value.
    # ------------------------------------------------------------------
    if ref_to_min:
        min_val = np.min(y_plot)
        y_plot = y_plot - min_val

    # ------------------------------------------------------------------
    # Build the figure.
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(4.5, 3))
    ax.plot(x_plot, y_plot, marker="o", linestyle="-", color="tab:blue")
    ax.set_xlabel("Iteration" if x is None else "x")
    ylabel = "y"
    if ref_to_min:
        ylabel += " – min"
    ax.set_ylabel(ylabel)
    ax.set_title("Convergence")
    ax.grid(True, which="both", ls=":", alpha=0.7)

    # Highlight the final value with a horizontal line.
    ax.axhline(
        y_plot[-1], color="tab:red", ls="--", lw=1, label=f"final = {y_plot[-1]:.3e}"
    )
    ax.legend(fontsize="small", loc="best")

    return fig
