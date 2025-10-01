"""
Operations and views for pandas DataFrames in Pyiron Nodes.
This module provides functions to manipulate and analyze pandas DataFrames
"""

import pandas as pd

from pyiron_core.pyiron_workflow import Node, as_function_node


@as_function_node("df")
def ReadDataFrame(filename: str, compression: str = None):
    import pandas as pd

    return pd.read_pickle(filename, compression=compression)


# get column from dataframe
@as_function_node
def GetColumnFromDataFrame(df, column_name: str, as_array: bool = False):
    import numpy as np

    column = df[column_name]
    if as_array:
        column = np.asarray(column.tolist())

    return column


# get rows from dataframe (from min to max index)
@as_function_node
def GetRowsFromDataFrame(df, min_index: int = 0, max_index: int = None):
    if max_index is None:  # return all rows from min_index to the end of the dataframe.
        max_index = len(df)

    rows = df.iloc[min_index:max_index]
    return rows


# display dataframe
@as_function_node
def DisplayDataFrame(df: pd.DataFrame) -> pd.DataFrame:
    """
    Display the DataFrame.
    """
    return df


# merge two dataframes
@as_function_node
def MergeDataFrames(
    df1: pd.DataFrame, df2: pd.DataFrame, on: str = "index", how: str = "outer"
):
    """
    Merge two DataFrames on a specified column or index.
    """
    merged_df = pd.merge(df1, df2, on=on, how=how)
    return merged_df


# Using apply to map the function on the Series
@as_function_node
def ApplyFunctionToSeries(series: pd.Series, func: Node, store: bool = False):
    """
    Apply a function to each element of a pandas Series.
    """
    import numpy as np

    kwargs = func.kwargs
    first_arg = list(kwargs.keys())[0]
    del kwargs[first_arg]
    transformed_series = np.stack(series.apply(func._func, **kwargs))

    return transformed_series


# Using apply to map the function on the Series
@as_function_node
def ApplyFunctionToSeriesNew(series: pd.Series, function: Node, store: bool = False):
    """
    Apply a function to each element of a pandas Series.
    """
    import numpy as np

    kwargs = function.kwargs
    first_arg = list(kwargs.keys())[0]
    del kwargs[first_arg]
    out = series.apply(function._func, **kwargs)
    print(f"apply function: {function._func}, out")
    transformed_series = np.stack(out)

    return transformed_series
