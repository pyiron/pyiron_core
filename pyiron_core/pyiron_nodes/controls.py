from concurrent.futures import as_completed
from copy import copy
from typing import Any, List, Union

import numpy as np
import pandas as pd

from pyiron_core.pyiron_workflow import Node, as_function_node


@as_function_node
def recursive(x: int, stop_at: int = 10) -> tuple[int, bool]:
    """Toy example for a recursive function."""
    x_new = x + 1

    break_condition = False
    if x_new > stop_at:
        break_condition = True
    return x_new, break_condition


@as_function_node
def loop_until(recursive_function: Node, max_steps: int = 10):
    x = recursive_function.inputs.x.value
    for i in range(max_steps):
        x, break_condition = recursive_function(x)
        print("loop: ", i, x, break_condition)

        if break_condition:
            break

    return x


def _iterate_node(
    node,
    input_label: str,
    values,
    copy_results=True,
    collect_input=False,
    debug=False,
    executor=None,
):
    out_lst = []
    inp_lst = [] if collect_input else None

    if executor is None:
        # Sequential execution
        for value in values:
            node.inputs.__setattr__(input_label, value)
            # TODO: provide more elaborate options
            try:
                out = node.run()
            except Exception as e:
                print("execution error: ", e)
                continue
            if copy_results:
                out = copy(out)
            out_lst.append(out)
            if collect_input:
                inp_lst.append(value)
            if debug:
                print(f"iterating over {input_label} = {value}, out={out}")
                print("out list: ", [id(o) for o in out_lst])
    else:
        # Parallel execution
        futures = {
            executor.submit(node, **{input_label: value}): (idx, value)
            for idx, value in enumerate(values)
        }
        # Placeholder, to restore original order after as_completed
        results = [None] * len(values)
        for future in as_completed(futures):
            idx, val = futures[future]
            out = future.result()
            if copy_results:
                out = copy(out)
            results[idx] = out
            if debug:
                print(f"Parallel iter: {input_label}={val}, out={out}")
        out_lst = results
        if collect_input:
            inp_lst = list(values)

    return (out_lst, inp_lst) if collect_input else out_lst


# --- Node iteration to DataFrame ---
@as_function_node
def IterToDataFrame(
    node: Node,
    input_label: str,
    values: Union[list, np.ndarray],
    debug: bool = False,
    executor=None,
    store: bool = False,
) -> pd.DataFrame:
    """
    Iterate over ``values`` feeding each element into ``node`` under the name
    ``input_label`` and collect the results in a pandas DataFrame.

    New feature:
        – If the node returns a *dataclass* instance, each field of the
          dataclass becomes its own column in the DataFrame.

    Parameters
    ----------
    node : Node
        The node that will be executed for each value.
    input_label : str
        Name of the input attribute on ``node`` that receives each element of
        ``values``.
    values : list | np.ndarray
        Iterable of input values.
    debug : bool, optional
        Print debugging information.
    executor : concurrent.futures.Executor, optional
        If supplied, the iteration runs in parallel using the executor.
    store : bool, optional
        Used by decorator to implement hash storage (if set to True)

    Returns
    -------
    pd.DataFrame
        DataFrame where each column corresponds to an input or an output
        field.  When the node returns a dataclass, each field of the
        dataclass is a separate column.
    """
    from dataclasses import fields, is_dataclass

    # ------------------------------------------------------------------
    # 1️⃣ Run the node over all values
    # ------------------------------------------------------------------
    out_lst, inp_lst = _iterate_node(
        node,
        input_label,
        values,
        copy_results=True,
        collect_input=True,
        debug=debug,
        executor=executor,
    )

    # ------------------------------------------------------------------
    # 2️⃣ Prepare a dict that will be fed to pd.DataFrame
    # ------------------------------------------------------------------
    data_dict: dict[str, List[Any]] = {}

    # ------------------------------------------------------------------
    # 2.1 Input column – avoid name clash with node outputs
    # ------------------------------------------------------------------
    output_labels = list(node.outputs.keys())
    if input_label in output_labels:
        data_dict[f"input_{input_label}"] = inp_lst
    else:
        data_dict[input_label] = inp_lst

    # ------------------------------------------------------------------
    # 3️⃣ Analyse the first output to decide how to unpack the rest
    # ------------------------------------------------------------------
    first_out = out_lst[0] if out_lst else None

    # Helper: is the result a dataclass instance?
    def _is_dataclass_instance(obj: Any) -> bool:
        return is_dataclass(obj) and not isinstance(obj, type)

    # ------------------------------------------------------------------
    # 3.1 Tuple / list / np.ndarray output (multiple scalar outputs)
    # ------------------------------------------------------------------
    multi_output = isinstance(first_out, (tuple, list, np.ndarray)) and len(
        first_out
    ) == len(output_labels)

    # ------------------------------------------------------------------
    # 3.2 Dataclass output – each field becomes a column
    # ------------------------------------------------------------------
    if _is_dataclass_instance(first_out):
        # Extract field names once – they will be the column names
        dc_fields = [f.name for f in fields(first_out)]

        # Initialise a list for each field
        for f_name in dc_fields:
            data_dict[f_name] = []

        # Fill the column lists
        for out in out_lst:
            # Defensive: if a particular iteration returned something else,
            # fall back to NaN for all fields.
            if _is_dataclass_instance(out):
                for f_name in dc_fields:
                    data_dict[f_name].append(getattr(out, f_name))
            else:
                for f_name in dc_fields:
                    data_dict[f_name].append(np.nan)

    # ------------------------------------------------------------------
    # 3.3 Regular scalar / single‑value output
    # ------------------------------------------------------------------
    elif multi_output:
        # Node returns a sequence that matches the declared output labels
        for idx, label in enumerate(output_labels):
            data_dict[label] = [out[idx] for out in out_lst]

    else:
        # Node returns a single scalar (or a single object) per iteration
        if len(output_labels) == 1:
            # Simple case – one declared output
            data_dict[output_labels[0]] = out_lst
        else:
            # Ambiguous case – more declared outputs than we can unpack.
            # We store the whole object under each label (the original
            # behaviour) – this mirrors the previous implementation.
            for label in output_labels:
                data_dict[label] = out_lst

    # ------------------------------------------------------------------
    # 4️⃣ Build the DataFrame (fallback to raw dict on error)
    # ------------------------------------------------------------------
    try:
        df = pd.DataFrame(data_dict)
    except Exception as e:
        print(f"Error creating DataFrame: {e}")
        df = pd.DataFrame.from_dict(data_dict, orient="columns")

    return df


# --- Simple iterator, parallel aware ---
@as_function_node
def iterate(
    node: Node,
    input_label: str,
    values: list | np.ndarray,
    debug: bool = False,
    executor=None,
):
    out_lst = _iterate_node(
        node,
        input_label,
        values,
        copy_results=True,
        collect_input=False,
        debug=debug,
        executor=executor,
    )
    # For compatibility: flatten if only one result
    if out_lst and isinstance(out_lst, list) and len(out_lst) == 1:
        out_lst = out_lst[0]
    return out_lst


@as_function_node
# pick a single element from a list
def pick_element(lst: list, index: int) -> any:
    element = lst[index]
    return element


@as_function_node
def ExtractList(out_list: list, label: str, flatten: bool = True):
    import numpy as np

    collect = np.array([out.__getattribute__(label) for out in out_list])
    if flatten:
        collect = collect.flatten()
    return collect


@as_function_node
def InputVector(vec: str = ""):
    try:
        vector = eval(vec)
    except Exception:
        vector = None
    return vector


@as_function_node
def Slice(matrix, slice: str = "::"):
    try:
        result = eval(f"matrix[{slice}]")
    except Exception as e:
        result = None
        print("Slice failed: ", e)
    return result


@as_function_node
def Code(x, code: str = "x**2"):
    try:
        y = eval(code)
    except Exception:
        y = None
    return y


@as_function_node
def GetAttribute(obj, attr: str):
    """Get an attribute from an object."""
    try:
        value = obj.__getattribute__(attr)
    except AttributeError:
        value = None
    return value


@as_function_node
def SetAttribute(obj, attr: str, val: str) -> any:
    """Set an attribute on an object."""
    try:
        obj.__setattr__(attr, val)
    except AttributeError:
        print(f"Attribute {attr} not found in object {obj}")
    return obj


@as_function_node
def Print(x):
    """Print the input value."""
    print(f"Input value: {x}")
    return x


@as_function_node
def GetMask(x: np.ndarray, index: int = 0):
    mask = np.array(x) == index
    return mask


@as_function_node
def Filter(x: np.ndarray, index_vec: np.ndarray):
    result = x[:, index_vec]
    return result
