from pyiron_core.pyiron_workflow import Node, as_function_node
import numpy as np
import pandas as pd


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


from concurrent.futures import as_completed
from copy import copy

def _iterate_node(
    node, input_label: str, values, copy_results=True, collect_input=False, debug=False, executor=None
):
    out_lst = []
    inp_lst = [] if collect_input else None

    if executor is None:
        # Sequential execution
        for value in values:
            # out = node(**{input_label: value})
            # node.inputs[input_label].value = value
            node.inputs.__setattr__(input_label, value)
            # print(f"Setting {input_label} = {value}", node.inputs[input_label].value)
            out = node.run()
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
        futures = {executor.submit(node, **{input_label: value}): (idx, value) 
                   for idx, value in enumerate(values)}
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
    node: Node, input_label: str, values: list | np.ndarray, debug: bool = False, executor=None
) -> pd.DataFrame:
    import pandas as pd

    out_lst, inp_lst = _iterate_node(
        node, input_label, values, copy_results=True, collect_input=True, debug=debug, executor=executor
    )

    data_dict = {}
    # Decide whether node returns a single value or tuple/list of values
    first_out = out_lst[0] if out_lst else None
    output_labels = list(node.outputs.keys())
    multi_output = isinstance(first_out, (tuple, list, np.ndarray)) and len(
        first_out) == len(output_labels)

    # Ensure no column name conflict for input
    if input_label in output_labels:
        data_dict[f"input_{input_label}"] = inp_lst
    else:
        data_dict[input_label] = inp_lst

    if multi_output:
        for idx, label in enumerate(output_labels):
            data_dict[label] = [out[idx] for out in out_lst]
    else:
        if len(output_labels) == 1:
            data_dict[output_labels[0]] = out_lst
        else:
            data_dict.update({label: out_lst for label in output_labels})

    try:
        df = pd.DataFrame(data_dict)
    except Exception as e:
        print(f"Error creating DataFrame: {e}")
        df = data_dict

    return df

# --- Simple iterator, parallel aware ---
@as_function_node
def iterate(
    node: Node, input_label: str, values: list | np.ndarray, debug: bool = False, executor=None
):
    out_lst = _iterate_node(
        node, input_label, values, copy_results=True, collect_input=False, debug=debug, executor=executor
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
    except:
        vector = None
    return vector


@as_function_node
def Slice(matrix, slice: str = "::"):
    try:
        result = eval(f"matrix[{slice}]")
    except:
        result = None
    return result


@as_function_node
def Code(x, code: str = "x**2"):
    try:
        y = eval(code)
    except:
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


# @as_function_node
# def ExtractColumnFromDataFrame(df, column_name: str, n_max: int = -1):
#     if n_max == -1:
#         column = df[column_name]
#     else:
#         column = df[column_name][:n_max]
#     return column

@as_function_node
def Print(x):
    """Print the input value."""
    print(f"Input value: {x}")
    return x
