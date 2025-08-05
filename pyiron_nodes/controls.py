from pyiron_workflow import Node, as_function_node
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


def _iterate_node(
    node, input_label: str, values, copy_results=True, collect_input=False, debug=False
):
    from copy import copy

    out_lst = []
    inp_lst = [] if collect_input else None
    for value in values:
        out = node(**{input_label: value})
        if copy_results:
            out = copy(out)
        out_lst.append(out)
        if collect_input:
            inp_lst.append(value)
        if debug:
            print(f"iterating over {input_label} = {value}, out={out}")
            print("out list: ", [id(o) for o in out_lst])

    return (out_lst, inp_lst) if collect_input else out_lst


@as_function_node
def IterToDataFrame(
    node: Node, input_label: str, values: list | np.ndarray, debug: bool = False
) -> pd.DataFrame:
    import pandas as pd

    out_lst, inp_lst = _iterate_node(
        node, input_label, values, copy_results=True, collect_input=True, debug=debug
    )

    data_dict = {}

    # Decide whether node returns a single value or tuple/list of values
    first_out = out_lst[0] if out_lst else None
    output_labels = list(node.outputs.keys())
    multi_output = isinstance(first_out, (tuple, list, np.ndarray)) and len(
        first_out
    ) == len(output_labels)

    if input_label in output_labels:
        data_dict[f"input_{input_label}"] = inp_lst
    else:
        data_dict[input_label] = inp_lst

    if multi_output:
        # Each item in out_lst is a tuple/list/array with len==number of outputs
        for idx, label in enumerate(output_labels):
            data_dict[label] = [out[idx] for out in out_lst]
    else:
        # Scalar output, assign as a simple column
        if len(output_labels) == 1:
            data_dict[output_labels[0]] = out_lst
        else:
            # Unexpected case: Node declares multiple outputs but returns scalar per call
            data_dict.update({label: out_lst for label in output_labels})

    try:
        df = pd.DataFrame(data_dict)
    except Exception as e:
        print(f"Error creating DataFrame: {e}")
        df = data_dict

    return df


@as_function_node
def iterate(
    node: Node, input_label: str, values: list | np.ndarray, debug: bool = False
):
    out_lst = _iterate_node(
        node, input_label, values, copy_results=True, collect_input=False, debug=debug
    )
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
