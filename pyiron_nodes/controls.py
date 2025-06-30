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


@as_function_node
def iterate(
    node: Node, input_label: str, values: list | np.ndarray, debug: bool = False
):
    from copy import copy

    out_lst = list()
    for value in values:
        out = node(**{input_label: value})
        # print("iter out: ", value, len(out[0]))
        out_lst.append(copy(out))
        if debug:
            print(f"iterating over {input_label} = {value}")
            print("out list: ", [id(o[0]) for o in out_lst])

    # transpose output list if node returns multiple outputs
    if out_lst:
        if isinstance(out_lst[0], (tuple, list, np.ndarray)):
            out_lst = list(zip(*out_lst))
        if debug:
            print(f"iterated over {len(values)} values, output length: {len(out_lst)}")
        if len(out_lst) == 1:
            out_lst = out_lst[0]

    return out_lst


@as_function_node
def IterToDataFrame(
    node: Node, input_label: str, values: list | np.ndarray, debug: bool = False
) -> pd.DataFrame:
    from copy import copy

    out_lst = list()
    inp_lst = list()
    for value in values:
        out = node(**{input_label: value})
        # print("iter out: ", value, len(out[0]))
        inp_lst.append(value)
        out_lst.append(copy(out))
        if debug:
            print(f"iterating over {input_label} = {value}")
            print("out list: ", [id(o[0]) for o in out_lst])

    # transpose output list if node returns multiple outputs
    if out_lst:
        if isinstance(out_lst[0], (tuple, list, np.ndarray)):
            out_lst = list(zip(*out_lst))

        data_dict = {}
        if input_label in node.outputs.keys():
            # rename input_label to f'input_{input_label}'
            data_dict[f"input_{input_label}"] = inp_lst
        else:
            data_dict[input_label] = inp_lst
        for out, out_label in zip(out_lst, node.outputs.keys()):
            data_dict[out_label] = out 

        try:
            df = pd.DataFrame(data_dict)
        except Exception as e:
            print(f"Error creating DataFrame: {e}")
            df = data_dict
        if debug:
            print(f"iterated over {len(values)} values, output length: {len(out_lst)}")

    return df #, data_dict


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


# @as_function_node
# def ExtractColumnFromDataFrame(df, column_name: str, n_max: int = -1):
#     if n_max == -1:
#         column = df[column_name]
#     else:
#         column = df[column_name][:n_max]
#     return column
