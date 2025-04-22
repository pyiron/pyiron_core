from pyiron_workflow import Node, as_function_node
import numpy as np


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
    out_lst = list()
    for value in values:
        if debug:
            print(f"iterating over {input_label} = {value}")
        out_lst.append(node(**{input_label: value}))
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
def ExtractColumnFromDataFrame(df, column_name: str, n_max: int = -1):
    if n_max == -1:
        column = df[column_name]
    else:
        column = df[column_name][:n_max]
    return column
