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
def loop_until(recursive_function: Node, max_steps: int=10):
    x = recursive_function.inputs.x.value
    for i in range(max_steps):
        x, break_condition = recursive_function(x)
        print('loop: ', i, x, break_condition)

        if break_condition:
            break

    return x

@as_function_node
def iterate(node: Node, input_label: str, values: list | np.ndarray):
    out_lst = list()
    for value in values:
        out_lst.append(node(**{input_label: value}))
    return out_lst 