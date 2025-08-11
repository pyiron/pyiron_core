from pyiron_workflow import Node, as_function_node, as_inp_dataclass_node


@as_function_node("Executor")
def SingleNodeExecutor(max_workers: int = 1):
    from executorlib import SingleNodeExecutor as Executor

    return Executor(max_workers=max_workers)


@as_function_node("Executor")
def ThreadPoolExecutor(max_workers: int = 1):
    from concurrent.futures import ThreadPoolExecutor as Executor

    return Executor(max_workers=max_workers)


@as_function_node("Executor")
def ProcessPoolExecutor(max_workers: int = 1):
    from concurrent.futures import ProcessPoolExecutor as Executor

    return Executor(max_workers=max_workers)


@as_function_node
def IterNode(
    node: Node,
    kwarg_name: str,
    kwarg_list: list,
    Executor: type = None,
    store: bool = True,
    _db=None,
):

    # TODO: add to node ._hash_parent (with hash of this node)
    from concurrent.futures import as_completed
    from pyiron_workflow.api.graph import run_node
    from copy import copy
    from pandas import DataFrame

    # import numpy as np

    # kwarg_initial = node.inputs[kwarg_name].value
    # print("kwarg_initial", kwarg_initial)

    out_dict = dict()
    if Executor is None:
        for el in kwarg_list:
            # print('iter kwargs: ', node.kwargs)
            out_dict[el] = node(**{kwarg_name: el})
            # if hasattr(node, "_hash_parent"):
            #     print("node._hash_parent", el, node._hash_parent)
            # print("iter_node: ", node.inputs, node_run.inputs)
    else:
        with Executor as executor:
            # Start the load operations and mark each future with its index
            futures = dict()
            if hasattr(node, "_graph_node"):
                graph_node = node._graph_node
                print("graph_node", graph_node.label)
            else:
                raise NotImplementedError(
                    "Node must contain a link to its graph node to use the executor"
                )
            for el in kwarg_list:
                futures[
                    executor.submit(run_node, graph_node, **{kwarg_name: el})
                ] = el

            for future in as_completed(futures):
                out = futures[future]
                data = future.result()
                out_dict[out] = data

    # create dataframe, keep the order of the input list
    results = [out_dict[el] for el in kwarg_list]
    # node.inputs[kwarg_name] = kwarg_initial

    df = DataFrame({kwarg_name: kwarg_list, "result": results})
    return df


@as_function_node
def fcc_metals():
    """
    List of metals with FCC structure
    """
    list_of_metals = ["Cu", "Ag", "Au", "Pt", "Ni", "Pd", "Rh", "Ir"]
    return list_of_metals
