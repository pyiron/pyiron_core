from typing import Optional, TYPE_CHECKING
from pyiron_core.pyiron_workflow.api.graph import Graph, GraphNode

# import numpy as np
# import pandas as pd

from pyiron_core.pyiron_workflow import as_function_node

# if TYPE_CHECKING:
#     from pyiron_core.pyiron_workflow.graph import Node, Graph


@as_function_node
def LoadGraph(path: str):
    """
    Load a workflow graph from *path*.

    If *path* is not an absolute path, it is interpreted as relative to
    the directory ``~/pyiron_core.pyiron_workflows`` (the ``~`` is expanded to the
    current user's home directory).

    Parameters
    ----------
    path : str
        Path to the saved graph (JSON, YAML, …).

    Returns
    -------
    pyiron_core.pyiron_workflow.api.graph.Graph
        The deserialized workflow graph.
    """
    import os
    from pyiron_core.pyiron_workflow.api.graph import _load_graph

    # 1️⃣  Is the supplied path already absolute?
    if not os.path.isabs(path):
        # 2️⃣  Resolve ``~/pyiron_core.pyiron_workflows`` → absolute directory
        base_dir = os.path.expanduser("~/pyiron_core.pyiron_workflows")
        # 3️⃣  Join the base directory with the user‑provided relative path
        path = os.path.join(base_dir, path)

    # 4️⃣  Load and return the graph
    graph = _load_graph(path)
    return graph


@as_function_node
def Display(graph: Graph, full_graph: bool = True):
    from pyiron_core.pyiron_workflow.api.gui import GuiGraph

    plot = GuiGraph(graph, full_graph=full_graph)
    return plot


@as_function_node
def DisplayNodes(graph: Graph):
    nodes = graph.nodes
    return nodes


@as_function_node
def DisplayEdges(graph: Graph):
    edges = graph.edges
    return edges


@as_function_node
def DisplayNodeData(graph: Graph):
    from pyiron_core.pyiron_workflow.api.gui import display_gui_data

    data = display_gui_data(graph)
    return data


@as_function_node
def DisplayNodeStyle(graph: Graph):
    from pyiron_core.pyiron_workflow.api.gui import display_gui_style

    style = display_gui_style(graph)
    return style


@as_function_node
def NodesToGui(graph: Graph, remove_none: Optional[bool] = False):
    from pyiron_core.pyiron_workflow.api.gui import _nodes_to_gui

    nodes = _nodes_to_gui(graph, remove_none=False)
    return nodes


@as_function_node
def EdgesToGui(graph: Graph):
    from pyiron_core.pyiron_workflow.api.gui import _edges_to_gui

    edges = _edges_to_gui(graph, remove_none=False)
    return edges


@as_function_node(labels=["GraphNode", "Node"])
def ExtractNode(node_label: str, graph: Graph):
    node = graph.nodes[node_label]
    return node, node.node


@as_function_node
def NodeInput(node: GraphNode):
    inputs = node.node.inputs
    return inputs


@as_function_node
def DisplayGraphAsJson(graph: Graph):
    import json
    from IPython.display import display, JSON

    graph_json = JSON(json.dumps(graph.__getstate__(), indent=2), exanded=True)

    return graph_json


@as_function_node
def OptimizeGraphConnections(graph: Graph):
    raise NotImplementedError(
        "pyiron_core.pyiron_workflow.graph.base._optimize_graph_connections did not exist at time of refactoring"
    )
    # graph = base._optimize_graph_connections(graph)
    # return graph


@as_function_node
def MarkNodeAsExpanded(graph: Graph, node_label: str, expanded: Optional[bool] = True):
    from pyiron_core.pyiron_workflow.api.gui import _mark_node_as_collapsed, _mark_node_as_expanded

    if expanded:
        graph = _mark_node_as_expanded(graph, node_label)
    else:
        graph = _mark_node_as_collapsed(graph, node_label)

    return graph


# def get_graph_from_macro(macro_node: Node) -> Graph:
@as_function_node
def GetGraphFromMacro(macro_node):
    raise NotImplementedError(
        "pyiron_core.pyiron_workflow.graph.base._get_graph_from_macro did not exist at time of refactoring"
    )
    # graph = base.get_graph_from_macro(macro_node)
    # return graph


# def _get_active_nodes(graph: Graph) -> Nodes:
@as_function_node
def GetActiveNodes(graph: Graph):
    from pyiron_core.pyiron_workflow.api.gui import _get_active_nodes

    nodes = _get_active_nodes(graph)
    return nodes


@as_function_node
def GetCodeFromGraph(
    graph: Graph,
    sort_graph: bool = True,
    use_node_default: bool = False,
    scope_inputs: bool = True,
):
    from pyiron_core.pyiron_workflow.api.graph import get_code_from_graph

    code = get_code_from_graph(
        graph,
        sort_graph=sort_graph,
        use_node_default=use_node_default,
        scope_inputs=scope_inputs,
    )
    return code


@as_function_node
def GetFunctionFromNode(graph_node):
    import inspect

    code = inspect.getsource(graph_node.node._func)
    return code


@as_function_node
def DisplayCode(code):
    from pygments import highlight
    from pygments.lexers import Python2Lexer
    from pygments.formatters import TerminalFormatter

    print(highlight(code, Python2Lexer(), TerminalFormatter()))

    end = None
    return end


@as_function_node
def ConvertMacroToWorkflow(macro_node):
    kwargs = {}
    for inp in macro_node.inputs.data["label"]:
        inp_port_label = f"inp_port_{inp}"
        kwargs[inp] = inp_port_label

    out = macro_node._func(**kwargs)

    workflow = out._workflow
    return workflow


@as_function_node
def GetUpdatedGraph(full_graph, level: Optional[int] = 0):
    from pyiron_core.pyiron_workflow.api.graph import get_updated_graph

    graph = get_updated_graph(full_graph, level=level)
    return graph


@as_function_node
def TopologicalSort(graph: Graph):
    from pyiron_core.pyiron_workflow.api.graph import topological_sort

    graph = topological_sort(graph)
    return graph


@as_function_node
def RemoveNode(graph: Graph, node_label: str):
    from pyiron_core.pyiron_workflow.graph.base import remove_node

    graph = remove_node(graph, node_label)
    return graph
