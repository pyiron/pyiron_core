from typing import Optional, TYPE_CHECKING

# import numpy as np
# import pandas as pd

from pyiron_workflow import as_function_node

# if TYPE_CHECKING:
#     from pyiron_workflow.graph import Node, Graph


@as_function_node
def LoadGraph(path: str):
    from pyiron_workflow.api.graph import _load_graph

    graph = _load_graph(path)
    return graph


@as_function_node
def Display(graph, optimize: bool = True):
    from pyiron_workflow.api.gui import GuiGraph

    plot = GuiGraph(graph, optimze=optimize)
    return plot


@as_function_node
def DisplayNodes(graph):
    nodes = graph.nodes
    return nodes


@as_function_node
def DisplayEdges(graph):
    edges = graph.edges
    return edges


@as_function_node
def DisplayNodeData(graph):
    from pyiron_workflow.api.gui import display_gui_data

    data = display_gui_data(graph)
    return data


@as_function_node
def DisplayNodeStyle(graph):
    from pyiron_workflow.api.gui import display_gui_style

    style = display_gui_style(graph)
    return style


@as_function_node
def NodesToGui(graph, remove_none: Optional[bool] = False):
    from pyiron_workflow.api.gui import _nodes_to_gui
    nodes = _nodes_to_gui(graph, remove_none=False)
    return nodes


@as_function_node
def EdgesToGui(graph):
    from pyiron_workflow.api.gui import _edges_to_gui

    edges = _edges_to_gui(graph, remove_none=False)
    return edges


@as_function_node(labels=["GraphNode", "Node"])
def ExtractNode(node_label: str, graph):
    node = graph.nodes[node_label]
    return node, node.node


@as_function_node
def NodeInput(node):
    inputs = node.node.inputs
    return inputs


@as_function_node
def DisplayGraphAsJson(graph):
    import json
    from IPython.display import display, JSON

    graph_json = JSON(json.dumps(graph.__getstate__(), indent=2), exanded=True)

    return graph_json


@as_function_node
def OptimizeGraphConnections(graph):
    raise NotImplementedError(
        "pyiron_workflow.graph.base._optimize_graph_connections did not exist at time of refactoring"
    )
    # graph = base._optimize_graph_connections(graph)
    # return graph


@as_function_node
def MarkNodeAsExpanded(graph, node_label: str, expanded: Optional[bool] = True):
    from pyiron_workflow.api.gui import _mark_node_as_collapsed, _mark_node_as_expanded

    if expanded:
        graph = _mark_node_as_expanded(graph, node_label)
    else:
        graph = _mark_node_as_collapsed(graph, node_label)

    return graph


# def get_graph_from_macro(macro_node: Node) -> Graph:
@as_function_node
def GetGraphFromMacro(macro_node):
    raise NotImplementedError("pyiron_workflow.graph.base._get_graph_from_macro did not exist at time of refactoring")
    # graph = base.get_graph_from_macro(macro_node)
    # return graph


# def _get_active_nodes(graph: Graph) -> Nodes:
@as_function_node
def GetActiveNodes(graph):
    from pyiron_workflow.api.gui import _get_active_nodes

    nodes = _get_active_nodes(graph)
    return nodes


@as_function_node
def GetCodeFromGraph(graph):
    from pyiron_workflow.api.graph import get_code_from_graph

    code = get_code_from_graph(graph)
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
    from pyiron_workflow.api.graph import get_updated_graph

    graph = get_updated_graph(full_graph, level=level)
    return graph


@as_function_node
def TopologicalSort(graph):
    from pyiron_workflow.api.graph import topological_sort

    graph = topological_sort(graph)
    return graph
