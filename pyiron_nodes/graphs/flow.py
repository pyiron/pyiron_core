from typing import Optional

# import numpy as np
# import pandas as pd

from pyiron_workflow import as_function_node


@as_function_node
def LoadGraph(path: str):
    from pyiron_workflow.graph import base

    graph = base._load_graph(path)
    return graph


@as_function_node
def Display(graph):
    from pyiron_workflow.graph import base

    plot = base.GuiGraph(graph)
    return plot


@as_function_node
def DisplayNodes(graph):
    from pyiron_workflow.graph import base

    nodes = graph.nodes
    return nodes


@as_function_node
def DisplayEdges(graph):
    from pyiron_workflow.graph import base

    edges = graph.edges
    return edges


@as_function_node
def DisplayNodeData(graph):
    from pyiron_workflow.graph import base

    data = base.display_gui_data(graph)
    return data


@as_function_node
def DisplayNodeStyle(graph):
    from pyiron_workflow.graph import base

    style = base.display_gui_style(graph)
    return style


@as_function_node
def DisplayGuiEdges(graph):
    from pyiron_workflow.graph import base

    edges = base._edges_to_gui(graph, remove_none=False)
    return edges


@as_function_node
def ExtractNode(node_label: str, graph):
    from pyiron_workflow.graph import base

    node = graph.nodes[node_label]
    return node


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
