__author__ = "Joerg Neugebauer"
__copyright__ = (
    "Copyright 2024, Max-Planck-Institut for Sustainable Materials GmbH - "
    "Computational Materials Design (CM) Department"
)
__version__ = "0.2"
__maintainer__ = ""
__email__ = ""
__status__ = "development"
__date__ = "Jan 3, 2025"

from pyiron_workflow import Node, Port, as_function_node

from dataclasses import field
from pyiron_workflow.graph.decorators import (
    as_dotdict_dataclass,
    NestedDict,
    NestedList,
    transpose_list_of_dicts,
    get_import_path_from_type,
)
from typing import Union, List, Tuple
import pandas as pd
import numpy as np
import importlib

from copy import deepcopy

NotData = "NotData"


def get_node_from_path(import_path, log=None):
    # Split the path into module and object part
    module_path, _, name = import_path.rpartition(".")
    # print('module_path: ', module_path)
    # Import the module
    try:
        module = importlib.import_module(module_path)
    except ModuleNotFoundError as e:
        log.append_stderr(e)
        return None
    # Get the object
    object_from_path = getattr(module, name)
    return object_from_path


def _getstate__graph_node(self):
    node_dict = {k: self[k] for k in self.keys() if k != "node"}
    if self.node is not None:
        node_dict["node"] = self.node.__getstate__()
    # node_dict["node"] = self.node.__getstate__()
    if self.graph is not None:
        node_dict["graph"] = self.graph.__getstate__()
    return node_dict


def _setstate__graph_node(self, state):
    for k, v in state.items():
        if k == "node":
            self.node = Node().__setstate__(v)
        elif k == "graph":
            if v is not None:
                self.graph = Graph().__setstate__(v)
            # print("setting graph: ", v, state)
            # self.graph = Graph().__setstate__(v)
        else:
            self[k] = v
    return self


@as_dotdict_dataclass(
    __getstate__=_getstate__graph_node, __setstate__=_setstate__graph_node
)
class GraphNode:
    id: str = None  # unique identifier for the node (no two nodes can have the same id)
    import_path: str = None  # import path to the node
    label: str = None  # label/name of the node
    parent_id: str = None  # parent node id
    level: int = 0  # level in the graph
    node: Node = None  # node object
    graph: "Graph" = None  # graph object if node is a macro node
    node_type: str = "node"  # type of node
    widget_type: str = None  # type of widget to display the node
    expanded: bool = False  # expanded or collapsed state


def _getstate_GraphEdge(self):
    return self.asdict(remove_none=True)


def _setstate_GraphEdge(self, state):
    for k, v in state.items():
        self.__setattr__(k, v)
    return self


@as_dotdict_dataclass(
    __getstate__=_getstate_GraphEdge, __setstate__=_setstate_GraphEdge
)
class GraphEdge:
    source: str
    target: str
    sourceHandle: str = None
    targetHandle: str = None


Nodes = NestedDict[str, GraphNode]
# Edges = NestedList[str, GraphEdge]


class Edges(NestedList):
    def __init__(self, obj_type=GraphEdge):
        super().__init__(obj_type=obj_type)


def _add_obj_to_graph(graph, obj):
    if isinstance(obj, (Node, Graph, GraphNode)):
        new_graph = add_node(graph, obj)
    elif isinstance(obj, GraphEdge):
        new_graph = add_edge(graph, **obj.asdict())
    else:
        raise TypeError(f"Unexpected node type {type(obj)}")
    return new_graph


def _getstate__graph(self):
    state = dict(
        label=self.label,
        nodes=self.nodes.__getstate__(),
        edges=self.edges.__getstate__(),
    )
    if self.root_node is not None:
        state["root_node"] = self.root_node.__getstate__()
    return state


def _setstate__graph(self, state):
    self.label = state["label"]
    self.nodes = Nodes().__setstate__(state["nodes"])
    # make update to the nodes without copying the object
    # edges = Edges().__setstate__(state["edges"])
    # for edge in edges:
    #     self += edge
    self.edges = Edges().__setstate__(state["edges"])
    if "graph" in state:
        self.graph = Graph().__setstate__(state["graph"])
    else:
        self.graph = dict()
    if "root_node" in state:
        self.root_node = GraphNode().__setstate__(state["root_node"])
    return self


@as_dotdict_dataclass(
    __add__=_add_obj_to_graph,
    __getstate__=_getstate__graph,
    __setstate__=_setstate__graph,
)
class Graph:
    # (TODO: check whether we need both id and label)
    id: str = None  # to make it behave like a node
    label: str = None
    root_node: Node = (
        None  # root node of the graph (if the graph represents a macro node)
    )
    nodes: Nodes = field(default_factory=lambda: NestedDict(obj_type=GraphNode))
    edges: Edges = field(default_factory=lambda: NestedList(obj_type=GraphEdge))
    graph: dict = field(default_factory=lambda: {})


@as_dotdict_dataclass()
class GuiNode:
    id: str  # unique identifier for the node (no two nodes can have the same id)
    data: dict = None
    position: dict = None
    style: dict = None
    targetPosition: str = "left"
    sourcePosition: str = "right"
    type: str = None
    parentId: str = None
    extent: str = None
    expanded: bool = False


@as_dotdict_dataclass()
class GuiData:
    label: str = None  # label/name of the node as shown in the gui
    source_labels: list = field(default_factory=lambda: [])
    target_labels: list = field(default_factory=lambda: [])
    import_path: str = None
    target_values: list = field(default_factory=lambda: [])
    target_types: list = field(default_factory=lambda: [])
    source_values: list = field(default_factory=lambda: [])
    source_types: list = field(default_factory=lambda: [])
    expanded: bool = False


@as_dotdict_dataclass()
class GuiStyle:
    backgroundColor: str = "rgba(0, 255, 0, 0.5)"  # light green
    height: int = 50
    width: int = 100
    padding: int = 5
    border: str = "1px black solid"
    borderRadius: str = "10px"


def copy_graph(graph: Graph) -> Graph:
    from copy import copy

    return Graph(label=graph.label, nodes=copy(graph.nodes), edges=copy(graph.edges))


def add_node(
    graph: "Graph", node: Union["Node", "Graph", "GraphNode"], label=None
) -> "Graph":
    label = _get_label(node, label)
    label = get_unique_label(graph, label)
    _check_label_exists(graph, label)

    if isinstance(node, Node):
        if node.node_type == "macro_node":
            node = get_graph_from_macro_node(node)

    node.label = label

    if isinstance(node, Node):
        new_graph = _add_node_instance(graph, node, label)
    elif isinstance(node, Graph):
        new_graph = _add_graph_instance(graph, node, label)
    elif isinstance(node, GraphNode):
        if node.node is None:
            # should be done on GraphNode creation
            if label is None:
                label = node.id
            node.node = get_node_from_path(node.import_path)(label=label)

        new_graph = copy_graph(graph)
        new_graph.nodes[label] = node
    else:
        raise TypeError(f"Unexpected node type {type(node)}")
    new_graph = _expand_node(new_graph, label)  # TODO: validate that it is recursive
    return new_graph


def remove_node(graph: Graph, label: str) -> Graph:
    """
    Remove a node and its associated edges from the graph.

    Args:
        graph (Graph): The graph from which to remove the node.
        label (str): The label of the node to be removed.

    Returns:
        Graph: The updated graph with the node and its edges removed.
    """
    new_graph = copy_graph(graph)
    if label in graph.nodes.keys():
        del new_graph.nodes[label]

    edges_to_remove = [
        edge for edge in graph.edges if edge.source == label or edge.target == label
    ]
    for edge in edges_to_remove:
        new_graph.edges.remove(edge)

    # TODO: remove node from connected ports?

    return new_graph


def remove_edge(graph: Graph, edge: GraphEdge) -> Graph:
    new_graph = copy_graph(graph)
    if edge in graph.edges:
        new_graph.edges.remove(edge)
    else:
        raise ValueError(f"Edge {edge} not found in graph")
    return new_graph


# def get_graph_from_macro(macro_node: Node) -> Graph:

#     kwargs = {}
#     for inp in macro_node.inputs.data["label"]:
#         inp_port_label = f"inp_port__{inp}"
#         kwargs[inp] = inp_port_label
#         # kwargs[inp] = identity(inp_port_label, label=inp_port_label)

#     out = macro_node._func(**kwargs)

#     graph = get_graph_from_wf(out._workflow, wf_label=macro_node.label)
#     graph.root_node = macro_node
#     return graph


def _get_label(node, label):
    if label is None:
        label = node.label
    return label


def get_unique_label(graph: Graph, label: str):
    if label in graph.nodes.keys():
        i = 1
        while f"{label}_{i}" in graph.nodes.keys():
            i += 1
        label = f"{label}_{i}"
    return label


def _check_label_exists(graph: Graph, label):
    if label in graph.nodes.keys():
        raise ValueError(f"Node with label {label} already exists")
    return True


def _add_node_instance(graph: Graph, node, label):
    new_graph = copy_graph(graph)
    new_graph.nodes[label] = GraphNode(
        id=label,
        import_path=get_import_path_from_type(node._func),
        node=node,
        label=label,
        widget_type="customNode",
    )
    return new_graph


def _add_graph_instance(graph: Graph, sub_graph: Graph, label: str = None):
    new_graph = copy_graph(graph)
    sub_graph.id = label
    new_graph.nodes[label] = GraphNode(
        id=label,
        node=sub_graph.root_node,
        graph=sub_graph,
        label=label,
        node_type="graph",
        widget_type="customNode",
    )
    return new_graph


def add_edge(
    graph: Graph,
    source: str = None,
    target: str = None,
    sourceHandle: str = None,
    targetHandle: str = None,
):
    new_graph = copy_graph(graph)
    new_graph.edges.append(
        GraphEdge(
            source=source,
            sourceHandle=sourceHandle,
            target=target,
            targetHandle=targetHandle,
        )
    )
    if not (source.startswith("va_") or target.startswith("va_")):
        new_graph = _update_target_port(new_graph, new_graph.edges[-1])
    return new_graph


def _update_target_port(graph: Graph, edge: GraphEdge):
    source_port = graph.nodes[edge.source].node.outputs.__getattr__(edge.sourceHandle)
    graph.nodes[edge.target].node.inputs.__setattr__(edge.targetHandle, source_port)
    # print(f"updated target port {edge.targetHandle} in node {edge.target}, {source_port}")
    return graph


# tools for expanding and collapsing nodes


# def _get_inner_edge_input(graph, edge):
#     inner_source = f"var_i_{edge.target}__{edge.targetHandle}"
#     df = _edges_to_gui(graph).df
#     matching_edge = df.loc[df["source"] == inner_source].iloc[0].to_dict()
#     new_edge = GraphEdge(
#         edge.source,
#         matching_edge["target"],
#         edge.sourceHandle,
#         matching_edge["targetHandle"],
#     )
#     return new_edge


# def _get_inner_edge_output(graph, edge):
#     inner_source = f"var_o_{edge.source}__{edge.sourceHandle}"
#     df = _edges_to_gui(graph).df
#     matching_edge = df.loc[df["target"] == inner_source].iloc[0].to_dict()
#     new_edge = GraphEdge(
#         matching_edge["source"],
#         edge.target,
#         matching_edge["sourceHandle"],
#         edge.targetHandle,
#     )
#     return new_edge


# def _get_expanded_nodes(graph: Graph) -> List[str]:
#     expanded_nodes = []
#     for k, v in graph.nodes.items():
#         if v.expanded:
#             expanded_nodes.append(k)
#     return expanded_nodes


# def _remove_var_input_edges(graph: Graph):
#     edges_to_remove = []
#     for edge in graph.edges:
#         if edge.source.startswith("var_i_"):
#             # print(f"removing edge {edge}")
#             edges_to_remove.append(edge)
#     new_graph = copy_graph(graph)
#     for edge in edges_to_remove:
#         new_graph.edges.remove(edge)
#     return new_graph


# def _remove_var_input_nodes(graph: Graph):
#     nodes_to_remove = []
#     for node_label in graph.nodes.keys():
#         if node_label.startswith("var_i_"):
#             nodes_to_remove.append(node_label)
#     new_graph = copy_graph(graph)
#     for node_label in nodes_to_remove:
#         del new_graph.nodes[node_label]
#     return new_graph


# def _remove_var_output_edges(graph: Graph):
#     edges_to_remove = []
#     for edge in graph.edges:
#         if edge.target.startswith("var_o_"):
#             # print(f"removing edge {edge}")
#             edges_to_remove.append(edge)
#     new_graph = copy_graph(graph)
#     for edge in edges_to_remove:
#         new_graph.edges.remove(edge)
#     return new_graph


# def _remove_var_output_nodes(graph: Graph):
#     nodes_to_remove = []
#     for node_label in graph.nodes.keys():
#         if node_label.startswith("var_o_"):
#             nodes_to_remove.append(node_label)
#     new_graph = copy_graph(graph)
#     for node_label in nodes_to_remove:
#         del new_graph.nodes[node_label]
#     return new_graph


# def _optimize_graph_connections(graph):
#     new_graph = copy_graph(graph)
#     for node_label in _get_expanded_nodes(new_graph):
#         for i_edge, edge in enumerate(new_graph.edges):
#             if edge.target == node_label:
#                 print(
#                     f"updating edge {edge.targetHandle} in node {edge.target}, {edge.source} {edge.sourceHandle}"
#                 )
#                 new_edge = _get_inner_edge_input(graph, edge)
#                 new_graph.edges[i_edge] = new_edge
#                 # update edge port in target node
#                 new_graph = _update_target_port(new_graph, new_edge)
#     new_graph = _remove_var_input_edges(new_graph)
#     new_graph = _remove_var_input_nodes(new_graph)
#     new_graph = _remove_var_output_edges(new_graph)
#     new_graph = _remove_var_output_nodes(new_graph)

#     return new_graph


def _expand_node(graph, node_label: str):
    new_graph = copy_graph(graph)
    graph_node = new_graph.nodes[node_label]
    if graph_node.node_type == "graph":
        graph_node.expanded = True
        node = graph_node.graph

        for k, v in node.nodes.items():
            v.parent_id = (
                node.label
            )  # set parent id, make assignment more robust (easy to confuse id and label)
            v.level += 1
            new_graph.nodes[k] = v

        for edge in node.edges:
            new_graph.edges.append(edge)
    return new_graph


def _mark_node_as_collapsed(graph, node_label: str):
    new_graph = copy_graph(graph)
    graph_node = new_graph.nodes[node_label]
    if graph_node.node_type == "graph":
        graph_node.expanded = False
    return new_graph


def _mark_node_as_expanded(graph, node_label: str):
    new_graph = copy_graph(graph)
    graph_node = new_graph.nodes[node_label]
    if graph_node.node_type == "graph":
        graph_node.expanded = True
    return new_graph


def _get_active_nodes(graph: Graph) -> Nodes:
    active_nodes = NestedDict()
    # get all nodes that are not inside a collapsed node
    for k, v in graph.nodes.items():
        if v.parent_id is None:
            active_nodes[k] = v
        else:
            parent = graph.nodes[v.parent_id]
            if parent.expanded:
                active_nodes[k] = v
    return active_nodes


def _get_active_edges(graph: Graph) -> Edges:
    active_edges = NestedList(obj_type=GraphEdge)
    active_nodes = _get_active_nodes(graph)
    # get all edges that are not inside a collapsed node
    for edge in graph.edges:
        if edge.source in active_nodes.keys() and edge.target in active_nodes.keys():
            active_edges.append(edge)
    return active_edges


####################################################################################################
# Utility functions
####################################################################################################
def remove_node_with_reconnected_edges(graph: Graph, node_label: str) -> Graph:
    new_graph = deepcopy(graph)
    # node = new_graph.nodes[node_label]
    # find single target edge to node
    source_nodes = []
    source_node_labels = []
    for edge in new_graph.edges:
        if edge.target == node_label:
            source_node = new_graph.nodes[edge.source]
            if source_node.label in source_node_labels:
                # print(f"Source node {source_node.label} already connected to {node_label}")
                continue
            source_node_labels.append(source_node.label)
            source_nodes.append(source_node)
            source_handle = edge.sourceHandle
            inner_edge = edge

    if len(source_nodes) > 1:
        print(f"Source nodes: {source_nodes}", node_label)
        raise ValueError("InputNode has multiple sources")

    if len(source_nodes) == 1:
        source_node = source_nodes[0]
        # print(f"Found source node {source_node.label}", inner_edge)

        for edge in new_graph.edges:
            if edge.source == node_label:
                new_edge = GraphEdge(
                    inner_edge.source,
                    edge.target,
                    source_handle,
                    edge.targetHandle,
                )
                # print(f"Rewiring edge {edge} to {new_edge}")
                new_graph.edges.append(new_edge)

        remove_edge(new_graph, inner_edge)
    else:
        print(f"Node {node_label} has no source nodes")

    del new_graph.nodes[node_label]
    return new_graph


####################################################################################################
# Transform to and from Graph, Workflow and Code
####################################################################################################


def graph_edges_to_wf_edges(edges: Edges) -> List[dict]:
    wf_edges = []
    for edge in edges:
        if not edge.source.startswith("va_i_") and not edge.target.startswith(
            "va_o_"
        ):
            wf_edges.append(edge.asdict())
    return wf_edges


def get_wf_from_graph(graph: Graph) -> "Workflow":
    from pyiron_workflow import Workflow

    wf = Workflow(graph.label)
    # Add nodes to Workflow
    for node in graph.nodes.values():
        label, import_path = node.label, node.import_path

        if not label.startswith("va_i_"):
            kwargs = dict()

            # Add non-default arguments to node
            for edge in graph.edges:
                if edge.target == label:
                    # TODO: get value from source node (is there not converted to string)
                    if edge.source.startswith("va_i_"):
                        if edge.sourceHandle.startswith("__str_"):
                            kwargs[edge.targetHandle] = edge.sourceHandle[6:]
                        else:
                            kwargs[edge.targetHandle] = eval(edge.sourceHandle)
                    # kwargs[target_handle] = source_handle

            new_node = get_node_from_path(import_path)(**kwargs)
            wf.add_node(label, new_node)
    wf._set_edges(graph_edges_to_wf_edges(graph.edges))

    return wf


def get_code_from_graph(
    graph: Graph,
    workflow_lib: str = "pyiron_workflow",
    pyiron_nodes_lib: str = "pyiron_nodes",
):
    """
    Generate Python source code from graph.

    Args:
        label (str): The label to use in the generated code.
        module_a (str): The name of the module to import from.
        module_b (str): The name of the module to import.

    Returns:
        str: The generated Python source code.
    """
    import black

    # get input kwargs from graph
    kwargs = str()
    for node in graph.nodes.values():
        if node.label.startswith("va_i_"):
            print(f"Found input node {node.label}")
            inp = node.label.split("__")[-1]
            kwargs += inp + ", "  # =None, " include default values and type hints
            for edge in graph.edges:
                if edge.target == node.label:
                    print(f"Found edge {edge}")

    code = f"""
def {graph.label}({kwargs}):

    from {workflow_lib} import Workflow
    import {pyiron_nodes_lib}

    wf = Workflow('{graph.label}')

"""

    return_args = []

    # Add nodes to Workflow
    for node in graph.nodes.values():
        label, import_path = node.label, node.import_path
        if not label.startswith("va_"):
            code += f"""    wf.{label} = {import_path}("""

        # Add edges
        first_edge = True
        for edge in graph.edges:
            if edge.target == label:
                if first_edge:
                    first_edge = False
                else:
                    code += """, """
                if edge.source.startswith("va_"):
                    code += f"""{edge.targetHandle}={edge.sourceHandle}"""
                else:
                    if edge.target.startswith("va_o_"):
                        return_args.append(f"wf.{edge.source}")
                    else:
                        source_node = graph.nodes[edge.source]
                        if source_node.node.n_out_labels == 1:
                            code += f"""{edge.targetHandle}=wf.{edge.source}"""
                        else:
                            code += f"""{edge.targetHandle}=wf.{edge.source}.outputs.{edge.sourceHandle}"""
        if not label.startswith("va_"):
            code += f""") \n"""

    code += "\n" + "    return " + ", ".join(return_args) + "\n"
    return code


def get_graph_from_wf(
    wf: "Workflow",
    wf_outputs: Tuple[Node | Port],
    out_labels: List[str],
    wf_label: str = None,
) -> Graph:
    if wf_label is None:
        wf_label = wf.label

    print("wf_label: ", wf_label)
    graph = Graph(label=wf_label)

    for label, node in wf._nodes.items():
        # TODO: node input changes due o rewiring edges!
        # Should be copied but in the present implementation deepcopy does not work
        # print(f"Adding node {label}")
        graph = add_node(graph, node, label=label)

        data = node.inputs.data
        changed_args = _different_indices(data["default"], data["value"])

        # construct the input nodes for the non-default arguments
        for i in changed_args:
            value = data["value"][i]
            handle = data["label"][i]
            if not isinstance(value, (Node, Port)):
                if isinstance(value, str) and value.startswith("va_i_"):
                    # print(f"Adding input node {handle}", value)
                    inp_node_label = value
                    if inp_node_label not in graph.nodes:
                        # print(f"Adding input node {inp_node_label}")
                        graph += identity(label=inp_node_label)

                    edge = GraphEdge(
                        target=label,
                        targetHandle=handle,
                        source=inp_node_label,
                        sourceHandle="x",  # input label of identity node
                    )

                    graph += edge

    for edge in wf._edges:
        graph += GraphEdge(**edge)

    # print(f"Adding output nodes {out_labels}")
    for out_label, wf_output in zip(out_labels, wf_outputs):
        out_node_label = f"va_o_{wf_label}__{out_label}"
        graph += identity(label=out_node_label)

        if isinstance(wf_output, Port):
            target = wf_output.node.label
            target_handle = wf_output.label
        elif isinstance(wf_output, Node):
            target = wf_output.label
            output_ports_labels = wf_output.outputs.data["label"]
            if len(output_ports_labels) == 1:
                target_handle = output_ports_labels[0]
            else:
                raise ValueError()
        else:
            raise ValueError()

        edge = GraphEdge(
            source=target,
            sourceHandle=target_handle,
            target=out_node_label,
            targetHandle="x",  # input label of identity node
        )

        print("target: ", target, target_handle)
        graph += edge

    sorted_graph = topological_sort(graph)
    return sorted_graph


def get_graph_from_macro_node(macro_node: Node) -> Graph:
    kwargs = {}
    for inp in macro_node.inputs.data["label"]:
        inp_port_label = f"va_i_{macro_node.label}__{inp}"
        kwargs[inp] = inp_port_label

    out = macro_node._func(**kwargs)
    if not isinstance(out, tuple):
        out = (out,)

    # each output instance contains link to workflow, check that it works for multiple outputs
    wf = out[0]._workflow
    print("label: ", wf.label, macro_node.label)
    wf.label = macro_node.label

    out_labels = macro_node.outputs.data["label"]

    new_graph = get_graph_from_wf(
        wf, wf_outputs=out, out_labels=out_labels, wf_label=macro_node.label
    )
    print("new_graph: ", new_graph.label)
    return new_graph


####################################################################################################
# Collapse and Expand Graphs and Macro Nodes
####################################################################################################


def expand_node(graph: Graph, node_label: str) -> Graph:
    new_graph = deepcopy(graph)
    graph_node = new_graph.nodes[node_label]

    if graph_node.node_type == "graph":
        print(f"Expanding node {node_label}")
        graph_node.expanded = True
        inner_graph = graph_node.graph

        # # Add inner nodes to the graph
        # for inner_node_label, inner_node in inner_graph.nodes.items():
        #     print(f"Adding inner node {inner_node_label}", type(inner_node))
        #     inner_node.parent_id = node_label
        #     inner_node.level += 1
        #     new_graph.nodes[inner_node_label] = inner_node

        # Rewire edges
        new_edges = Edges()
        # Add inner edges
        for inner_edge in inner_graph.edges:
            new_edges.append(inner_edge)

        for edge in new_graph.edges:
            new_edges.append(edge)

        new_graph.edges = new_edges

        var_nodes = []
        for k, nodes in new_graph.nodes.items():
            if k.startswith("va_"):
                var_nodes.append(k)

        print(f"Removing nodes {set(var_nodes)}")

        for k in set(var_nodes):
            new_graph = remove_node_with_reconnected_edges(new_graph, k)

    return new_graph


def collapse_node(graph: Graph, node_label: str) -> Graph:
    new_graph = deepcopy(graph)
    graph_node = new_graph.nodes[node_label]

    if graph_node.node_type == "graph":
        print(f"Collapsing node {node_label}")
        graph_node.expanded = False
        for edge in new_graph.edges:
            n = len("va_i_")
            if edge.source.startswith("va_"):
                source, source_handle = edge.source[n:].split("__")
                edge.source = source
                edge.sourceHandle = source_handle
            if edge.target.startswith("va_"):
                target, target_handle = edge.target[n:].split("__")
                edge.target = target
                edge.targetHandle = target_handle

        nodes_to_remove = []
        for node in new_graph.nodes.values():
            if node.label.startswith("va_"):
                nodes_to_remove.append(node.label)
            elif node.parent_id == node_label:
                nodes_to_remove.append(node.label)

        for node_label in nodes_to_remove:
            del new_graph.nodes[node_label]

    return new_graph


def get_full_graph_from_wf(wf: "Workflow") -> Graph:
    graph = Graph(label=wf.label)

    macro_node_labels = []
    for label, node in wf._nodes.items():
        if node.node_type == "macro_node":
            new_node = get_graph_from_macro_node(node)
            graph = add_node(graph, new_node, label=label)
            graph.nodes[node.label].node = node
            macro_node_labels.append(label)
        else:
            graph = add_node(graph, node, label=label)

    for edge in wf._edges:
        source = edge["source"]
        target = edge["target"]
        source_handle = edge["sourceHandle"]
        target_handle = edge["targetHandle"]

        if source in macro_node_labels:
            source = "va_o_" + source + "__" + source_handle
            source_handle = "x"
        elif target in macro_node_labels:
            target = "va_i_" + target + "__" + target_handle
            target_handle = "x"

        graph += GraphEdge(source, target, source_handle, target_handle)

    return graph


####################################################################################################
# Graph topology, sorting, and traversal functions
####################################################################################################
from collections import defaultdict
import pathlib
import json
from typing import List, Tuple, Union


from typing import TYPE_CHECKING, List, Tuple

if TYPE_CHECKING:
    from pyiron_workflow import Workflow


def _different_indices(default, value):
    # TODO: quick fix, use _get_non_default_inputs from simple_workflow.py
    return [
        i
        for i in range(len(default))
        if (str(default[i]) != str(value[i])) or (str(value[i]) in (NotData))
    ]


def _convert_to_integer_representation(graph: Graph):
    # Create a dictionary mapping node labels to indices
    node_to_index = {
        node_label: index for index, node_label in enumerate(graph.nodes.keys())
    }

    # Convert edge list to integer representation
    integer_edges = [
        (node_to_index[edge.source], node_to_index[edge.target]) for edge in graph.edges
    ]

    return integer_edges


def _get_variable_nodes(graph: Graph):
    variable_nodes = [
        i for i, node_label in enumerate(graph.nodes.keys()) if "va_" in node_label
    ]
    return variable_nodes


def _find_input_nodes(graph: Graph, last_node_id):
    # Create a reverse adjacency list from the graph
    integer_edges = _convert_to_integer_representation(graph)
    variable_nodes_bool = _get_variable_nodes(graph)

    reverse_adjacency_list = {}
    for source, target in integer_edges:
        if target not in reverse_adjacency_list:
            reverse_adjacency_list[target] = []
        reverse_adjacency_list[target].append(source)

    # Perform DFS to find all input nodes
    input_nodes = set()
    stack = [last_node_id]

    while stack:
        node = stack.pop()
        if node not in input_nodes:
            if node not in variable_nodes_bool:
                input_nodes.add(node)
                if node in reverse_adjacency_list:
                    stack.extend(reverse_adjacency_list[node])

    sorted_subgraph = _get_subgraph(graph, list(input_nodes))
    node_list = []
    for node_label in sorted_subgraph.nodes.keys():
        graph_labels = _get_node_labels(graph)
        node_list.append(graph_labels.index(node_label))

    return node_list


def _get_subgraph(graph: Graph, node_indices):
    edges = graph.edges
    subgraph_nodes = graph.nodes.iloc(node_indices)
    subgraph_edges = NestedList(obj_type=GraphEdge)
    integer_edges = _convert_to_integer_representation(graph)
    for i, (id_source, id_target) in enumerate(integer_edges):
        if id_source in node_indices and id_target in node_indices:
            subgraph_edges.append(edges[i])

    subgraph = Graph(nodes=subgraph_nodes, edges=subgraph_edges, label="subgraph")
    sorted_subgraph = topological_sort(subgraph)

    return sorted_subgraph


def topological_sort(graph: Graph):
    # Kahn's algorithm for topological sorting
    # Create a graph and in-degree count for each node

    sort_graph = defaultdict(list)
    edges = _convert_to_integer_representation(graph)
    nodes = range(len(graph.nodes))

    in_degree = {node: 0 for node in nodes}

    # Build the graph and count in-degrees
    for edge in edges:
        n_i, n_j = edge
        sort_graph[n_i].append(n_j)
        in_degree[n_j] += 1

    # Initialize queue with nodes having 0 in-degree
    queue = [node for node in nodes if in_degree[node] == 0]

    result = []
    while queue:
        node = queue.pop(0)
        result.append(node)

        # Reduce in-degree of adjacent nodes
        for neighbor in sort_graph[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    # Check if there's a cycle
    if len(result) != len(nodes):
        return None  # Graph has a cycle

    sorted_nodes = graph.nodes.iloc(result)
    sorted_graph = Graph(nodes=sorted_nodes, edges=graph.edges, label=graph.label)

    return sorted_graph


@as_function_node
def identity(x):
    return x


# find in a pandas dataframe whether the target and tagetHandle column match a given target and targetHandle
def _find_target_edge(graph: Graph, target, targetHandle):
    df = graph.edges.df
    edges = df.loc[(df["target"] == target) & (df["targetHandle"] == targetHandle)]
    return edges


def _find_source_edge(graph: Graph, source, sourceHandle):
    df = graph.edges.df
    edges = df.loc[(df["source"] == source) & (df["sourceHandle"] == sourceHandle)]
    return edges


# def _is_connected(graph: Graph, port: Port) -> bool:
#     connected = len(_is_connected(graph, port.node.label, port.label)) > 0
#     return connected


# def get_graph_from_wf(wf: "Workflow", wf_label: str = None) -> Graph:
#     graph = Graph()

#     if wf_label is None:
#         wf_label = wf.label

#     for label, node in wf._nodes.items():
#         graph = add_node(graph, node, label=label)

#         data = node.inputs.data
#         changed_args = _different_indices(data["default"], data["value"])

#         # construct the input nodes for the non-default arguments
#         for i in changed_args:
#             value = data["value"][i]
#             handle = data["label"][i]
#             if not isinstance(value, (Node, Port)):
#                 inp_node_label = f"var_i_{wf_label}__{handle}"
#                 edge = GraphEdge(
#                     target=label,
#                     targetHandle=handle,
#                     source=inp_node_label,
#                     sourceHandle="x",  # input label of identity node
#                 )

#                 graph += edge
#                 graph += identity(
#                     label=inp_node_label
#                 )  # GraphNode(id=inp_node_label, import_path=value)

#     for edge in wf._edges:
#         graph += GraphEdge(**edge)

#     for label, node in wf._nodes.items():
#         # construct the output nodes for all output ports
#         data = node.outputs.data
#         for i, handle in enumerate(data["label"]):
#             if len(_find_source_edge(graph, label, handle)) > 0:
#                 continue
#             out_node_label = f"var_o_{wf_label}__{handle}"
#             edge = GraphEdge(
#                 source=label,
#                 sourceHandle=handle,
#                 target=out_node_label,
#                 targetHandle="x",  # input label of identity node
#             )

#             graph += edge
#             graph += identity(label=out_node_label)

#     sorted_graph = topological_sort(graph)
#     return sorted_graph


def update_input_values(graph: Graph, node_label: str, values: list):
    node: Node = graph.nodes[node_label].node
    for i, value in enumerate(values):
        handle = node.inputs.data["label"][i]
        update_input_value(graph, node_label, handle, value)

    return graph


def update_input_value(
    graph: Graph,
    node_label: str,
    handle: str,
    value: Union[str, int, float, Node, Port],
) -> Graph:
    node = graph.nodes[node_label].node
    index = handle  #  node.inputs.data["label"].index(handle)
    node.inputs.data["value"][index] = value
    if not node.inputs.data["ready"][index] and str(
        node.inputs.data["default"][index]
    ) != str(
        value
    ):  # TODO: check if value type is correct
        node.inputs.data["ready"][index] = True

    return graph


# def get_code_from_graph(
#     graph: Graph,
#     workflow_lib="pyiron_workflow",
#     pyiron_nodes_lib="pyiron_nodes",
# ):
#     """
#     Generate Python source code from workflow graph.

#     Args:
#         label (str): The label to use in the generated code.
#         module_a (str): The name of the module to import from.
#         module_b (str): The name of the module to import.

#     Returns:
#         str: The generated Python source code.
#     """
#     import black

#     code = f"""
# from {workflow_lib} import Workflow
# import {pyiron_nodes_lib}

# wf = Workflow('{graph.label}')

# """

#     # Add nodes to Workflow
#     for key, node in graph.nodes.items():
#         label, import_path = key, node.import_path
#         if not label.startswith("var_i_"):
#             code += f"""wf.{label} = {import_path}("""

#             # Add edges
#             first_edge = True
#             for edge in graph.edges:
#                 if edge.target == label:
#                     if first_edge:
#                         first_edge = False
#                     else:
#                         code += """, """
#                     if edge.source.startswith("var_i_"):
#                         if edge.sourceHandle.startswith("__str_"):
#                             code += f"""{edge.targetHandle}='{edge.sourceHandle[6:]}'"""
#                         else:
#                             code += f"""{edge.targetHandle}={edge.sourceHandle}"""
#                     else:
#                         source_node = graph.nodes[edge.source]  # wf._nodes[edge.source]
#                         if source_node.node.n_out_labels == 1:
#                             code += f"""{edge.targetHandle}=wf.{edge.source}"""
#                         else:
#                             code += f"""{edge.targetHandle}=wf.{edge.source}.outputs.{edge.sourceHandle}"""
#             code += f""") \n"""
#             # code += '\n' + 'print(wf.run()) \n'

#     formatted_code = black.format_str(code, mode=black.FileMode())

#     return formatted_code


def get_code_from_wf(wf: "Workflow"):
    """Generate Python source code from pyiron_workflow"""

    graph = get_graph_from_wf(wf)

    code = get_code_from_graph(graph)

    return code


def run_wf(wf, debug=False):
    graph = get_graph_from_wf(wf)
    variable_nodes = _get_variable_nodes(graph)
    for i, node_label in enumerate(graph.nodes.keys()):
        if i not in variable_nodes:
            if debug:
                print(f"running node {node_label}")
            out = wf._nodes[node_label].run()

    return out


def _get_node_labels(graph: Graph):
    """
    Retrieve list of the labels of all nodes in the workflow graph.

    Args:
        graph (Graph): The workflow graph containing nodes.

    Returns:
        list: A list of node labels.
    """
    return [node_label for node_label in graph.nodes.keys()]


def pull_node(graph: Graph, node_label: str):
    """
    Pull a node from the workflow graph and run it. Execute only nodes that
    are required as input to run the node.

    Args:
        graph (Graph): The graph containing the node.
        node_label (str): The label of the node to pull.
    """
    # TODO: implement
    # opt_graph = _optimize_graph_connections(graph)
    opt_graph = deepcopy(graph)
    node_labels = _get_node_labels(opt_graph)
    if node_label not in node_labels:
        raise ValueError(f"Node label '{node_label}' not found in the workflow graph.")
    node_index = node_labels.index(node_label)
    input_nodes = _find_input_nodes(opt_graph, node_index)
    input_nodes_labels = [node_labels[i] for i in input_nodes]

    for input_node_label in input_nodes_labels:
        out = opt_graph.nodes[input_node_label].node.run()
    return out


# def graph_edges_to_wf_edges(graph_edges: List[Tuple[str, str]]):
#     wf_edges = []
#     for edge in graph_edges:
#         edge_source, edge_target = edge
#         target, target_handle = edge_target.split("/")
#         source, source_handle = edge_source.split("/")
#         if not source.startswith("var_i_"):
#             edge_dict = dict(
#                 source=source,
#                 sourceHandle=source_handle,
#                 target=target,
#                 targetHandle=target_handle,
#             )
#             wf_edges.append(edge_dict)
#     return wf_edges


# def get_wf_from_graph(graph: Graph) -> "Workflow":
#     from pyiron_workflow import Workflow

#     wf = Workflow(graph.label)
#     # Add nodes to Workflow
#     for node in graph.nodes:
#         label, import_path = node

#         if not label.startswith("var_i_"):
#             kwargs = dict()

#             # Add non-default arguments to node
#             for edge in graph.edges:
#                 edge_source, edge_target = edge
#                 target, target_handle = edge_target.split("/")
#                 source, source_handle = edge_source.split("/")
#                 if target == label:
#                     # TODO: get value from source node (is there not converted to string)
#                     if source.startswith("var_i_"):
#                         if source_handle.startswith("__str_"):
#                             kwargs[target_handle] = source_handle[6:]
#                         else:
#                             kwargs[target_handle] = eval(source_handle)
#                     # kwargs[target_handle] = source_handle

#             new_node = get_node_from_path(import_path)(**kwargs)
#             wf.add_node(label, new_node)
#     wf._set_edges(graph_edges_to_wf_edges(graph.edges))

#     return wf


def _save_graph(
    graph: Graph,
    filename: str | pathlib.Path = None,
    workflow_dir: str = ".",
    overwrite: bool = False,
):
    if filename is None:
        filename = f"{graph.label}.json"

    if isinstance(filename, str):
        # check if filename has extension json, if not add it
        if not filename.endswith(".json"):
            filename = f"{filename}.json"

        filename = pathlib.Path(filename)

    file = pathlib.Path(workflow_dir) / filename
    if file.exists() and not overwrite:
        raise FileExistsError(
            f"File '{filename}' already exists in dir {workflow_dir}."
        )

    with open(file, "w") as f:
        f.write(json.dumps((graph.__getstate__()), indent=4))

    return True


def _load_graph(filename: str | pathlib.Path, workflow_dir: str = "."):
    # check if filename has extension json, if not add it
    if isinstance(filename, str):
        if not filename.endswith(".json"):
            filename = f"{filename}.json"

    if isinstance(filename, str):
        filename = pathlib.Path(filename)

    wf_file = workflow_dir / filename
    if not wf_file.exists():
        raise FileNotFoundError(f"File '{filename}' not found in dir {workflow_dir}.")

    with open(wf_file, "r") as f:
        graph = Graph().__setstate__(json.load(f))

    return graph


####################################################################################################
# GUI related functions
####################################################################################################


def _to_jsonifyable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, Port):
        value = obj.value
        # print("value: ", obj._to_dict())
        if isinstance(value, (str, int, float, bool)):
            return value
        else:
            return NotData
    elif isinstance(obj, Node):
        return NotData
    elif isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    else:
        return NotData


def _is_connected(obj):
    return isinstance(obj, (Port, Node))


def gui_data(node: Node, key: str = None, expanded: bool = False) -> GuiData:

    label = key  # node.label
    # The following does not work since the label change is not reflected in the edges
    # if (node.label != key) and (key is not None):
    #     label = f"{node.label}_{key}"

    if node is None:
        return GuiData(label=label)

    target_values = [
        _to_jsonifyable(v) if not isinstance(v, Node) else NotData
        for v in node.inputs.data["value"]
    ]
    is_connected = [_is_connected(v) for v in node.inputs.data["value"]]

    # TODO: set to None if it contains an edge (include connected parameter)
    target_types = [
        "None" if (t == "builtins.NoneType") or connected else t
        for t, connected in zip(node.inputs.data["type"], is_connected)
    ]

    return GuiData(
        label=label,
        source_labels=node.outputs.data["label"],
        target_labels=node.inputs.data["label"],
        import_path=node.function["import_path"],
        target_values=target_values,
        target_types=target_types,
        source_values=[NotData for _ in node.outputs.data["value"]],
        source_types=node.outputs.data["type"],
        expanded=expanded,
    )


def _get_node_height(node: Node) -> int | float:
    if isinstance(node, Graph) or node is None:
        height = 250
    else:
        n_max_ports = max(node.n_out_labels, node.n_inp_labels)
        height = 30 + 16 * n_max_ports
    return height


def _nodes_to_gui(graph: Graph, remove_none=True) -> NestedList:
    node_width = 200

    nodes = NestedList()
    active_nodes = _get_active_nodes(graph)
    for i, (k, v) in enumerate(active_nodes.items()):
        # print("gui node: ", k, v.label, v.expanded)
        # print('node: ', k, v.label, v.node.label)
        node_dict = GuiNode(
            id=k,
            data=gui_data(v.node, key=k, expanded=v.expanded).asdict(
                remove_none=remove_none
            ),
            position=dict(x=i * (node_width + 20), y=0),
            style=GuiStyle(width=node_width, height=_get_node_height(v.node)).asdict(
                remove_none=remove_none
            ),
            targetPosition="left",
            sourcePosition="right",
            type=v.widget_type,
            expanded=v.expanded,
        )
        if v.expanded:
            node_dict["type"] = "customNode"
            node_dict["data"] = GuiData(label=v.label, expanded=True).asdict(
                remove_none=remove_none
            )
        if v.parent_id is not None:
            node_dict.parentId = v.parent_id
            node_dict.extent = "parent"
        if v.node_type == "graph":
            node_dict.type = "customNode"  # None
            node_dict.style["backgroundColor"] = "rgba(255, 165, 0, 0.3)"
        if v.label.startswith("var_"):
            node_dict.style["border"] = "1px black dashed"
            node_dict.style["backgroundColor"] = "rgba(50, 50, 50, 0.1)"
        if v.label.startswith("va_"):
            node_dict.style["border"] = "1px black dashed"
            node_dict.style["backgroundColor"] = "rgba(50, 50, 50, 0.1)"

        # if not v.expanded:  # for testing automated layout
        nodes.append(node_dict.asdict(remove_none=remove_none))

    return nodes


def get_child_dict(graph, node):
    if node["expanded"]:
        node_children = _gui_children(graph, node)
    targetPorts = [
        dict(id=f"{node['id']}_in_{label}", properties=dict(side="WEST"))
        for label in node["data"]["target_labels"]
    ][
        ::-1
    ]  # TODO: provide port positions x, y (this is only a quick fix)
    sourcePorts = [
        dict(id=f"{node['id']}_out_{label}", properties=dict(side="EAST"))
        for label in node["data"]["source_labels"]
    ][::-1]
    child = dict(
        id=node["id"],
        width=node["style"]["width"],
        height=node["style"]["height"],
        properties={"org.eclipse.elk.portConstraints": "FIXED_ORDER"},
        ports=[*targetPorts, *sourcePorts],
    )

    return child


def _gui_children(graph, gui_node):
    children = NestedList()
    nodes = _nodes_to_gui(graph, remove_none=False)  # TODO: cache it, avoid recomputing
    for node in nodes:
        node_children = []
        if node["parentId"] == gui_node["id"]:
            child = get_child_dict(graph, node)

            child["parent"] = 1  # level in the graph? no documentation
            if len(node_children) > 0:
                child["children"] = node_children
            children.append(child)

    return children


def _graph_to_gui(graph: Graph, remove_none=True, optimize=True) -> dict:
    layoutOptions = {
        "elk.algorithm": "layered",
        "elk.direction": "RIGHT",
        "elk.layered.spacing.edgeNodeBetweenLayers": "40",
        "elk.spacing.nodeNode": "40",
        "elk.layered.nodePlacement.strategy": "SIMPLE",
        "elk.hierarchyHandling": "INCLUDE_CHILDREN",
    }
    graph_dict = dict(id="root", layoutOptions=layoutOptions)

    nodes = _nodes_to_gui(graph, remove_none=remove_none)
    edges = _edges_to_gui(graph, remove_none=remove_none)
    children = []
    for node in nodes:
        if not "parentId" in node.keys():  # TODO: make this recursive
            child = get_child_dict(graph, node)
            node_children = _gui_children(graph, node)
            if len(node_children) > 0:
                child["children"] = node_children
            children.append(child)

    elk_edges = NestedList()
    for edge in edges:
        elk_edges.append(
            dict(
                id=edge["id"],
                source=edge["source"],
                target=edge["target"],
                sourcePort=f"{edge['source']}_out_{edge['sourceHandle']}",
                targetPort=f"{edge['target']}_in_{edge['targetHandle']}",
            )
        )

    graph_dict["children"] = children
    graph_dict["edges"] = elk_edges
    return graph_dict


def display_gui_data(graph):
    data = _nodes_to_gui(graph, remove_none=False).df.data
    return pd.DataFrame(transpose_list_of_dicts(data))


def display_gui_style(graph):
    style = _nodes_to_gui(graph, remove_none=False).df["style"]
    return pd.DataFrame(transpose_list_of_dicts(style))


def _edges_to_gui(graph, remove_none=True):
    edges = NestedList()
    active_edges = _get_active_edges(graph)
    for i, edge in enumerate(active_edges):
        edge_dict = edge.asdict(remove_none=remove_none)
        edge_dict["id"] = i
        edge_dict["style"] = {"strokeWidth": 2, "stroke": "black"}

        edges.append(edge_dict)

    return edges


class GuiGraph:
    def __init__(self, graph: Graph, optimze=True, sleep=0.5):
        self.graph = graph
        self.optimze = optimze
        self._reactflow_widget_status = "ina"
        self._sleep = sleep

    def on_value_change(self, change):
        # print("print command: ", change["new"])
        command, node_name = change["new"].split(":")
        if command == "finished":
            self._reactflow_widget_status = "done"
            # print("done")

    def _update_graph_view(self, w):
        import time
        import json

        w.observe(self.on_value_change, names="commands")
        self._reactflow_widget_status = "running"
        # if self.optimze:
        #     opt_graph = _optimize_graph_connections(self.graph)
        # else:
        # TODO: implement
        opt_graph = copy_graph(self.graph)
        data = dict(
            #    label=graph.label,
            nodes=_nodes_to_gui(opt_graph),
            edges=_edges_to_gui(opt_graph),
            graph=_graph_to_gui(opt_graph),
        )
        time.sleep(0.2)

        w.mydata = json.dumps(data)

        time.sleep(self._sleep)  # wait to give the gui time to finalize the graph

    def _repr_html_(self):
        from IPython.display import display

        """
        Display the graph using the ReactFlowWidget.

        This method initializes a ReactFlowWidget, updates the graph view in a separate thread,
        and returns the widget for display.
        """
        import threading
        from pyironflow.reactflow import ReactFlowWidget

        w = ReactFlowWidget()

        if not hasattr(self, "_thread") or not self._thread.is_alive():
            self._thread = threading.Thread(target=self._update_graph_view, args=(w,))
            self._thread.start()
        return display(w)
