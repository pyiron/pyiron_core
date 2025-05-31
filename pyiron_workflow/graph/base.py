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

import copy
import dataclasses
import importlib
from typing import Union, List, Tuple

from pyiron_workflow.graph.decorators import (
    as_dotdict_dataclass,
    NestedDict,
    NestedList,
    get_import_path_from_type,
)
from pyiron_workflow.graph.edges import GraphEdge, Edges
from pyiron_workflow.simple_workflow import Data, Node, Port, Workflow, as_function_node


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
            self.node = (
                None if state["node_type"] == "graph" else Node().__setstate__(v)
            )
        elif k == "graph":
            if v is not None:
                self.graph = Graph().__setstate__(v)
            # print("setting graph: ", v, state)
            # self.graph = Graph().__setstate__(v)
        else:
            self[k] = v

    if self.node is None:
        # print(f"node is None: {self}")
        if self.graph is not None and not self.graph.label.startswith("va_"):
            self.label = self.graph.label
            self.node = graph_to_node(self.graph)

    if self.node is not None:
        # print(f"node is not None: set NodeGraph {self.node.label}, {self.label}")
        self.node._graph_node = self

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


# Nodes = NestedDict[str, GraphNode]
# Edges = NestedList[str, GraphEdge]
class Nodes(NestedDict):
    def __init__(self, obj_type=GraphNode):
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
    self.edges = Edges().__setstate__(state["edges"])

    if "graph" in state:
        self.graph = Graph().__setstate__(state["graph"])
    else:
        self.graph = dict()
    if "root_node" in state:
        self.root_node = GraphNode().__setstate__(state["root_node"])
    self.nodes = Nodes().__setstate__(state["nodes"])
    # update node inputs according to edges
    get_updated_graph(self)

    # instantiate virtual macros in node.node using node.graph
    for key, node in self.nodes.items():
        if node is not None and node["node"] is None and not key.startswith("va_"):
            # print(f"key: {key}, node: {node}")
            graph = node.graph
            node.node = graph_to_node(graph)
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
    nodes: Nodes = dataclasses.field(
        default_factory=lambda: NestedDict(obj_type=GraphNode)
    )
    edges: Edges = dataclasses.field(
        default_factory=lambda: NestedList(obj_type=GraphEdge)
    )
    graph: dict = dataclasses.field(default_factory=lambda: {})


def copy_nodes(nodes: Nodes) -> Nodes:

    new_nodes = Nodes(obj_type=GraphNode)
    for k, node in nodes.items():
        new_nodes[k] = GraphNode(
            **{
                kk: node[kk] if kk in ("node", "graph") else copy.copy(node[kk])
                for kk in node.keys()
            }
        )
    return new_nodes


def copy_graph(graph: Graph) -> Graph:
    return Graph(
        label=graph.label,
        nodes=copy_nodes(graph.nodes),
        edges=copy.deepcopy(graph.edges),
    )


def add_node(
    graph: "Graph", node: Union["Node", "Graph", "GraphNode"], label=None
) -> "Graph":
    label = _get_label(node, label)
    label = get_unique_label(graph, label)
    _check_label_exists(graph, label)

    node.label = label

    if isinstance(node, Node):
        if node.node_type == "macro_node":
            print(f"Adding macro node {label}")
            macro_graph = get_graph_from_macro_node(node)
            new_graph = _add_graph_instance(
                graph,
                macro_graph,
                label,
                node,
            )
        else:
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
        if graph.nodes[label].node_type == "graph":
            # remove all child nodes of this macro node
            for node_label in graph.nodes.keys():
                if graph.nodes[node_label].parent_id == label:
                    new_graph = remove_node(new_graph, node_label)
        del new_graph.nodes[label]

    edges_to_remove = [
        edge for edge in graph.edges if edge.source == label or edge.target == label
    ]
    for edge in edges_to_remove:
        new_graph.edges.remove(edge)

    # TODO: remove node from connected ports?

    return new_graph


def remove_edge(graph: Graph, edge: GraphEdge) -> Graph:
    edge = _rewire_edge(graph, edge)
    new_graph = copy_graph(graph)
    if edge in graph.edges:
        new_graph.edges.remove(edge)
    else:
        raise ValueError(f"Edge {edge} not found in graph")
    return new_graph


def _get_label(node, label):
    if label is None:
        label = node.label
    return label


def get_unique_label(graph: Graph, label: str) -> str:
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


def _add_graph_instance(graph: Graph, sub_graph: Graph, label: str = None, node=None):
    new_graph = copy_graph(graph)
    sub_graph.id = label
    # print('sub_graph: ', sub_graph.id, sub_graph.label)

    if node is None:
        if sub_graph.root_node is None:
            sub_graph = update_execution_graph(sub_graph)
            sub_graph_node = graph_to_node(sub_graph)
        else:
            sub_graph_node = sub_graph.root_node

        node = sub_graph_node
        # node = sub_graph.root_node
        import_path = None
    else:
        import_path = get_import_path_from_type(node._func)
    new_graph.nodes[label] = GraphNode(
        id=label,
        import_path=import_path,
        node=node,
        graph=sub_graph,
        label=label,
        node_type="graph",
        widget_type="customNode",
    )
    return new_graph


def _rewire_edge(graph: Graph, input_edge: GraphEdge) -> GraphEdge:
    edge = copy.copy(input_edge)
    source_node = graph.nodes[edge.source]
    target_node = graph.nodes[edge.target]
    if target_node.node_type == "graph":
        if source_node.parent_id == target_node.parent_id:
            edge.target = f"va_i_{edge.target}__{edge.targetHandle}"
            edge.targetHandle = "x"
            print(f"Rewiring edge to {edge.target}__{edge.targetHandle}")
    if source_node.node_type == "graph":
        if source_node.parent_id == target_node.parent_id:
            edge.source = f"va_o_{edge.source}__{edge.sourceHandle}"
            edge.sourceHandle = "x"
            print(f"Rewiring edge to {edge.source}__{edge.sourceHandle}")
    return edge


def add_edge(
    graph: Graph,
    source: str = None,
    target: str = None,
    sourceHandle: str = None,
    targetHandle: str = None,
):
    edge = _rewire_edge(graph, GraphEdge(source, target, sourceHandle, targetHandle))
    new_graph = copy_graph(graph)
    new_graph.edges.append(edge)
    if not (is_virtual_node(source) or is_virtual_node(target)):
        new_graph = _update_target_port(new_graph, new_graph.edges[-1])
    return new_graph


def _update_target_port(graph: Graph, edge: GraphEdge):
    source_port = graph.nodes[edge.source].node.outputs.__getattr__(edge.sourceHandle)
    graph.nodes[edge.target].node.inputs.__setattr__(edge.targetHandle, source_port)
    # print(f"updated target port {edge.targetHandle} in node {edge.target}, {source_port}")
    return graph


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
    active_nodes = NestedDict(obj_type=GraphNode)
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
    new_graph = copy_graph(graph)
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
    # else:
    #     print(f"Node {node_label} has no source nodes")

    del new_graph.nodes[node_label]
    return new_graph


####################################################################################################
# Transform to and from Graph, Workflow and Code
####################################################################################################


def graph_edges_to_wf_edges(edges: Edges) -> List[dict]:
    wf_edges = []
    for edge in edges:
        if not edge.source.startswith("va_i_") and not edge.target.startswith("va_o_"):
            wf_edges.append(edge.asdict())
    return wf_edges


def get_wf_from_graph(graph: Graph) -> Workflow:
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


def _build_input_argument_string(k, v, first_arg, as_string=True):
    code = ""
    if first_arg:
        first_arg = False
    else:
        code += """, """

    if isinstance(v, str) and as_string:
        code += f"""{k}='{v}'"""
    else:
        code += f"""{k}={v}"""

    return code, first_arg


def get_code_from_graph(
    graph: Graph,
    workflow_lib: str = "pyiron_workflow",
    pyiron_nodes_lib: str = "pyiron_nodes",
    sort_graph: bool = True,
    include_non_default_inputs: bool = True,
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
    from pyiron_workflow.graph.to_code import get_code_from_graph

    return get_code_from_graph(
        graph,
        sort_graph=sort_graph,
        use_node_default=include_non_default_inputs,
        scope_inputs=False,
        enforced_node_library=None,  # pyiron_nodes_lib
        # can't enforce for dataclasses: https://github.com/JNmpi/pyiron_core/issues/80
    )


def get_graph_from_wf(
    wf: Workflow,
    wf_outputs: Tuple[Node | Port],
    out_labels: List[str],
    wf_label: str = None,
) -> Graph:
    if wf_label is None:
        wf_label = wf.label

    # print("wf_label: ", wf_label)
    graph = Graph(label=wf_label)

    for label, node in wf._nodes.items():
        # TODO: node input changes due to rewiring edges!
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

        # print("target: ", target, target_handle)
        graph += edge

    sorted_graph = topological_sort(graph)
    return sorted_graph


def run_macro_node(macro_node):
    macro_graph = get_graph_from_macro_node(macro_node)

    output_nodes = list()
    output_labels = dict()
    for edge in macro_graph.edges:
        if f"va_o_{macro_node.label}__" in edge.target:
            output_nodes.append(edge.source)
            o_label = edge.target.split("__")[-1]
            output_labels[o_label] = (edge.source, edge.sourceHandle)

    # connect inner node input directly with outer node, eliminate in execution macro input port
    for graph_node in macro_graph.nodes.values():
        values = graph_node.node.inputs.data["value"]
        labels = graph_node.node.inputs.data["label"]
        for port_label, port_value in zip(labels, values):
            # print('label: ', port_label)
            if isinstance(port_value, (Port)):
                # print(port_label, type(port_value.value))
                if isinstance(port_value.value, (Port)):
                    # print('double: ', port_value.value.label, port_value.value.node.label)
                    graph_node.node.inputs.__setattr__(port_label, port_value.value)

    outputs = list()
    # output_labels = macro_node.outputs.data["label"]
    for out_label in set(output_nodes):
        # print(f"output node {out_label} of macro {macro_node.label}")
        outputs.append(
            pull_node(macro_graph, out_label)
        )  # use graph theory to avoid recalculating nodes (or use ready)

    if len(outputs) == 1:
        return outputs[0]  # works only for nodes with single output
    else:
        outputs = list()
        for label in macro_node.outputs.data["label"]:
            # print(f"output label {label}")
            o_source, o_handle = output_labels[label]
            out = macro_graph.nodes[o_source].node.outputs.__getattr__(o_handle)
            outputs.append(out.value)

        # raise NotImplementedError("Multiple outputs not yet implemented. Sort sequence by macro output labels.")
        return outputs


def run_node(node: Node | GraphNode, **kwargs):
    """
    Executes a given node and returns its output.
    Parameters:
        node (Node | GraphNode): The node to be executed. It can either be an instance of `Node`
                                 or `GraphNode`. If it is a `GraphNode`, its underlying `node`
                                 attribute is used for execution. If it is a `Node`, a copy of
                                 the node is created for execution.
        **kwargs: Additional keyword arguments to be passed as inputs to the node. These are
                  added to the node's inputs before execution.
    Returns:
        Any: The result of the node's execution. If an error occurs during execution, `None`
             is returned.
    Raises:
        TypeError: If the provided `node` is neither a `Node` nor a `GraphNode`.
    Notes:
        - If an exception occurs during the execution of the node, it is caught, and an error
          message is printed. The function then returns `None`.
    """
    if isinstance(node, GraphNode):
        node_to_run = node.node
    elif isinstance(node, Node):
        node_to_run = node.copy()
    else:
        raise TypeError(f"Unexpected node type {type(node)}")

    for key, value in kwargs.items():
        node_to_run.inputs[key] = value
    try:
        result = node_to_run.run()
    except Exception as e:
        print(f"An error occurred while running the node: {e}")
        result = None

    return result


def get_graph_from_macro_node(macro_node: Node) -> Graph:
    orig_values = dict()
    kwargs = {}
    for inp in macro_node.inputs.data["label"]:
        inp_port_label = f"va_i_{macro_node.label}__{inp}"
        kwargs[inp] = inp_port_label
        orig_values[inp_port_label] = macro_node.inputs.__getattr__(inp)

    out = macro_node._func(**kwargs)
    if isinstance(out, tuple):
        out_0 = out[0]
    else:
        out_0 = out
        out = (out,)

    # each output instance contains link to workflow, check that it works for multiple outputs
    if isinstance(out[0], Port):
        out_0 = out_0.node
    wf = out_0._workflow
    # print("label: ", wf.label, macro_node.label)
    wf.label = macro_node.label

    out_labels = macro_node.outputs.data["label"]

    new_graph = get_graph_from_wf(
        wf, wf_outputs=out, out_labels=out_labels, wf_label=macro_node.label
    )
    # restore original kwargs

    # print("new_graph: ", new_graph.label)
    for node in new_graph.nodes.values():
        # iterate over all non-virtual nodes
        if not is_virtual_node(node.label):
            for i, value in enumerate(node.node.inputs.data["value"]):
                if value in orig_values:
                    # print(f"Setting value {value} to {orig_values[value]}")
                    node.node.inputs.data["value"][i] = orig_values[value]

    return new_graph


####################################################################################################
# Collapse and Expand Graphs and Macro Nodes
####################################################################################################


def is_virtual_node(node_label: str) -> bool:
    return node_label.startswith("va_")


def handle_to_port_label(handle: str) -> str:
    if is_virtual_node(handle):
        path_list = handle[len("va_i_") :].split("__")
        # print(f"path_list: {path_list}")
        if len(path_list) > 2:
            return "__".join(path_list[1:])
        return handle.split("__")[-1]
    return handle


def handle_to_node_label(handle: str) -> str:
    if is_virtual_node(handle):
        return handle.split("__")[-2]


def handle_to_parent_label(handle: str) -> str:
    if is_virtual_node(handle):
        return handle[len("va_i_") :].split("__")[0]


def _is_parent_in_node_label(label: str, parent_label: str) -> bool:
    if parent_label is None:
        return False

    if is_virtual_node(label):
        return handle_to_parent_label(label) == parent_label

    return False


def move_parent_nodes_to_top(graph):
    # reorder parent nodes that they are before their children
    # this is necessary for elk to work properly
    # get a list of the node labels in correct order
    node_labels = []
    reordered_nodes = []
    for node in graph.nodes.values():
        if node.parent_id is not None:
            if node.label not in reordered_nodes and node.parent_id not in node_labels:
                node_labels.append(node.parent_id)
                reordered_nodes.append(node.parent_id)
        if node.label not in node_labels:
            node_labels.append(node.label)

    print(reordered_nodes)
    new_nodes = Nodes(obj_type=GraphNode)
    # print("sub_graph22: ", "_obj_type" in new_nodes.__getstate__())
    for label in node_labels:
        new_nodes[label] = graph.nodes[label]
    new_graph = copy_graph(graph)
    new_graph.nodes = new_nodes
    return new_graph


def _node_labels_to_node_ids(graph: Graph, node_labels: List[str]) -> List[str]:
    ind_dict = dict()
    for ind, label in enumerate(graph.nodes.keys()):
        ind_dict[label] = ind

    return [ind_dict[label] for label in node_labels]


def create_group(full_graph, node_ids=[], label=None):
    full_graph = copy_graph(full_graph)
    sub_graph = _get_subgraph(full_graph, node_ids, label)
    sub_graph_node = graph_to_node(sub_graph)

    # print("sub_graph: ", sub_graph.label, "_obj_type" in full_graph.nodes.__getstate__())
    full_graph.nodes[sub_graph.label] = GraphNode(
        id=sub_graph.label,
        label=sub_graph.label,
        parent_id=None,
        graph=sub_graph,
        node_type="graph",
        node=sub_graph_node,
        widget_type="customNode",
        expanded=False,
    )
    # print("sub_graph1: ", sub_graph.label, "_obj_type" in full_graph.nodes.__getstate__())
    # print("sub_graph_node: ", full_graph.nodes[sub_graph.label])

    for node in sub_graph.nodes.values():
        full_graph.nodes[node.label].parent_id = sub_graph.label
        full_graph.nodes[node.label].level += 1

    add_edges = []
    for io_type in ["inputs", "outputs"]:
        values = getattr(sub_graph_node, io_type).data["value"]
        labels = getattr(sub_graph_node, io_type).data["label"]
        print("labels", labels)
        for handle, value in zip(labels, values):
            handle = f"va_{io_type[0]}_{sub_graph.label}__{handle}"
            # handle = f"va_{io_type[0]}_{handle}"
            full_graph += identity(label=handle)
            full_graph.nodes[handle].parent_id = sub_graph.label
            if io_type[0] == "i":
                target_node, target_handle = handle.split("__")[1:]
                print("inp: ", target_node, target_handle)
                edge = GraphEdge(
                    source=handle,
                    target=target_node,
                    sourceHandle="x",
                    targetHandle=target_handle,
                )
                add_edges.append(edge)
                print(edge)

    # rewire connections to external output nodes
    node_ports = get_externally_connected_input_ports(sub_graph)
    for node, handle in node_ports:
        # print(node, handle)
        for edge in full_graph.edges:
            if edge.target == node and edge.targetHandle == handle:
                new_edge = copy.copy(edge)
                new_edge.target = f"va_i_{sub_graph.label}__{edge.targetHandle}"
                new_edge.targetHandle = "x"
                add_edges.append(new_edge)

    # print("sub_graph1b: ", sub_graph.label, "_obj_type" in full_graph.nodes.__getstate__())
    # print("Sub_Graph_node: ", full_graph.nodes[sub_graph.label])
    # rewire connections to external input nodes
    for key, node in full_graph.nodes.items():
        print("node: ", key, node)
        marker = f"va_o_{sub_graph.label}__"
        if marker in node.label:
            # print("virtual output node", node.label)
            source_node, source_handle = node.label[len(marker) :].split("__")
            # print(source_node, source_handle)
            for edge in full_graph.edges:
                if edge.source == source_node:
                    new_edge = copy.copy(edge)
                    edge.source = (
                        f"va_o_{sub_graph.label}__{source_node}__{edge.sourceHandle}"
                    )
                    edge.sourceHandle = "x"
                    new_edge.target = f"va_o_{sub_graph.label}__{source_node}__{new_edge.sourceHandle}"
                    new_edge.targetHandle = "x"
                    add_edges.append(new_edge)

    for edge in add_edges:
        full_graph.edges.append(edge)

    # print("sub_graph1c: ", sub_graph.label, "_obj_type" in full_graph.nodes.__getstate__())
    full_graph = move_parent_nodes_to_top(full_graph)
    # print("sub_graph2: ", sub_graph.label, "_obj_type" in full_graph.nodes.__getstate__())
    return full_graph


def _remove_virtual_nodes(
    graph: Graph, reconnect_edges=False, parent_label: str = None
) -> Graph:
    # first collect all va_nodes in a list, then remove it from the graph
    new_graph = copy_graph(graph)
    va_list = [
        node_label
        for node_label in graph.nodes.keys()
        if _is_parent_in_node_label(node_label, parent_label)
    ]

    for node_label in va_list:
        if reconnect_edges:
            new_graph = remove_node_with_reconnected_edges(new_graph, node_label)
        else:
            del new_graph.nodes[node_label]

    return new_graph


def _remove_virtual_edges(graph: Graph) -> Graph:
    new_graph = copy_graph(graph)
    edges_to_remove = [
        edge
        for edge in graph.edges
        if is_virtual_node(edge.source) or is_virtual_node(edge.target)
    ]
    for edge in edges_to_remove:
        new_graph.edges.remove(edge)
    return new_graph


def _remove_edges_to_hidden_nodes(graph: Graph) -> Graph:
    new_graph = copy_graph(graph)
    edges_to_remove = []
    for edge in graph.edges:
        if not (edge.source in graph.nodes and edge.target in graph.nodes):
            edges_to_remove.append(edge)

    for edge in edges_to_remove:
        new_graph.edges.remove(edge)

    return new_graph


def get_updated_graph(full_graph: Graph, level: int = 0) -> Graph:
    """
    Get the updated graph after marking nodes as expanded or collapsed.
    """
    graph = copy_graph(full_graph)
    for node in full_graph.nodes.values():
        if node.level == 0 and node.graph is not None:
            if node.expanded:
                graph = expand_node(graph, node.label)
            else:
                graph = collapse_node(graph, node.label)

    graph = _remove_virtual_edges(graph)
    graph = _remove_edges_to_hidden_nodes(graph)
    graph = update_execution_graph(graph)
    return graph


def expand_node(
    full_graph: Graph, node_label: str, remove_virtual_nodes: bool = True
) -> Graph:
    new_graph = copy_graph(full_graph)
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
        if remove_virtual_nodes:
            new_graph = _remove_virtual_nodes(
                new_graph, reconnect_edges=True, parent_label=node_label
            )

    return new_graph


def remove_hidden_nodes(graph: Graph, node_label: str) -> Graph:
    new_graph = copy_graph(graph)
    nodes_to_remove = []
    for node in new_graph.nodes.values():
        if node.parent_id == node_label:
            if not new_graph.nodes[node.parent_id].expanded:
                nodes_to_remove.append(node.label)

    for node_label in nodes_to_remove:
        del new_graph.nodes[node_label]

    return new_graph


def collapse_node(
    graph: Graph, node_label: str, remove_virtual_nodes: bool = True
) -> Graph:
    new_graph = copy_graph(graph)
    graph_node = new_graph.nodes[node_label]

    if graph_node.node_type == "graph":
        # print(f"Collapsing node {node_label}")
        graph_node.expanded = False
        for edge in new_graph.edges:
            if _is_parent_in_node_label(edge.source, node_label):
                source = handle_to_parent_label(edge.source)
                edge.sourceHandle = handle_to_port_label(edge.source)
                edge.source = source
                # print(f"rewiring edge {edge.source}/{edge.sourceHandle}")
            if _is_parent_in_node_label(edge.target, node_label):
                target = handle_to_parent_label(edge.target)
                edge.targetHandle = handle_to_port_label(edge.target)
                edge.target = target

        if remove_virtual_nodes:
            new_graph = _remove_virtual_nodes(
                new_graph, reconnect_edges=False, parent_label=node_label
            )

        # TODO: make the following recursive, i.e., also remove children of nodes in the collapsed graph
        new_graph = remove_hidden_nodes(new_graph, node_label)

    return new_graph


def get_full_graph_from_wf(wf: Workflow) -> Graph:
    graph = Graph(label=wf.label)

    macro_node_labels = []
    for label, node in wf._nodes.items():
        if node.node_type == "macro_node":
            node.label = label
            graph += node
            # new_node = get_graph_from_macro_node(node)
            # graph = add_node(graph, new_node, label=label)
            # graph.nodes[node.label].node = node
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
from typing import Union


from typing import TYPE_CHECKING, List, Tuple

if TYPE_CHECKING:
    from pyiron_workflow import Workflow


def get_ports_of_node_type(graph: Graph) -> List[Port]:
    ports = []
    for node in graph.nodes.values():
        inds = node.inputs.data["type"].select("Node")
        for ind in inds:
            ports += node.inputs.data["value"][ind]
    return ports


def _different_indices(default, value):
    # TODO: quick fix, use _get_non_default_inputs from simple_workflow.py
    return [
        i
        for i in range(len(default))
        if (str(default[i]) != str(value[i])) or (str(value[i]) in (NotData))
    ]


def get_non_default_input(
    graph: Graph, exclude_connections=False, flatten=False
) -> dict:
    nodes = dict()
    node_port_list = []  # list of tuples (node_label, port_label)
    for node in graph.nodes.values():
        data = node.node.inputs.data
        changed_args = _different_indices(data["default"], data["value"])
        node_dict = dict()
        for i in changed_args:
            if not (exclude_connections and isinstance(data["value"][i], (Node, Port))):
                node_dict[data["label"][i]] = data["value"][i]
                node_port_list.append((node.label, data["label"][i]))

        if node_dict:
            nodes[node.label] = node_dict

    if flatten:
        return node_port_list
    return nodes


def get_unconnected_ports(graph: Graph, port_type: str) -> List[Tuple[str, str]]:
    not_connected_ports = []
    for node in graph.nodes.values():
        if node.node_type == "graph":
            continue
        ports = (
            node.node.inputs.data["label"]
            if port_type == "input"
            else node.node.outputs.data["label"]
        )
        for port_label in ports:
            connected = False
            for edge in graph.edges:
                if (
                    port_type == "input"
                    and edge.target == node.label
                    and edge.targetHandle == port_label
                ) or (
                    port_type == "output"
                    and edge.source == node.label
                    and edge.sourceHandle == port_label
                ):
                    connected = True
                    break
            if not connected:
                not_connected_ports.append((node.label, port_label))
    return not_connected_ports


def get_unconnected_output_ports(graph: Graph) -> List[Tuple[str, str]]:
    return get_unconnected_ports(graph, "output")


def get_node_output_port(node: Node, port_label: str) -> Port:
    for port in node.node.outputs.data["label"]:
        if port == port_label:
            return node.node.outputs.__getattr__(port_label)
    return None


def get_externally_connected_input_ports(graph):
    external_ports = []
    for node in graph.nodes.values():
        ports = node.node.inputs.data["label"]
        values = node.node.inputs.data["value"]
        for port_label, value in zip(ports, values):
            if is_port_external_to_graph(value, graph):
                external_ports.append((node.label, port_label))

    return external_ports


def get_unconnected_input_ports(graph: Graph) -> List[Tuple[str, str]]:
    return get_unconnected_ports(graph, "input")


def get_inputs_of_graph(graph: Graph, exclude_unconnected_default_ports=False) -> Data:
    if exclude_unconnected_default_ports:
        include_ports = get_non_default_input(
            graph, exclude_connections=True, flatten=True
        ) + get_externally_connected_input_ports(graph)

    labels, values, types, default, ready = [], [], [], [], []
    for node_label, port_label in get_unconnected_input_ports(graph):
        node = graph.nodes[node_label]
        port = get_node_input_port(node, port_label)
        # ensure that label is unique
        # if port_label in labels:
        #   port_label = f"{node_label}__{port_label}"
        if (
            exclude_unconnected_default_ports
            and (node_label, port_label) not in include_ports
        ):
            continue

        port_label = f"{node_label}__{port_label}"
        labels.append(port_label)
        values.append(port.value)
        types.append(port.type)
        default.append(port.default)
        ready.append(port.ready)

    return Data(
        dict(label=labels, value=values, type=types, default=default, ready=ready),
        attribute=Port,
    )


def is_port_external_to_graph(value: any, graph: Graph) -> bool:
    """
    Check if a port is external to the graph.

    Args:
        value: The value of the port.
        graph: The graph to check against.

    Returns:
        bool: True if the port is external, False otherwise.
    """
    has_value_attr = hasattr(value, "value")
    is_external_node = isinstance(value, Node) and value.label not in graph.nodes
    is_external_port = isinstance(value, Port) and value.node.label not in graph.nodes
    return has_value_attr and (is_external_node or is_external_port)


def get_node_input_port(node: Node, port_label: str) -> Port:
    for port in node.node.inputs.data["label"]:
        if port == port_label:
            return node.node.inputs.__getattr__(port_label)
    return None


def get_outputs_of_graph(graph: Graph) -> Data:
    labels, values, types, ready = [], [], [], []
    for node_label, port_label in get_unconnected_output_ports(graph):
        node = graph.nodes[node_label]
        port = get_node_output_port(node, port_label)
        # ensure that label is unique
        # if port_label in labels:
        port_label = f"{node_label}__{port_label}"
        labels.append(port_label)
        values.append(port.value)
        types.append(port.type)
        ready.append(port.ready)

    return Data(
        dict(label=labels, value=values, type=types, ready=ready), attribute=Port
    )


def _find_node_inputs(graph: Graph) -> List[Port]:
    node_inputs = []
    for graph_node in graph.nodes.values():
        node_inp_types = graph_node.node.inputs.data["type"]
        if "Node" in node_inp_types:
            indices = [i for i, x in enumerate(node_inp_types) if x == "Node"]
            target = graph_node.node.label
            for i in indices:
                target_handle = graph_node.node.inputs.data["label"][i]
                node_inputs.append((target, target_handle))
                # print(target, target_handle, i)

    return node_inputs


def _remove_node_inputs(graph: Graph) -> Graph:
    new_graph = copy_graph(graph)
    node_inputs = _find_node_inputs(graph)
    for node, handle in node_inputs:
        for edge in new_graph.edges:
            if edge.target == node and edge.targetHandle == handle:
                new_graph.edges.remove(edge)

    return new_graph


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
        i
        for i, node_label in enumerate(graph.nodes.keys())
        if is_virtual_node(node_label)
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


def graph_to_code(graph):
    graph = get_code_from_graph(graph, sort_graph=True)
    return graph


def graph_to_node(graph: Graph, exclude_unconnected_default_ports=True) -> Node:
    from pyiron_workflow.graph.to_code import (
        get_code_from_graph,
    )

    # print("graph_to_node: ", _build_function_parameters(graph, use_node_default=False))
    function_string = get_code_from_graph(graph, use_node_default=False)

    # Create a dictionary to serve as the local namespace
    virtual_namespace = {}

    # Execute the function string in the local namespace
    # print("function_string: ", function_string)
    exec(function_string, globals(), virtual_namespace)

    # Retrieve the function from the local namespace
    func = virtual_namespace[graph.label]

    node = Node(
        func=func,
        label=graph.label,
        node_type="graph",
        inputs=get_inputs_of_graph(graph, exclude_unconnected_default_ports=True),
        outputs=get_outputs_of_graph(graph),
    )
    node.label = graph.label  # should not be necessary
    node._code = function_string  # TODO: add macro decorator with output labels
    node.graph = graph

    return node


def _get_subgraph(graph: Graph, node_indices, label=None) -> Graph:
    # collapse all nodes that are in the subgraph
    # TODO: remove child nodes in subgraph of collapsed nodes
    graph = copy_graph(graph)
    for subgraph_node in graph.nodes.iloc(node_indices):
        # print(f"Collapsing node {subgraph_node}", type(subgraph_node))
        graph.nodes[subgraph_node].expanded = False

    edges = graph.edges
    subgraph_nodes = graph.nodes.iloc(node_indices)

    subgraph_edges = NestedList(obj_type=GraphEdge)
    integer_edges = _convert_to_integer_representation(graph)
    for i, (id_source, id_target) in enumerate(integer_edges):
        if id_source in node_indices and id_target in node_indices:
            subgraph_edges.append(edges[i])

    if label is None:
        label = "subgraph"
    subgraph = Graph(
        nodes=subgraph_nodes, edges=subgraph_edges, label=get_unique_label(graph, label)
    )
    sorted_subgraph = topological_sort(subgraph)

    return sorted_subgraph


def topological_sort(graph: Graph) -> Graph:
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


def update_input_values(graph: Graph, node_label: str, values: list):
    node: Node = graph.nodes[node_label].node
    for i, value in enumerate(values):
        handle = node.inputs.data["label"][i]
        update_input_value(graph, node_label, handle, value)

    return graph


def update_input_value(
    graph: Graph,
    node_label: str,
    handle: str | int,
    value: Union[str, int, float, Node, Port],
) -> Graph:
    node = graph.nodes[node_label].node
    if isinstance(handle, int):
        index = handle
    else:
        index = node.inputs.data["label"].index(handle)
    node.inputs.data["value"][index] = value
    if not node.inputs.data["ready"][index] and str(
        node.inputs.data["default"][index]
    ) != str(
        value
    ):  # TODO: check if value type is correct
        node.inputs.data["ready"][index] = True

    return graph


def update_execution_graph(graph: Graph, debug=False) -> Graph:
    """
    Update the execution graph after changing node expansions or collapses
    """
    # graph = copy_graph(graph)
    for edge in graph.edges:
        source_node = graph.nodes[edge.source]

        # update source output ports
        data = source_node.node.outputs.data
        data["node"] = [source_node.node for _ in data["label"]]

        if debug:
            print(
                f"Updating input {source_node.label} in node {edge.source}",
                edge.sourceHandle,
                edge.targetHandle,
                id(source_node.node),
                id(source_node.node.outputs.data["node"][0]),
            )

        graph = update_input_value(
            graph,
            edge.target,
            edge.targetHandle,
            source_node.node.outputs.__getattr__(edge.sourceHandle),
        )
        # print(f"Updated input {graph.nodes[edge.target].label} in node {edge.target}", graph.nodes[edge.target].node.inputs)

    return graph


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
    opt_graph = copy_graph(graph)

    # closures are not part of the execution pipeline (called inside the function!)
    opt_graph = _remove_node_inputs(opt_graph)
    node_labels = _get_node_labels(opt_graph)
    if node_label not in node_labels:
        raise ValueError(f"Node label '{node_label}' not found in the workflow graph.")
    node_index = node_labels.index(node_label)
    input_nodes = _find_input_nodes(opt_graph, node_index)
    input_nodes_labels = [node_labels[i] for i in input_nodes]

    for input_node_label in input_nodes_labels:
        print(f"Running node {input_node_label}")
        out = opt_graph.nodes[input_node_label].node.run()
    return out


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
