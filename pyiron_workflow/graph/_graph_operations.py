# Contains all the graph operations needed to convert graph to workflow and vice versa as well as to perform
# collaps and expand operations on macro nodes and graphs
import pyiron_workflow.graph.edges
from pyiron_workflow.simple_workflow import Data, get_node_from_path, Workflow
from pyiron_workflow import Node, Port, as_function_node
from pyiron_workflow.graph import base
from pyiron_workflow.graph.base import Graph
from copy import deepcopy

from typing import Tuple, List
import pyiron_nodes as pn


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


def get_externally_connected_input_ports(graph):
    external_ports = []
    for node in graph.nodes.values():
        ports = node.node.inputs.data["label"]
        values = node.node.inputs.data["value"]
        for port_label, value in zip(ports, values):
            if is_port_external_to_graph(value, graph):
                external_ports.append((node.label, port_label))

    return external_ports


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
    is_external_node = isinstance(value, base.Node) and value.label not in graph.nodes
    is_external_port = (
        isinstance(value, base.Port) and value.node.label not in graph.nodes
    )
    return has_value_attr and (is_external_node or is_external_port)


def get_unconnected_input_ports(graph: Graph) -> List[Tuple[str, str]]:
    return get_unconnected_ports(graph, "input")


def get_unconnected_output_ports(graph: Graph) -> List[Tuple[str, str]]:
    return get_unconnected_ports(graph, "output")


def get_node_input_port(node: Node, port_label: str) -> Port:
    for port in node.node.inputs.data["label"]:
        if port == port_label:
            return node.node.inputs.__getattr__(port_label)
    return None


def get_node_output_port(node: Node, port_label: str) -> Port:
    for port in node.node.outputs.data["label"]:
        if port == port_label:
            return node.node.outputs.__getattr__(port_label)
    return None


def get_inputs_of_graph(graph: Graph, exclude_unconnected_default_ports=False) -> Data:
    labels, values, types, default, ready = [], [], [], [], []
    for node_label, port_label in get_unconnected_input_ports(graph):
        node = graph.nodes[node_label]
        port = get_node_input_port(node, port_label)
        # ensure that label is unique
        if port_label in labels:
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


def get_outputs_of_graph(graph: Graph) -> Data:
    labels, values, types, ready = [], [], [], []
    for node_label, port_label in get_unconnected_output_ports(graph):
        node = graph.nodes[node_label]
        port = get_node_output_port(node, port_label)
        # ensure that label is unique
        if port_label in labels:
            port_label = f"{node_label}__{port_label}"
        labels.append(port_label)
        values.append(port.value)
        types.append(port.type)
        ready.append(port.ready)
        print("ready: ", port.ready)

    return Data(
        dict(label=labels, value=values, type=types, ready=ready), attribute=Port
    )


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
                new_edge = pyiron_workflow.graph.edges.GraphEdge(
                    inner_edge.source,
                    edge.target,
                    source_handle,
                    edge.targetHandle,
                )
                # print(f"Rewiring edge {edge} to {new_edge}")
                new_graph.edges.append(new_edge)

        base.remove_edge(new_graph, inner_edge)
    else:
        print(f"Node {node_label} has no source nodes")

    del new_graph.nodes[node_label]
    return new_graph


# very simple implementation, needs severe improvement/generalization
# specifically the ._func attribute needs to be updated to provide run functionality!
@as_function_node
def foo(structure=None, repeat_scalar=1, Repeat_2__repeat_scalar=1):
    return structure


def graph_to_node(graph: Graph) -> Node:
    foo_node = foo()
    node = Node(
        func=foo_node._func,
        label=graph.label,
        node_type="graph",
        inputs=get_inputs_of_graph(graph),
        outputs=get_outputs_of_graph(graph),
    )
    node.label = graph.label  # should not be necessary

    return node


####################################################################################################
# Transform to and from Graph, Workflow and Code
####################################################################################################


def graph_edges_to_wf_edges(edges: pyiron_workflow.graph.edges.Edges) -> List[dict]:
    wf_edges = []
    for edge in edges:
        if not edge.source.startswith("var_i_") and not edge.target.startswith(
            "var_o_"
        ):
            wf_edges.append(edge.asdict())
    return wf_edges


def get_wf_from_graph(graph: Graph) -> "Workflow":
    from pyiron_workflow import Workflow

    wf = Workflow(graph.label)
    # Add nodes to Workflow
    for node in graph.nodes.values():
        label, import_path = node.label, node.import_path

        if not label.startswith("var_i_"):
            kwargs = dict()

            # Add non-default arguments to node
            for edge in graph.edges:
                if edge.target == label:
                    # TODO: get value from source node (is there not converted to string)
                    if edge.source.startswith("var_i_"):
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
        graph = base.add_node(graph, node, label=label)

        data = node.inputs.data
        changed_args = base._different_indices(data["default"], data["value"])

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
                        graph += base.identity(label=inp_node_label)

                    edge = pyiron_workflow.graph.edges.GraphEdge(
                        target=label,
                        targetHandle=handle,
                        source=inp_node_label,
                        sourceHandle="x",  # input label of identity node
                    )

                    graph += edge

    for edge in wf._edges:
        graph += pyiron_workflow.graph.edges.GraphEdge(**edge)

    # print(f"Adding output nodes {out_labels}")
    for out_label, wf_output in zip(out_labels, wf_outputs):
        out_node_label = f"va_o_{wf_label}__{out_label}"
        graph += base.identity(label=out_node_label)

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

        edge = pyiron_workflow.graph.edges.GraphEdge(
            source=target,
            sourceHandle=target_handle,
            target=out_node_label,
            targetHandle="x",  # input label of identity node
        )

        print("target: ", target, target_handle)
        graph += edge

    sorted_graph = base.topological_sort(graph)
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
        new_edges = pyiron_workflow.graph.edges.Edges()
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
            graph = base.add_node(graph, new_node, label=label)
            graph.nodes[node.label].node = node
            macro_node_labels.append(label)
        else:
            graph = base.add_node(graph, node, label=label)

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

        graph += pyiron_workflow.graph.edges.GraphEdge(source, target, source_handle, target_handle)

    return graph
