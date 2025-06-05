from pyiron_workflow import simple_workflow
from pyiron_workflow.graph import base, group


def run_macro_node(macro_node):
    macro_graph = base.get_graph_from_macro_node(macro_node)

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
            if isinstance(port_value, (simple_workflow.Port)):
                # print(port_label, type(port_value.value))
                if isinstance(port_value.value, (simple_workflow.Port)):
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


def run_node(node: simple_workflow.Node | base.GraphNode, **kwargs):
    """
    Executes a given node and returns its output.
    Parameters:
        node (simple_workflow.Node | base.GraphNode): The node to be executed. It can either be an instance of `simple_workflow.Node`
                                 or `base.GraphNode`. If it is a `base.GraphNode`, its underlying `node`
                                 attribute is used for execution. If it is a `simple_workflow.Node`, a copy of
                                 the node is created for execution.
        **kwargs: Additional keyword arguments to be passed as inputs to the node. These are
                  added to the node's inputs before execution.
    Returns:
        Any: The result of the node's execution. If an error occurs during execution, `None`
             is returned.
    Raises:
        TypeError: If the provided `node` is neither a `simple_workflow.Node` nor a `base.GraphNode`.
    Notes:
        - If an exception occurs during the execution of the node, it is caught, and an error
          message is printed. The function then returns `None`.
    """
    if isinstance(node, base.GraphNode):
        node_to_run = node.node
    elif isinstance(node, simple_workflow.Node):
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


def run_wf(wf, debug=False):
    graph = base.get_graph_from_wf(wf)
    variable_nodes = group._get_variable_nodes(graph)
    for i, node_label in enumerate(graph.nodes.keys()):
        if i not in variable_nodes:
            if debug:
                print(f"running node {node_label}")
            out = wf._nodes[node_label].run()

    return out


def pull_node(graph: base.Graph, node_label: str):
    """
    Pull a node from the workflow graph and run it. Execute only nodes that
    are required as input to run the node.

    Args:
        graph (base.Graph): The graph containing the node.
        node_label (str): The label of the node to pull.
    """
    # TODO: implement
    # opt_graph = _optimize_graph_connections(graph)
    opt_graph = base.copy_graph(graph)

    # closures are not part of the execution pipeline (called inside the function!)
    opt_graph = base._remove_node_inputs(opt_graph)
    node_labels = group._get_node_labels(opt_graph)
    if node_label not in node_labels:
        raise ValueError(f"Node label '{node_label}' not found in the workflow graph.")
    node_index = node_labels.index(node_label)
    input_nodes = group._find_input_nodes(opt_graph, node_index)
    input_nodes_labels = [node_labels[i] for i in input_nodes]

    for input_node_label in input_nodes_labels:
        print(f"Running node {input_node_label}")
        out = opt_graph.nodes[input_node_label].node.run()
    return out
