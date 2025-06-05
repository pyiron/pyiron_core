from pyiron_workflow import simple_workflow
from pyiron_workflow.graph import base, group


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
