import json
import pathlib

from pyiron_core.pyiron_workflow import simple_workflow
from pyiron_core.pyiron_workflow.graph import base


def _compact_graph(graph: base.Graph):
    """
    Compact the graph by collapsing macro nodes at the top level.
    This function iterates through the nodes in the graph and collapses
    any macro nodes that are at the top level (i.e., have no parent).
    It creates a new `GraphNode` for each macro node, setting its `expanded`
    attribute to `False` and copying the relevant properties from the original
    node. The graph is then updated to reflect these changes. Should be later moved
    to serialization module.
    Args:
        graph (base.Graph): The graph to compact.
    Returns:
        base.Graph: The compacted graph with macro nodes collapsed.
    """
    graph = base.copy_graph(graph)
    for k, node in graph.nodes.items():
        # find macro nodes in the top level and collapse them
        if (
            (node.graph is not None)
            and (node.parent_id is None)
            and (node.import_path is not None)
        ):
            # print("collapse: ", k)
            new_node = base.GraphNode(
                node=node.node,
                id=node.id,
                label=node.label,
                expanded=False,
                import_path=node.import_path,
                node_type=node.node_type,
            )
            graph.nodes[k] = new_node
            graph = base.collapse_node(graph, k)

    graph = base.get_updated_graph(graph)
    return graph


def _uncompact_graph_from_state(state: dict):
    """
    Uncompact the graph from its state representation.
    This function takes a state dictionary representing a graph and reconstructs
    the graph by creating `GraphNode` and `GraphEdge` objects from the state.
    Args:
        state (dict): The state representation of the graph.
    Returns:
        base.Graph: The reconstructed graph."""
    from pyiron_core.pyiron_workflow.graph import gui

    graph = base.Graph(label=state["label"])
    for k, node_state in state["nodes"].items():
        if isinstance(node_state, dict):
            # print(k, type(node_state))
            graph_node = base.GraphNode().__setstate__(node_state)
            if (graph_node.node is None) and (graph_node.import_path is not None):
                node = simple_workflow.Node().__setstate__(node_state["node"])
                graph = base.add_node(graph, node, label=node.label)
                graph = gui._mark_node_as_collapsed(graph, node.label)
            else:
                graph += graph_node
                if not graph_node.expanded:
                    # Otherwise += calls `base._expand_node` on us
                    graph = gui._mark_node_as_collapsed(graph, k)
                if graph_node.graph is not None:
                    graph = base.uncollapse_node(graph, k)

    for edge_state in state["edges"]["values"]:
        # print(edge_state)
        graph += base.GraphEdge(**edge_state)

    return graph


def _save_graph(
    graph: base.Graph,
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
        state = _compact_graph(graph).__getstate__()
        f.write(json.dumps(state, indent=4))

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
        state = json.load(f)
    graph = _uncompact_graph_from_state(state)

    return graph
