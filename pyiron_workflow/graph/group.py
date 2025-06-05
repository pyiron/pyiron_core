import copy

from pyiron_workflow.graph.base import (
    Graph,
    GraphNode,
    Nodes,
    _convert_to_integer_representation,
    copy_graph,
    get_externally_connected_input_ports,
    get_unique_label,
    is_virtual_node,
    graph_to_node,
    topological_sort,
)
from pyiron_workflow.graph.decorators import NestedList
from pyiron_workflow.graph.edges import GraphEdge
from pyiron_workflow.simple_workflow import identity


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


def _get_node_labels(graph: Graph):
    """
    Retrieve list of the labels of all nodes in the workflow graph.

    Args:
        graph (Graph): The workflow graph containing nodes.

    Returns:
        list: A list of node labels.
    """
    return [node_label for node_label in graph.nodes.keys()]


def _get_variable_nodes(graph: Graph):
    variable_nodes = [
        i
        for i, node_label in enumerate(graph.nodes.keys())
        if is_virtual_node(node_label)
    ]
    return variable_nodes
