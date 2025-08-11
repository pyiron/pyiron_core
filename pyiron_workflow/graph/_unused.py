from typing import List

from pyiron_workflow import simple_workflow
from pyiron_workflow.graph import base, edges, group, labelling, run


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


def _find_source_edge(graph: base.Graph, source, sourceHandle):
    df = graph.edges.df
    edges = df.loc[(df["source"] == source) & (df["sourceHandle"] == sourceHandle)]
    return edges


def _find_target_edge(graph: base.Graph, target, targetHandle):
    df = graph.edges.df
    edges = df.loc[(df["target"] == target) & (df["targetHandle"] == targetHandle)]
    return edges


def get_ports_of_node_type(graph: base.Graph) -> List[simple_workflow.Port]:
    ports = []
    for node in graph.nodes.values():
        inds = node.inputs.data["type"].select("Node")
        for ind in inds:
            ports += node.inputs.data["value"][ind]
    return ports


def get_wf_from_graph(graph: base.Graph) -> simple_workflow.Workflow:
    wf = simple_workflow.Workflow(graph.label)
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

            new_node = base.get_node_from_path(import_path)(**kwargs)
            wf.add_node(label, new_node)
    wf._set_edges(graph_edges_to_wf_edges(graph.edges))

    return wf


def graph_edges_to_wf_edges(edges: edges.Edges) -> List[dict]:
    wf_edges = []
    for edge in edges:
        if not edge.source.startswith("va_i_") and not edge.target.startswith("va_o_"):
            wf_edges.append(edge.asdict())
    return wf_edges


def handle_to_node_label(handle: str) -> str:
    if labelling.is_virtual(handle):
        return handle.split("__")[-2]


def update_input_values(graph: base.Graph, node_label: str, values: list):
    node: simple_workflow.Node = graph.nodes[node_label].node
    for i, value in enumerate(values):
        handle = node.inputs.data["label"][i]
        base.update_input_value(graph, node_label, handle, value)

    return graph


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
            run.pull_node(macro_graph, out_label)
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


def run_wf(wf, debug=False):
    graph = base.get_graph_from_wf(wf)
    variable_nodes = group._get_variable_nodes(graph)
    for i, node_label in enumerate(graph.nodes.keys()):
        if i not in variable_nodes:
            if debug:
                print(f"running node {node_label}")
            out = wf._nodes[node_label].run()

    return out
