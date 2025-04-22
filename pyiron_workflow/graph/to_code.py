from pyiron_workflow.graph.base import (
    Graph,
    Node,
    Port,
    get_non_default_input,
    get_node_input_port,
    NotData,
    is_virtual_node,
    get_unconnected_output_ports,
    handle_to_parent_label,
)

from typing import Dict, List


def get_code_from_graph(
    graph: Graph,
    workflow_lib: str = "pyiron_workflow",
    pyiron_nodes_lib: str = "pyiron_nodes",
    use_node_default: bool = False,
) -> str:
    """
    Generate Python source code from a graph representation.

    Args:
        graph (Graph): The graph object containing nodes and edges.
        workflow_lib (str, optional): Name of the workflow library. Defaults to "pyiron_workflow".
        pyiron_nodes_lib (str, optional): Name of the PyIron nodes library. Defaults to "pyiron_nodes".
        use_node_default (bool, optional): Whether to use node default values or actual node value as
        default value for macro. Defaults to False.

    Returns:
        str: The generated Python source code as a string.
    """

    # Initialize function parameters
    kwargs = _build_function_parameters(graph, use_node_default=use_node_default)

    # Generate the function signature and initial code
    code = f"def {graph.label}({kwargs}):\n\n"
    code += f"    from {workflow_lib} import Workflow\n"
    code += f"    import {pyiron_nodes_lib}\n\n"
    code += f"    wf = Workflow('{graph.label}')\n\n"

    # Process nodes and edges to build the workflow
    return_args, code = _process_nodes_and_edges(graph, code)

    # Add return statement
    if not return_args:
        return_args = _get_default_return_args(graph)

    code += f"\n    return {', '.join(return_args)}\n"

    return code


def _build_function_parameters(graph: Graph, use_node_default) -> str:
    """
    Build the function parameter string with type hints and default values.
    Args:
        graph (Graph): The graph object containing nodes and edges.
        use_node_default (bool): Whether to use node default values or actual node value as default value for macro.
    Returns:
        str: The function parameter string.
    """
    parameters = []
    seen_params = set()

    for node in graph.nodes.values():
        if node.label.startswith("va_i_"):
            inp = handle_to_parent_label(node.label)
            parameters.append((inp, None))  # No default value
            seen_params.add(inp)

            # Process edges for input nodes
            for edge in graph.edges:
                if edge.target == node.label:
                    # Handle edge processing if necessary
                    pass

    # Add non-default values
    non_default_inputs = get_non_default_input(graph)
    for node in graph.nodes.values():
        if node.label in non_default_inputs:
            for key, value in non_default_inputs[node.label].items():
                if not isinstance(value, (Node, Port)):
                    param_name = f"{node.label}__{key}"
                    if param_name in seen_params:
                        raise ValueError(f"Duplicate parameter name: {param_name}")
                    seen_params.add(param_name)
                    port = get_node_input_port(node, key)
                    param = f"{param_name}: {port.type}"

                    if use_node_default:
                        value = port.default
                    else:
                        value = port.value

                    if value is not NotData:
                        if isinstance(value, str):
                            value = f'"{value}"'
                        param += f" = {value}"  # use actual value as default
                        parameters.append((param, True))  # Default value exists
                    else:
                        parameters.append((param, None))  # No default value

    # Sort parameters: args (no default) first, kwargs (with default) last
    parameters.sort(key=lambda x: x[1] is not None)

    # Return as a comma-separated string
    return ", ".join(param for param, _ in parameters)


def _process_nodes_and_edges(graph: Graph, base_code: str) -> List[str]:
    """
    Process nodes and edges to build the workflow code.
    """
    code = base_code
    return_args = []

    for node in graph.nodes.values():
        label, import_path = node.label, node.import_path
        # print("label: ", label, import_path)
        kwargs = dict()
        if not is_virtual_node(label):

            # Process edges for the current node
            for edge in graph.edges:
                if edge.target == label:
                    if is_virtual_node(edge.source):
                        kwargs[edge.targetHandle] = edge.sourceHandle
                    else:
                        if edge.target.startswith("va_o_"):
                            return_args.append(f"wf.{edge.source}")
                        else:
                            source_node = graph.nodes[edge.source]
                            if source_node.node.n_out_labels == 1:
                                kwargs[edge.targetHandle] = f"wf.{edge.source}"
                            else:
                                kwargs[edge.targetHandle] = (
                                    f"wf.{edge.source}.outputs.{edge.sourceHandle}"
                                )

            # Add non-default arguments
            non_default_inputs = get_non_default_input(graph)
            if label in non_default_inputs:
                for key, value in non_default_inputs[label].items():
                    if not isinstance(value, (Node, Port)):
                        kwargs[key] = f"{label}__{key}"
                        # if isinstance(value, str):
                        #     kwargs[key] = f'"{value}"'
                        # else: 
                        #     kwargs[key] = value

        line = f"    wf.{label} = {import_path}("
        line += _dict_to_kwargs(kwargs) + ")\n"
        code += line

    return return_args, code


def _get_default_return_args(graph: Graph) -> List[str]:
    """
    Get default return arguments if none are specified.
    """
    return_args = []
    outputs = get_unconnected_output_ports(graph)
    if len(outputs) == 1:
        node_label, _ = outputs[0]
        return [f"wf.{node_label}"]
    else:
        raise ValueError("Macro works only for single output node.")
    # uncomment if macro run issue is fixed
    # for node_label, port_label in outputs:
    #     return_args.append(f"wf.{node_label}.outputs.{port_label}")
    # return return_args


def _dict_to_kwargs(input_dict: dict) -> str:
    """
    Convert a dictionary into a string of keyword arguments.

    Args:
        input_dict (dict): The input dictionary.

    Returns:
        str: A string with the dictionary's key-value pairs formatted as kwargs.
    """
    return ", ".join(f"{key}={value}" for key, value in input_dict.items())
