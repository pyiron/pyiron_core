import textwrap

from pyiron_workflow import simple_workflow
from pyiron_workflow.graph import base, not_data, labelling


def port_to_code(port: base.Port, use_default: bool = False, scope: str = None):
    name = port.label if scope is None else _scope_label(scope, port.label)
    hint = "" if port.type in ("NotHinted", "NonPrimitive") else f": {port.type}"

    if port.value is not not_data.NotData and not use_default:
        value_str = simple_workflow.value_to_string(port.value)
    elif port.default is not not_data.NotData:
        value_str = simple_workflow.value_to_string(port.default)
    else:
        value_str = None

    space = " " if len(hint) > 0 else ""
    default = "" if value_str is None else f"{space}={space}{value_str}"

    return f"{name}{hint}{default}"


def _scope_label(scope: str, label: str, scope_delimiter: str = labelling.DELIM):
    return f"{scope}{scope_delimiter}{label}"


def get_code_from_graph(
    graph: base.Graph,
    sort_graph: bool = False,
    use_node_default: bool = False,
    scope_inputs: bool = True,
    enforced_node_library: str | None = None,
):
    """
    Generate Python source code from a graph representation.

    Args:
        graph (base.Graph): The graph object containing nodes and edges.
        sort_graph (bool): Whether to start by updating and topologically sorting the graph. (Default is False.)
        use_node_default (bool): Whether to prioritize the use of node default values over actual current values
        (if any) for defaults in the new macro. (Default is False, prefer to use current values if they are available.)
        scope_inputs (bool): Whether to include node labels ahead of arguments in signatures, i.e. to scope them. This
        is strictly necessary in the event that two nodes both used to populate the macro input have conflicting port
        labels. (Default is True, prepend arguments with node labels.)
        enforced_node_library (str | None): If provided, all nodes in the graph must have the given string at the start
        of their node's `import_path`. (Default is None, don't restrict node sources.)

    Returns:
        str: The generated Python source code as a string.
    """
    if sort_graph:
        graph = base.get_updated_graph(graph)
        graph = base.topological_sort(graph)

    kwargs = _build_function_parameters(
        graph, use_node_default=use_node_default, scope_labels=scope_inputs
    )
    returns, body_code = _process_nodes_and_edges(
        graph, scope_labels=scope_inputs, enforced_node_library=enforced_node_library
    )
    returns = returns if len(returns) > 0 else _get_default_return_args(graph)

    code = textwrap.dedent(
        f"""
    def {graph.label}({kwargs}):

        from pyiron_workflow import Workflow
        wf = Workflow('{graph.label}')

    """
    )
    code += body_code
    code += f"\n    return {', '.join(returns)}\n"

    return code


def _build_function_parameters(
    graph: base.Graph, use_node_default, scope_labels: bool = True
) -> str:
    """
    Build the function parameter string with type hints and default values.
    Args:
        graph (base.Graph): The graph object containing nodes and edges.
        use_node_default (bool): Whether to use node default values or actual node value as default value for macro.
    Returns:
        str: The function parameter string.
    """
    parameters = []
    seen_params = set()

    for node in graph.nodes.values():
        if labelling.is_virtual_input(node.label):
            inp = labelling.handle_to_parent_label(node.label)
            parameters.append((inp, None))  # No default value
            seen_params.add(inp)

            # Process edges for input nodes
            for edge in graph.edges:
                if edge.target == node.label:
                    # Handle edge processing if necessary
                    pass

    # Add non-default values
    non_default_inputs = base.get_non_default_input(graph)
    for node in graph.nodes.values():
        if node.label in non_default_inputs:
            for key, value in non_default_inputs[node.label].items():
                if not isinstance(value, (base.Node, base.Port)):
                    param_name = _scope_label(node.label, key) if scope_labels else key
                    if param_name in seen_params:
                        raise ValueError(
                            f'Duplicate parameter name "{param_name}" found when parsing node {node.label} in the graph {graph.label}; try activating scoping.'
                        )
                    seen_params.add(param_name)
                    port = base.get_node_input_port(node, key)
                    param = port_to_code(
                        port,
                        use_default=use_node_default,
                        scope=node.label if scope_labels else None,
                    )
                    value = port.default if use_node_default else port.value
                    param_has_default = None if value is not_data.NotData else True
                    parameters.append((param, param_has_default))

    # Sort parameters: args (no default) first, kwargs (with default) last
    parameters.sort(key=lambda x: x[1] is not None)

    # Return as a comma-separated string
    return ", ".join(param for param, _ in parameters)


def _process_nodes_and_edges(
    graph: base.Graph,
    scope_labels: bool = True,
    enforced_node_library: str | None = None,
) -> tuple[list[str], str]:
    """
    Process nodes and edges to build the workflow code.
    """
    code = ""
    return_args = []

    for node in (
        node for node in graph.nodes.values() if not labelling.is_virtual_node(node.label)
    ):
        if enforced_node_library is not None and not node.import_path.startswith(
            enforced_node_library
        ):
            raise ValueError(
                f"Only nodes from {enforced_node_library} are allowed during the conversion of the {graph.label} graph to code, but {node.label} has the import path {node.import_path}"
            )
        kwargs = dict()
        # Process edges for the current node
        for edge in graph.edges:
            if edge.target == node.label:
                if labelling.is_virtual_node(edge.source):
                    kwargs[edge.targetHandle] = edge.sourceHandle
                else:
                    if labelling.is_virtual_output(edge.target):
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
        non_default_inputs = base.get_non_default_input(graph)
        if node.label in non_default_inputs:
            for key, value in non_default_inputs[node.label].items():
                if not isinstance(value, (base.Node, base.Port)):
                    kwargs[key] = _scope_label(node.label, key) if scope_labels else key

        module_path, class_name = node.import_path.rsplit(".", 1)
        code += f"    from {module_path} import {class_name}\n"
        line = f"    wf.{node.label} = {class_name}("
        line += _dict_to_kwargs(kwargs) + ")\n"
        code += line

    return return_args, code


def _get_default_return_args(graph: base.Graph) -> list[str]:
    """
    Get default return arguments if none are specified.
    """
    outputs = base.get_unconnected_output_ports(graph)
    return [
        f"wf.{node_label}.outputs.{port_label}" for node_label, port_label in outputs
    ]


def _dict_to_kwargs(input_dict: dict) -> str:
    """
    Convert a dictionary into a string of keyword arguments.

    Args:
        input_dict (dict): The input dictionary.

    Returns:
        str: A string with the dictionary's key-value pairs formatted as kwargs.
    """
    return ", ".join(f"{key}={value}" for key, value in input_dict.items())


def graph_to_code(graph):
    graph = get_code_from_graph(graph, sort_graph=True)
    return graph
