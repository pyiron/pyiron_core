"""
Provide functions that are needed for pyironFlow, but that should be provided by
pyiron_core.pyiron_workflows in the end.
"""

import importlib
import typing

import numpy as np

from pyiron_core.pyironflow.themes import get_color

NotData = "NotData"


def get_import_path(obj):
    module = obj.__module__ if hasattr(obj, "__module__") else obj.__class__.__module__
    # name = obj.__name__ if hasattr(obj, "__name__") else obj.__class__.__name__
    name = obj.__name__ if "__name__" in dir(obj) else obj.__class__.__name__
    path = f"{module}.{name}"
    if path == "numpy.ndarray":
        path = "numpy.array"
    return path


def dict_to_node(dict_node, log):
    data = dict_node["data"]

    if "target_values" in data:
        target_values = data["target_values"]
        target_labels = data["target_labels"]
        target_dict = dict()
        for k, v in zip(target_labels, target_values, strict=True):
            if v not in ("NonPrimitive", NotData):
                target_dict[k] = v

    node = get_node_from_path(data["import_path"], log=log)(
        label=dict_node["id"], **target_dict
    )
    if "position" in dict_node:
        x, y = dict_node["position"].values()
        node.position = (x, y)
        # print('position exists: ', node.label, node.position)
    else:
        print("no position: ", node.label)

    return node


def dict_to_edge(dict_edge, nodes):
    out = nodes[dict_edge["source"]].outputs[dict_edge["sourceHandle"]]
    inp = nodes[dict_edge["target"]].inputs[dict_edge["targetHandle"]]
    inp.connect(out)

    return True


def is_primitive(obj):
    primitives = (bool, str, int, float, type(None))
    return isinstance(obj, primitives)


def get_node_values(channel_dict):
    values = list()
    for k, v in channel_dict.items():
        value = v.value
        if value != NotData:
            value = NotData
        elif not is_primitive(value):
            value = "NonPrimitive"

        values.append(value)

    return values


def _get_generic_type(t):
    non_none_types = [arg for arg in t.__args__ if arg is not type(None)]
    return float if float in non_none_types else non_none_types[0]


def _get_type_name(t):
    primitive_types = (bool, str, int, float, type(None))
    if t is None:
        return "None"
    elif t in primitive_types:
        return t.__name__
    else:
        return "NonPrimitive"


def get_node_types(node_io):
    node_io_types = list()
    for k in node_io.channel_dict:
        type_hint = node_io[k].type_hint
        if isinstance(type_hint, typing._UnionGenericAlias):
            type_hint = _get_generic_type(type_hint)

        node_io_types.append(_get_type_name(type_hint))
    return node_io_types


def get_node_position(node, id_num, node_width=200, y0=100, x_spacing=30):
    if "position" in dir(node):
        x, y = node.position
        # if isinstance(x, str):
        #     x, y = 0, 0
    else:
        x = id_num * (node_width + x_spacing)
        y = y0

    return {"x": x, "y": y}


def _to_jsonifyable(obj):
    from pyiron_core.pyiron_workflow import Node, Port

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
    from pyiron_core.pyiron_workflow import Node, Port

    return isinstance(obj, (Port, Node))


def get_node_dict(node, id_num, key=None):
    from pyiron_core.pyiron_workflow import Node

    node_width = 200
    label = node.label
    if (node.label != key) and (key is not None):
        label = f"{node.label}: {key}"

    target_values = [
        _to_jsonifyable(v) if not isinstance(v, Node) else "NotData"
        for v in node.inputs.data["value"]
    ]
    is_connected = [_is_connected(v) for v in node.inputs.data["value"]]
    # print("is_connected: ", is_connected)
    # source_values = [v if not isinstance(v, Node) else 'NotData' for v in node.outputs.data['value']]
    source_values = [NotData for _ in node.outputs.data["value"]]
    # TODO: set to None if it contains an edge (include connected parameter)
    target_types = [
        "None" if (t == "builtins.NoneType") or connected else t
        for t, connected in zip(node.inputs.data["type"], is_connected, strict=True)
    ]
    import_path = node.function["import_path"]
    # print('import_path: ', import_path)

    return {
        "id": node.label,
        "data": {
            "label": label,
            "source_labels": node.outputs.data["label"],
            "target_labels": node.inputs.data["label"],
            "import_path": import_path,
            "target_values": target_values,
            "target_types": target_types,
            "source_values": source_values,
            "source_types": node.outputs.data["type"],
        },
        "position": get_node_position(node, id_num),
        "type": "customNode",
        "style": {
            "border": "1px black solid",
            "padding": 5,
            "background": get_color(node=node, theme="light"),
            "borderRadius": "10px",
            "width": f"{node_width}px",
        },
        "targetPosition": "left",
        "sourcePosition": "right",
    }


def get_nodes(wf):
    nodes = []
    for i, (key, node) in enumerate(wf._nodes.items()):
        # print(node_dict)
        nodes.append(get_node_dict(node, id_num=i, key=key))
    return nodes


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


def get_edges(wf):
    edges = []
    for ic, edge in enumerate(wf._edges):

        edge_dict = dict()
        edge_dict["source"] = edge["source"]
        edge_dict["sourceHandle"] = edge["sourceHandle"]
        edge_dict["target"] = edge["target"]
        edge_dict["targetHandle"] = edge["targetHandle"]
        edge_dict["id"] = ic
        edge_dict["style"] = {
            "strokeWidth": 2,
            "stroke": "black",
        }

        edges.append(edge_dict)
    return edges
