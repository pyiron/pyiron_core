import hashlib
import json
import os
from collections.abc import Iterable
from typing import Any

from pyiron_core.pyiron_database.generic_storage import HDF5Storage, JSONGroup
from pyiron_core.pyiron_database.obj_reconstruction.util import (
    deserialize_obj,
    get_type,
    pickle_dump,
    pickle_load,
    recreate_obj,
    serialize_obj,
)
from pyiron_core.pyiron_workflow import NOT_DATA, Node
from pyiron_core.pyiron_workflow.api.graph import Graph, GraphEdge

from .InstanceDatabase import InstanceDatabase

PyironStoragePath: str = os.path.expanduser("~/pyiron_core/.storage")
# create storage path if it does not exist
if not os.path.exists(PyironStoragePath):
    os.makedirs(PyironStoragePath)


def store_node_outputs(node: Node, storage_path: str = PyironStoragePath) -> str:
    """
    Store a node's outputs into an HDF5 file.

    Args:
        node (Node): The node whose outputs should be stored.

    Returns:
        str: The file path where the node's outputs are stored.

    Raises:
        ValueError: If any output of the node is NOT_DATA.
    """
    node_hash = get_hash(node)
    output_path = f"{storage_path}/{node_hash}.hdf5"
    with HDF5Storage(output_path, "w") as storage:
        for k, v in node.outputs.items():
            is_default_check = v.value == v.default
            if isinstance(is_default_check, Iterable):
                if hasattr(is_default_check, "all"):
                    # special handling for pandas Dataframe needed
                    is_default_check_all = is_default_check.all()
                    if hasattr(is_default_check_all, "all"):
                        # special handling for pandas DataFrame (returns Series!)
                        is_default_check_all = is_default_check_all.all()
                    if is_default_check_all:
                        continue
                elif all(is_default_check):
                    continue
            elif is_default_check:
                continue

            if v.value is NOT_DATA:
                raise ValueError(f"Output '{k}' has no value.")
            try:
                storage[k] = pickle_dump(serialize_obj(v.value))
            except Exception as e:
                raise ValueError(
                    f"Failed to store output '{k}' of node '{node.label}': {e}"
                ) from e
    return output_path


def restore_node_outputs(node: Node, storage_path: str = PyironStoragePath) -> bool:
    """
    Restore a node's outputs from a stored HDF5 file, given by node.hash.

    Args:
        node (Node): the node whose outputs should be restored.

    Returns:
        bool: True if the outputs were restored, False if not.
    """

    import pathlib

    node_hash = get_hash(node)
    output_path = f"{storage_path}/{node_hash}.hdf5"
    print(
        f"Restoring node outputs  {node_hash} {node.label} {pathlib.Path(output_path).exists()}"
    )
    with HDF5Storage(output_path, "r") as storage:
        for k, v in storage.items():
            node.outputs[k]._value = deserialize_obj(
                node.outputs[k].type_hint(pickle_load(v))
            )
    return True


def recreate_node(
    module: str, qualname: str, version: str, init_args: dict[str, Any]
) -> Node:
    return recreate_obj(module, qualname, version, init_args)


def node_to_jsongroup(node: Node) -> JSONGroup:
    module, qualname, version = get_type(node)
    connected_inputs = [input.label for input in node.inputs if input.connected]
    json_group = JSONGroup({})
    json_group.update(
        {
            "inputs": node_inputs_to_jsongroup(node).data,
            "outputs": [o for o, _ in node.outputs.items()],
            "node": {
                "qualname": qualname,
                "module": module,
                "version": version,
                "connected_inputs": connected_inputs,
            },
        }
    )
    return json_group


def get_hash(obj_to_be_hashed: Node | JSONGroup) -> str:
    """
    Calculate the hash of a given node or JSONGroup.

    Args:
        obj_to_be_hashed (Node | JSONGroup): the object whose hash should be calculated.

    Returns:
        str: the SHA-256 hash of the object.
    """
    node_jsongroup = (
        obj_to_be_hashed
        if isinstance(obj_to_be_hashed, JSONGroup)
        else node_to_jsongroup(obj_to_be_hashed)
    )
    jsonified_dict = json.dumps(node_jsongroup.data, sort_keys=True)

    hasher = hashlib.sha256()
    hasher.update(jsonified_dict.encode("utf-8"))
    hash_value = hasher.hexdigest()

    return hash_value


def node_inputs_to_jsongroup(node: Node) -> JSONGroup:
    # TODO: check for input labels that should not be hashed such as “store", "_db", etc.
    def resolve_connections(value: Any) -> Any:
        if value.connected:
            return (
                get_hash(value.connections[0].owner) + "@" + value.connections[0].label
            )
        else:
            return value.value

    output = {k: resolve_connections(v) for k, v in node.inputs.items()}

    json_group = JSONGroup({})
    json_group.update(output)

    return json_group


def node_outputs_to_dict(node: Node) -> JSONGroup:
    output = JSONGroup({})
    output.update({k: v.value for k, v in node.outputs.items()})
    return output


def store_node_in_database(
    db: InstanceDatabase,
    node: Node,
    store_outputs: bool = False,
    store_input_nodes_recursively: bool = False,
) -> str:
    """
    Store a node in a database.

    This function stores all the information that is required to restore a node from the
    database. This includes the node's class, its inputs, its connected inputs and its
    outputs.

    Args:
        db (InstanceDatabase): The database to store the node in.
        node (Node): The node to store.
        store_outputs (bool): Whether to store the outputs of the node as well.
        store_input_nodes_recursively (bool): Whether to store all the nodes that are
            connected to the inputs of the node recursively.

    Returns:
        str: The hash of the stored node.
    """
    if store_input_nodes_recursively:
        connected_nodes = [
            input.connections[0].owner for input in node.inputs if input.connected
        ]
        for connected_node in connected_nodes:
            store_node_in_database(
                db,
                connected_node,
                store_outputs=store_outputs,
                store_input_nodes_recursively=store_input_nodes_recursively,
            )

    node_jsongroup = node_to_jsongroup(node)
    node_hash = get_hash(node_jsongroup)
    node_dict = node_jsongroup.data
    output_path = None
    if store_outputs:
        if "store" in node.inputs.keys():
            if node.inputs.store.value:
                output_path = store_node_outputs(node)
    if hasattr(node, "_hash_parent"):
        node_dict["node"]["master_hash"] = node._hash_parent
    else:
        node_dict["node"]["master_hash"] = None
    if hasattr(node, "_start_time"):
        node_dict["node"]["start_time"] = node._start_time
    if hasattr(node, "_execution_time"):
        node_dict["node"]["execution_time"] = node._execution_time
    if hasattr(node, "_user"):
        node_dict["node"]["user"] = node._user

    node_data = InstanceDatabase.NodeData(
        hash=node_hash,
        qualname=node_dict["node"]["qualname"],
        module=node_dict["node"]["module"],
        version=node_dict["node"]["version"],
        connected_inputs=node_dict["node"]["connected_inputs"],
        inputs=node_dict["inputs"],
        outputs=node_dict["outputs"],
        output_path=output_path,
        master_hash=node_dict["node"]["master_hash"],
        start_time=node_dict["node"].get("start_time", None),
        cpu_time=node_dict["node"].get("execution_time", None),
        user=node_dict["node"].get("user", None),
    )
    db.create(node_data)
    return node_hash


def restore_node_from_database(
    db: InstanceDatabase, node_hash: str, parent: Graph | None = None
) -> Node:
    """
    Restore a node from the database.

    The node is reconstructed from the database by calling recreate_node and
    adding it to the given parent workflow. The node's inputs are then restored
    either by connecting them to other nodes in the workflow or by setting their
    values directly.

    Args:
        db (InstanceDatabase): The InstanceDatabase instance to read from.
        node_hash (str): The hash of the node to restore.
        parent (Graph | None): The workflow to add the restored node to.

    Returns:
        Node: The restored node.

    Raises:
        RuntimeError: If the node with the given hash is not found in the database.
    """
    # restore node
    db_result = db.read(node_hash)
    if db_result is None:
        raise RuntimeError(f"Node with hash {node_hash} not found in database.")

    def generate_random_string(length: int = 20) -> str:
        import random
        import string

        letters = string.ascii_letters + string.digits
        return "".join(random.choice(letters) for i in range(length))

    node = recreate_node(
        module=db_result.module,
        qualname=db_result.qualname,
        version=db_result.version,
        init_args={
            "label": db_result.qualname
        },  # TODO: make sure that this is unique (use graph tool to generate a unique label)
    )

    if parent is None:
        parent = Graph()

    parent += node

    restored_inputs = JSONGroup(db_result.inputs)
    # restore inputs
    for k, v in restored_inputs.items():
        if k in db_result.connected_inputs:
            input_hash, input_label = v.split("@")
            input_node, parent = restore_node_from_database(db, input_hash, parent)
            parent += GraphEdge(
                target=node.label,
                targetHandle=k,
                source=input_node.label,
                sourceHandle=input_label,
            )
        else:
            node.inputs[k] = v

    return node, parent
