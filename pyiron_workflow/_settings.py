from pyiron_workflow import as_dataclass_node


@as_dataclass_node
class Storage:
    hash_output: bool = False
