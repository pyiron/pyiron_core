# from ._version import get_versions

# __version__ = get_versions()["version"]

# User entry point
from pyiron_core.pyiron_workflow.api.simple_workflow import Workflow

# Node developer entry points
from pyiron_core.pyiron_workflow.api.simple_workflow import (
    as_function_node,
    as_inp_dataclass_node,
    as_out_dataclass_node,
    as_macro_node,
    get_node_from_path,
    Node,
    NOT_DATA,
    Port,
)
