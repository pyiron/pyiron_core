# from ._version import get_versions

# __version__ = get_versions()["version"]

# User entry point
from pyiron_workflow.api import Workflow

# Node developer entry points
from pyiron_workflow.api import (
    as_function_node,
    as_inp_dataclass_node,
    as_out_dataclass_node,
    as_macro_node,
    Node,
    NotData,
    NOT_DATA,
    Port,
)
