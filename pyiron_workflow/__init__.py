# from ._version import get_versions

# __version__ = get_versions()["version"]

from pyiron_workflow import simple_workflow
from pyiron_workflow.simple_workflow import NotData as NOT_DATA

# User entry point
from pyiron_workflow.simple_workflow import Workflow

# Node developer entry points
from pyiron_workflow.simple_workflow import (
    as_function_node,
    as_inp_dataclass_node,
    as_out_dataclass_node,
    as_macro_node,
    Node,
    Port,
    NotData,
)
