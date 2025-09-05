__author__ = "Joerg Neugebauer"
__copyright__ = (
    "Copyright 2024, Max-Planck-Institut for Sustainable Materials GmbH - "
    "Computational Materials Design (CM) Department"
)
__version__ = "0.2"
__maintainer__ = ""
__email__ = ""
__status__ = "development"
__date__ = "Aug 1, 2024"

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from pyiron_workflow import Workflow

from pyiron_workflow import hash_based_storage as hs
from pyironflow.wf_extensions import (
    get_nodes,
    get_edges,
    get_node_from_path,
    dict_to_node,
    dict_to_edge,
)

import anywidget
import pathlib
import traitlets

# import os
import json


class ReactFlowWidget(anywidget.AnyWidget):
    # path = pathlib.Path(os.getcwd()) / 'static'
    path = pathlib.Path(__file__).parent.parent / "static"
    _esm = path / "widget.js"
    _css = path / "widget.css"
    nodes = traitlets.Unicode("[]").tag(sync=True)
    edges = traitlets.Unicode("[]").tag(sync=True)
    graph = traitlets.Unicode("[]").tag(sync=True) # graph for automated layout using ELK
    mydata = traitlets.Unicode("[]").tag(sync=True) # dictionary containing nodes, edges, and graph
    commands = traitlets.Unicode("[]").tag(sync=True)
