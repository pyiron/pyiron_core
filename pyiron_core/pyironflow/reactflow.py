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

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

import pathlib

import anywidget
import traitlets

# import os


class ReactFlowWidget(anywidget.AnyWidget):
    # path = pathlib.Path(os.getcwd()) / 'static'
    path = pathlib.Path(__file__).parent.parent.parent / "static"
    _esm = path / "widget.js"
    _css = path / "widget.css"
    nodes = traitlets.Unicode("[]").tag(sync=True)
    edges = traitlets.Unicode("[]").tag(sync=True)
    graph = traitlets.Unicode("[]").tag(
        sync=True
    )  # graph for automated layout using ELK
    mydata = traitlets.Unicode("[]").tag(
        sync=True
    )  # dictionary containing nodes, edges, and graph
    commands = traitlets.Unicode("[]").tag(sync=True)
