"""
An API for the GUI module.

`pyiron_database` leverages model elements from `pyiron_workflow`, and the
`pyiron_workflow` GUI in turn leverages `pyiron_database`. To avoid circular imports
due to this, we split apart the back-end model and front-end GUI APIs.
This is a reasonable choice since, in the long run, the front-end should depend on the
back-end, but not vice-versa and so we will ultimately be able to split them into two
components.
"""

from pyiron_workflow.graph.gui import (
    GuiGraph,
    display_gui_data,
    _edges_to_gui,
    _nodes_to_gui,
    display_gui_style,
)
