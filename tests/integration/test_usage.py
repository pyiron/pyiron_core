import unittest

import pyiron_workflow as pwf
from pyiron_workflow.graph import base

from static import nodes


class TestUsage(unittest.TestCase):
    """
    Verifying behavior for a variety of edge cases.
    """

    def test_local_nodes_in_groups(self):
        wf = pwf.Workflow("custom_nodes_in_a_group")
        wf.n = nodes.PassThrough(42)
        g = base.get_full_graph_from_wf(wf)
        g = base.create_group(g, [0], label="subgraph")
        out = base.pull_node(g, "subgraph")
        self.assertEqual(
            out,
            42,
            msg="It should be possible to put nodes into a group and pull them, even "
                "if those nodes do not belong to a special privileged node package"
                "(i.e. `pyiron_nodes`)."
        )