import unittest

import pyiron_workflow as pwf
from pyiron_workflow.graph import base

from static import nodes, other_nodes


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

    def test_group_node_name_conflicts(self):
        wf = pwf.Workflow("custom_nodes_in_a_group")
        wf.n1 = nodes.PassThrough(42)
        wf.n2 = other_nodes.PassThrough(wf.n1)
        g = base.get_full_graph_from_wf(wf)
        with self.subTest(
            msg="Two nodes with the same class name should be able to co-exist in the "
            "same group"
        ):
            g = base.create_group(g, [0, 1], label="subgraph")
            out = base.pull_node(g, "subgraph")
            self.assertEqual(
                out,
                42,
                msg="Just verifying the group is also operational"
            )
            print(g.nodes["subgraph"].node._code)