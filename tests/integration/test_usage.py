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

    def test_multiple_groups(self):
        wf = pwf.Workflow("multiple_groups")
        wf.m1 = nodes.PassThrough(0)
        wf.m2 = nodes.PassThrough(wf.m1)
        wf.n1 = nodes.PassThrough(1)
        wf.n2 = nodes.PassThrough(wf.n1)

        g = base.get_full_graph_from_wf(wf)

        m_ids = base._node_labels_to_node_ids(g, ["m1", "m2"])
        g = base.create_group(g, m_ids, label="m_subgraph")

        n_ids = base._node_labels_to_node_ids(g, ["n1", "n2"])
        g = base.create_group(g, n_ids, label="n_subgraph")

        self.assertEqual(
            0,
            base.pull_node(g, "m_subgraph"),
            msg="Both groups should be pullable",
        )
        self.assertEqual(
            1,
            base.pull_node(g, "n_subgraph"),
            msg="Both groups should be pullable",
        )