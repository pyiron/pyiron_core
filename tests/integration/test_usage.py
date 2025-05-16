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
            "(i.e. `pyiron_nodes`).",
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
                out, 42, msg="Just verifying the group is also operational"
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

    def test_group_output_propagation(self):

        def make_graph() -> tuple[int, base.Graph, list[str]]:
            # DO NOT REUSE WORKFLOWS
            # https://github.com/JNmpi/pyiron_core/issues/45
            def make_workflow():
                wf = pwf.Workflow("group_returns")
                wf.n1 = nodes.AddOne(0)
                wf.n2 = nodes.AddOne(wf.n1)
                # DO NOT CONNECT n2 TO n3
                # https://github.com/JNmpi/pyiron_core/issues/44
                wf.n3 = nodes.AddOne()
                wf.n4 = nodes.AddOne(wf.n3)
                return wf

            wf = make_workflow()
            wf.n3.inputs.x = wf.n2
            g_connected = base.get_full_graph_from_wf(make_workflow())
            g_connected = base.add_edge(g_connected, "n2", "n3", "y", "x")
            expected_terminal_result = base.pull_node(g_connected, "n4")

            g = base.get_full_graph_from_wf(make_workflow())
            ordered_node_labels = list(g.nodes.keys())
            return (
                expected_terminal_result,
                g,
                ordered_node_labels,
            )

        with self.subTest("Upstream group"):
            expected_out, g, labels = make_graph()
            ids = base._node_labels_to_node_ids(g, labels[:2])
            g = base.create_group(g, ids, label="upstream_group")
            g = base.add_edge(g, "upstream_group", "n3", "n2__y", "x")
            self.assertEqual(
                expected_out,
                base.pull_node(base.get_updated_graph(g), labels[-1]),
                "Output from groups should propagate to downstream nodes",
            )

        with self.subTest("Downstream group"):
            expected_out, g, labels = make_graph()
            ids = base._node_labels_to_node_ids(g, labels[2:])
            g = base.create_group(g, ids, label="downstream_group")
            g = base.add_edge(g, "n2", "downstream_group", "y", "n3__x")
            self.assertEqual(
                expected_out,
                base.pull_node(base.get_updated_graph(g), "downstream_group"),
                "Output from groups should propagate to downstream nodes",
            )

        with self.subTest("Two groups"):
            expected_out, g, labels = make_graph()
            upstream_ids = base._node_labels_to_node_ids(g, labels[:2])
            g = base.create_group(g, upstream_ids, label="upstream_group")
            downstream_ids = base._node_labels_to_node_ids(g, labels[2:])
            g = base.create_group(g, downstream_ids, label="downstream_group")
            g = base.add_edge(g, "upstream_group", "downstream_group", "n2__y", "n3__x")
            self.assertEqual(
                expected_out,
                base.pull_node(base.get_updated_graph(g), "downstream_group"),
                "Output from groups should propagate to downstream nodes",
            )

    def test_edge_shorthand(self):
        def make_graph() -> tuple[int, base.Graph]:
            def make_workflow():
                wf = pwf.Workflow("mywf")
                wf.n1 = nodes.AddOne(0)
                wf.n2 = nodes.AddOne()
                return wf

            run_group = base.get_full_graph_from_wf(make_workflow())
            run_group = base.add_edge(run_group, "n1", "n2", "y", "x")
            expected_result = base.pull_node(run_group, "n2")

            g = base.get_full_graph_from_wf(make_workflow())
            g = base.create_group(g, [0], label="subgraph")
            return expected_result, g

        expected_result, explicit_graph = make_graph()
        explicit_graph = base.add_edge(
            explicit_graph, "va_o_subgraph__n1__y", "n2", "y", "x"
        )
        explicit_result = base.pull_node(base.get_updated_graph(explicit_graph), "n2")
        self.assertEqual(
            expected_result,
            explicit_result,
            msg="If a virtual node exists, we should be allowed to reference it",
        )

        _, implicit_graph = make_graph()
        implicit_graph = base.add_edge(implicit_graph, "subgraph", "n2", "n1__y", "x")
        implicit_result = base.pull_node(base.get_updated_graph(implicit_graph), "n2")
        self.assertEqual(
            expected_result,
            implicit_result,
            msg="If the user has created a group, we should allow them to reference its IO directly",
        )
