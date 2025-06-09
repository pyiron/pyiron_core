import contextlib
import os
import unittest

import pyiron_workflow as pwf
from pyiron_workflow.graph import base, graph_json, group, run, labelling

from static import nodes, other_nodes


class TestUsage(unittest.TestCase):
    """
    Verifying behavior for a variety of edge cases.
    """

    def test_local_nodes_in_groups(self):
        data = 42
        wf = pwf.Workflow("custom_nodes_in_a_group")
        wf.n = nodes.Identity(data)
        g = base.get_full_graph_from_wf(wf)
        g = group.create_group(g, [0], label="subgraph")
        out = run.pull_node(base.get_updated_graph(g), "subgraph")
        self.assertEqual(
            out,
            data,
            msg="It should be possible to put nodes into a group and pull them, even "
            "if those nodes do not belong to a special privileged node package"
            "(i.e. `pyiron_nodes`).",
        )

    def test_group_node_name_conflicts(self):
        data = 42
        wf = pwf.Workflow("custom_nodes_in_a_group")
        wf.n1 = nodes.Identity(data)
        wf.n2 = other_nodes.Identity(wf.n1)
        g = base.get_full_graph_from_wf(wf)

        self.assertEqual(
            wf.n1._func.__name__,
            wf.n2._func.__name__,
            msg="Sanity check: The two nodes should have the same source name",
        )

        self.assertEqual(
            data,
            run.pull_node(base.get_updated_graph(g), "n1"),
            msg="Two nodes with the same source name should be able to co-exist in the "
            "same workflow/graph",
        )

        with self.subTest(
            msg="Two nodes with the same source name should be able to co-exist in the "
            "same group"
        ):
            g = group.create_group(g, [0, 1], label="subgraph")
            out = run.pull_node(base.get_updated_graph(g), "subgraph")
            self.assertEqual(
                out, data, msg="Just verifying the group is also operational"
            )
            print(g.nodes["subgraph"].node._code)

    def test_multiple_groups(self):
        m_data = 0
        n_data = 1
        wf = pwf.Workflow("multiple_groups")
        wf.m1 = nodes.Identity(m_data)
        wf.m2 = nodes.Identity(wf.m1)
        wf.n1 = nodes.Identity(n_data)
        wf.n2 = nodes.Identity(wf.n1)

        g = base.get_full_graph_from_wf(wf)

        g = group.create_group(g, ["m1", "m2"], label="m_subgraph")
        g = group.create_group(g, ["n1", "n2"], label="n_subgraph")

        self.assertEqual(
            m_data,
            run.pull_node(base.get_updated_graph(g), "m_subgraph"),
            msg="Both groups should be pullable",
        )
        self.assertEqual(
            n_data,
            run.pull_node(base.get_updated_graph(g), "n_subgraph"),
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
            expected_terminal_result = run.pull_node(
                base.get_updated_graph(g_connected), "n4"
            )

            g = base.get_full_graph_from_wf(make_workflow())
            ordered_node_labels = list(g.nodes.keys())
            return (
                expected_terminal_result,
                g,
                ordered_node_labels,
            )

        with self.subTest("Upstream group"):
            expected_out, g, labels = make_graph()
            g = group.create_group(g, labels[:2], label="upstream_group")
            g = base.add_edge(g, "upstream_group", "n3", "n2__y", "x")
            self.assertEqual(
                expected_out,
                run.pull_node(base.get_updated_graph(g), labels[-1]),
                "Output from groups should propagate to downstream nodes",
            )

        with self.subTest("Downstream group"):
            expected_out, g, labels = make_graph()
            g = group.create_group(g, labels[2:], label="downstream_group")
            g = base.add_edge(g, "n2", "downstream_group", "y", "n3__x")
            self.assertEqual(
                expected_out,
                run.pull_node(base.get_updated_graph(g), "downstream_group"),
                "Output from groups should propagate to downstream nodes",
            )

        with self.subTest("Two groups"):
            expected_out, g, labels = make_graph()
            g = group.create_group(g, labels[:2], label="upstream_group")
            g = group.create_group(g, labels[2:], label="downstream_group")
            g = base.add_edge(g, "upstream_group", "downstream_group", "n2__y", "n3__x")
            self.assertEqual(
                expected_out,
                run.pull_node(base.get_updated_graph(g), "downstream_group"),
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
            expected_result = run.pull_node(base.get_updated_graph(run_group), "n2")

            g = base.get_full_graph_from_wf(make_workflow())
            g = group.create_group(g, [0], label="subgraph")
            return expected_result, g

        expected_result, explicit_graph = make_graph()
        explicit_graph = base.add_edge(
            explicit_graph, f"{labelling.VOUTPUT}subgraph{labelling.DELIM}n1{labelling.DELIM}y", "n2", "y", "x"
        )
        explicit_result = run.pull_node(base.get_updated_graph(explicit_graph), "n2")
        self.assertEqual(
            expected_result,
            explicit_result,
            msg="If a virtual node exists, we should be allowed to reference it",
        )

        _, implicit_graph = make_graph()
        implicit_graph = base.add_edge(implicit_graph, "subgraph", "n2", "n1__y", "x")
        implicit_result = run.pull_node(base.get_updated_graph(implicit_graph), "n2")
        self.assertEqual(
            expected_result,
            implicit_result,
            msg="If the user has created a group, we should allow them to reference its IO directly",
        )

    def test_node_reinstantiation(self):
        fname = "node_reinstantiation.json"
        wf = pwf.Workflow(fname)
        wf.n = nodes.AddOne(0)
        g = base.get_full_graph_from_wf(wf)
        g = group.create_group(g, [0], label="subgraph")
        graph_json._save_graph(g, filename=fname)
        try:
            reloaded = graph_json._load_graph(fname)
            self.assertTrue(
                all(isinstance(n, pwf.Node) for n in reloaded.nodes.df["node"].values),
                msg="All reloaded nodes should get an accompanying python instance",
            )
        finally:
            with contextlib.suppress(FileNotFoundError):
                os.remove(fname)

    def test_macro_and_group_return_multiplicity_equivalence(self):
        wf = pwf.Workflow("macro_returns")
        wf.macro = nodes.ParallelIdentityMacro(0)
        macro_out = wf.macro.run()
        self.assertTupleEqual(
            (0, 0),
            macro_out,
            msg="Sanity check that we have non-trivial output including two elements",
        )

        g_from_graph = base.get_full_graph_from_wf(wf)
        workflow_graph_out = run.pull_node(g_from_graph, "macro")

        self.assertTupleEqual(
            macro_out,
            workflow_graph_out,
            msg="Cycling the workflow to a graph and pulling should yield the same result as pulling the workflow node",
        )

        g = base.Graph(label="pure_graph")
        g = base.add_node(g, nodes.Identity(label="input_fork", x=0))
        g = base.add_node(g, nodes.Identity(label="p1"))
        g = base.add_node(g, nodes.Identity(label="p2"))
        g = base.add_edge(g, "input_fork", "p1", "x", "x")
        g = base.add_edge(g, "input_fork", "p2", "x", "x")
        g = group.create_group(g, [0, 1, 2], label="group")
        # This construction and value setting needs to be revisited
        # Here the main point is to verify that we can manually create and use groups with multiple outputs
        # https://github.com/JNmpi/pyiron_core/issues/33

        pure_group_out = run.pull_node(base.get_updated_graph(g), "group")
        self.assertTupleEqual(
            macro_out,
            pure_group_out,
            msg="We should be able to obtain multiple outputs in a pure-graph paradigm",
        )

        updated_pure_group_out = run.pull_node(base.get_updated_graph(g), "group")
        self.assertTupleEqual(
            macro_out,
            updated_pure_group_out,
            msg="I don't know, really, the point is just that at the moment updating the graph does no harm",
        )
