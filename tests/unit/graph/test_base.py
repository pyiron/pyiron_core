import contextlib
import os
import unittest

from pyiron_core.pyiron_workflow.api import serial
from pyiron_core.pyiron_workflow.graph import base, graph_json, group, run
from static.nodes import AddOne


class TestSaveLoad(unittest.TestCase):
    def test_save_load(self):
        g = base.Graph("test")
        g = base.add_node(g, AddOne(label="n1", x=0))
        g = base.add_node(g, AddOne(label="n2"))
        g = base.add_edge(g, "n1", "n2", "y", "x")
        out = run.pull_node(base.get_updated_graph(g), "n2")

        fname = "serial.json"
        graph_json._save_graph(g, fname)
        try:
            g_loaded = graph_json._load_graph(fname)
            out_loaded = run.pull_node(base.get_updated_graph(g_loaded), "n2")
            self.assertEqual(
                out_loaded,
                out,
                msg="The loaded graph should be operational and give the same result.",
            )
        finally:
            with contextlib.suppress(FileNotFoundError):
                os.remove(fname)

    def test_serial_api_misdirection(self):
        """
        Two groups can coexist, and they redirect their "function import path" to the
        serial API even through a save/load cycle.
        """

        g = base.Graph("two_groups")
        g = base.add_node(g, AddOne(label="n1", x=0))
        g = base.add_node(g, AddOne(label="n2a", x=0))
        g = base.add_node(g, AddOne(label="n2b", x=0))
        g = base.add_edge(g, "n2a", "n2b", "y", "x")
        g = group.create_group(g, ["n1"], label="monogroup")
        g = group.create_group(g, ["n2a", "n2b"], label="digroup")

        expected_path = f"{serial.__name__}.{serial.subgraph.__name__}"

        def get_fnc_import(graph: base.Graph, node_name: str) -> str:
            return graph.nodes[node_name].node.function["import_path"]

        self.assertEqual(expected_path, get_fnc_import(g, "monogroup"))
        self.assertEqual(expected_path, get_fnc_import(g, "digroup"))
        mono_out = run.pull_node(base.get_updated_graph(g), "monogroup")
        self.assertEqual(mono_out, 1)
        di_out = run.pull_node(base.get_updated_graph(g), "digroup")
        self.assertEqual(di_out, 2)

        fname = "api_misdirection.json"
        graph_json._save_graph(g, filename=fname)
        try:
            reloaded = graph_json._load_graph(fname)
            self.assertEqual(expected_path, get_fnc_import(reloaded, "monogroup"))
            self.assertEqual(expected_path, get_fnc_import(reloaded, "digroup"))
            self.assertEqual(
                mono_out, run.pull_node(base.get_updated_graph(reloaded), "monogroup")
            )
            self.assertEqual(
                di_out, run.pull_node(base.get_updated_graph(reloaded), "digroup")
            )
        finally:
            with contextlib.suppress(FileNotFoundError):
                os.remove(fname)

    def test_private_expand_node(self):
        g = base.Graph("test")
        g = base.add_node(g, serial.identity(label="n", x=0))
        g = group.create_group(g, ["n"], label="subgraph")

        with self.subTest("Level consistency"):
            g_expanded = base._expand_node(g, "subgraph")
            self.assertEqual(
                g_expanded.nodes["subgraph"].level + 1,
                g_expanded.nodes["subgraph"].graph.nodes["n"].level,
                msg="Regardless of how many times the same node is expanded, the "
                "children of the expanded node should always have a level one deeper "
                "than their parent.",
            )

    def test_collapse_node(self):
        g = base.Graph("to_collapse")
        g = base.add_node(g, serial.identity(label="n1", x=0))
        g = base.add_node(g, serial.identity(label="n2"))
        g = base.add_node(g, serial.identity(label="n3"))
        g = base.add_edge(g, "n1", "n2", "x", "x")
        g = base.add_edge(g, "n2", "n3", "x", "x")
        g = group.create_group(g, ["n1", "n2"], label="group2")
        gc = base.collapse_node(g, "group2")

        for child_label in gc.nodes["group2"].graph.nodes:
            self.assertIn(child_label, g.nodes, msg="Sanity check")
            self.assertNotIn(
                child_label,
                gc.nodes,
                msg="Collapsed nodes should purge collapsed children from the overall "
                "graph",
            )

        for edge in gc.edges:
            self.assertNotIn(
                edge.source,
                gc.nodes["group2"].graph.nodes,
                msg="Purged nodes should not appear in outstanding edges",
            )
            self.assertNotIn(
                edge.target,
                gc.nodes["group2"].graph.nodes,
                msg="Purged nodes should not appear in outstanding edges",
            )

        gu = base.uncollapse_node(gc, "group2")
        self.assertSetEqual(
            set(label for label in g.nodes),
            set(label for label in gu.nodes),
            msg="Uncollapsing node should cause graph to recover original nodes",
        )
        self.assertEqual(
            len(g.nodes),
            len(gu.nodes),
            msg="Uncollapsed graph should have no extra nodes",
        )

        def hashable_edge(edge):
            return (edge.source, edge.target, edge.sourceHandle, edge.targetHandle)

        self.assertSetEqual(
            set(hashable_edge(edge) for edge in g.edges),
            set(hashable_edge(edge) for edge in gu.edges),
            msg="Uncollapsing node should cause graph to recover original edges",
        )
        self.assertEqual(
            len(g.edges),
            len(gu.edges),
            msg="Uncollapsed graph should have no extra edges",
        )

        guu = base.uncollapse_node(gu, "group2")
        self.assertSetEqual(
            set(label for label in g.nodes),
            set(label for label in guu.nodes),
            msg="Repeatedly un-collapsing should have no effect",
        )
        self.assertEqual(
            len(g.nodes),
            len(guu.nodes),
            msg="Repeatedly un-collapsing should have no effect",
        )
        self.assertSetEqual(
            set(hashable_edge(edge) for edge in g.edges),
            set(hashable_edge(edge) for edge in guu.edges),
            msg="Repeatedly un-collapsing should have no effect",
        )
        self.assertEqual(
            len(g.edges),
            len(guu.edges),
            msg="Repeatedly un-collapsing should have no effect",
        )
