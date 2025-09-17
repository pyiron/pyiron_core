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
        graph_json._save_graph(g, filename=fname, compact=True)
        try:
            reloaded = graph_json._load_graph(fname, compact=True)
            self.assertEqual(expected_path, get_fnc_import(reloaded, "monogroup"))
            self.assertEqual(expected_path, get_fnc_import(reloaded, "digroup"))
            self.assertEqual(mono_out, run.pull_node(base.get_updated_graph(reloaded), "monogroup"))
            self.assertEqual(di_out, run.pull_node(base.get_updated_graph(reloaded), "digroup"))
        finally:
            with contextlib.suppress(FileNotFoundError):
                os.remove(fname)



