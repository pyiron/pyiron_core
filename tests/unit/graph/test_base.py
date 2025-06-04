import contextlib
import os
import unittest

from pyiron_workflow.graph import base

from static.nodes import AddOne


class TestSaveLoad(unittest.TestCase):
    def test_save_load(self):
        g = base.Graph("test")
        g = base.add_node(g, AddOne(label="n1", x=0))
        g = base.add_node(g, AddOne(label="n2"))
        g = base.add_edge(g, "n1", "n2", "y", "x")
        out = base.pull_node(base.get_updated_graph(g), "n2")

        fname = "serial.json"
        base._save_graph(g, fname)
        try:
            g_loaded = base._load_graph(fname)
            out_loaded = base.pull_node(
                base.get_updated_graph(g_loaded), "n2"
            )
            self.assertEqual(
                out_loaded,
                out,
                msg="The loaded graph should be operational and give the same result."
            )
        finally:
            with contextlib.suppress(FileNotFoundError):
                os.remove(fname)
