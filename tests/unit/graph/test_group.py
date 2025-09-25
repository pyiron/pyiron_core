import unittest

from pyiron_core.pyiron_workflow.graph import base
from pyiron_core.pyiron_workflow.graph.group import create_group
from static.nodes import AddOne


class TestPortToCode(unittest.TestCase):

    def test_create_group(self):
        g = base.Graph("test")
        g = base.add_node(g, AddOne(label="n1", x=0))
        g = base.add_node(g, AddOne(label="n2"))
        g = base.add_edge(g, "n1", "n2", "y", "x")
        with self.assertRaises(ValueError, msg="No creating empty groups"):
            create_group(g, [])

        with self.assertRaises(TypeError, msg="Not identifier type"):
            create_group(g, [3.14])

        with self.assertRaises(TypeError, msg="One type at a time please"):
            create_group(g, [0, "n1"])

        g_by_id = create_group(g, [0, 1], label="sg")
        g_by_name = create_group(g, ["n1", "n2"], label="sg")
        self.assertEqual(
            g_by_id.nodes["sg"].node.graph,
            g_by_name.nodes["sg"].node.graph,
            msg="Subgroups should be equivalent regardless of whether created by index "
            "or label",
        )
