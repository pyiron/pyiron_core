import unittest

import pyiron_workflow.simple_workflow as swf
from pyiron_workflow.graph import base
from pyiron_workflow.graph.group import create_group

from static.nodes import AddOne


class TestPortToCode(unittest.TestCase):

    def test_create_group(self):
        g = base.Graph("test")
        g = base.add_node(g, AddOne(label="n1", x=0))
        g = base.add_node(g, AddOne(label="n2"))
        g = base.add_edge(g, "n1", "n2", "y", "x")
        with self.assertRaises(ValueError, msg="No creating empty groups"):
            create_group(g, [])