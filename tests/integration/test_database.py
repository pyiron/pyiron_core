"""
Tests for tools in the local version of `pyiron_database`
"""

import unittest

from static import nodes

import pyiron_core.pyiron_workflow as pwf
from pyiron_core import pyiron_database


class TestDatabase(unittest.TestCase):

    def test_node_connections(self):
        wf_port = pwf.Workflow("hash_wf_steps")
        wf_port.n1 = nodes.Identity(0)
        wf_port.n2 = nodes.Identity(wf_port.n1.outputs.x)
        wf_port.run()

        wf_node = pwf.Workflow("hash_wf_steps")
        wf_node.n1 = nodes.Identity(0)
        wf_node.n2 = nodes.Identity(wf_node.n1)
        wf_node.run()

        self.assertEqual(
            pyiron_database.instance_database.node.node_inputs_to_jsongroup(
                wf_port.n2
            ).data["x"],
            pyiron_database.instance_database.node.node_inputs_to_jsongroup(
                wf_node.n2
            ).data["x"],
            msg="The hash should be accessible and the same regardless of whether we "
            "exploit the connect-directly-to-(single-output)-node shortcut",
        )
