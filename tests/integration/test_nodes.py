"""
Tests for specific nodes in `pyiron_nodes`
"""

import unittest

import pyiron_workflow as pwf

import pyiron_nodes as pn
from pyiron_nodes import controls


@pwf.as_function_node
def AddUntilLimit(x, y, limit):
    sum = x + y
    condition = sum >= limit
    out = (sum, condition)
    return out


class TestNodes(unittest.TestCase):
    """
    Operating examples for specific nodes
    """

    def test_loop_until(self):
        # I found the requirement to package body node outputs as a single tuple
        # to be unintuitive, and want a log of the right way to do it.
        # This also serves as useful integration test for handling the passing of
        # node instances directly

        wf = pwf.Workflow("my_recusrion")
        wf.body = AddUntilLimit(1, 2, 10)
        wf.n = controls.loop_until(recursive_function=wf.body, max_steps=100)
        result = wf.run()
        self.assertEqual(11, result)

    def test_iter_node(self):
        wf = pwf.Workflow("iter_wf")
        wf.n = pn.math.Add(1, 2)
        wf.iter = pn.executors.IterNode(wf.n, "y", [1, 2, 3], store=False)
        out = wf.run()
        self.assertListEqual([2, 3, 4], out["result"].tolist())
