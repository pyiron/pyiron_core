import unittest

import pyiron_workflow as pwf
from pyiron_nodes.controls import loop_until


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
        wf.n = loop_until(recursive_function=wf.body, max_steps=100)
        result = wf.run()
        self.assertEqual(11, result)
