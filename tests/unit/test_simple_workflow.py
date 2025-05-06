import unittest
from pyiron_workflow.simple_workflow import (
    Workflow,
    Node,
    make_node_decorator,
    as_function_node,
    as_inp_dataclass_node,
)
from dataclasses import dataclass, field
import numpy as np
from collections import OrderedDict


@as_function_node
def test_func(a: int, b: int = 1):
    result = a + b
    return result


class TestSimpleWorkflow(unittest.TestCase):

    def setUp(self):
        self.workflow = Workflow("TestWorkflow")

    def test_workflow_initialization(self):
        self.assertEqual(self.workflow.label, "TestWorkflow")
        self.assertIsInstance(self.workflow._nodes, OrderedDict)
        self.assertIsInstance(self.workflow._edges, list)

    def test_function_node_creation(self):
        node = test_func()
        inp_labels = node.inputs.data["label"]
        out_labels = node.outputs.data["label"]
        ready = node.inputs.data["ready"]
        self.assertIsInstance(node, Node)
        self.assertEqual(node.n_out_labels, 1)
        self.assertEqual(inp_labels, ["a", "b"])
        self.assertEqual(ready, [False, True])
        self.assertEqual(out_labels, ["result"])
        # self.assertEqual(node._func, test_func)

    def test_make_node_decorator(self):
        def dummy_func():
            pass

        decorator = make_node_decorator(dummy_func, "_postfix", "dummy_node")
        self.assertTrue(callable(decorator))

    # def test_node_with_libpath(self):
    #     @dataclass
    #     class TestData:
    #         data: np.ndarray = field(default_factory=lambda: np.array([]))

    #     def test_func(data: TestData):
    #         return data.data.sum()

    #     node = Node(func=test_func, libpath="test/path")
    #     self.assertEqual(node.libpath, "test/path")


if __name__ == "__main__":
    unittest.main()
