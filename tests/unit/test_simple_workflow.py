import unittest
from pyiron_workflow.simple_workflow import (
    Workflow,
    Node,
    make_node_decorator,
    as_function_node,
    as_inp_dataclass_node,
    PORT_LABEL,
)
from dataclasses import dataclass, field
import numpy as np
from collections import OrderedDict


@as_function_node
def test_func(a: int, b: int = 1):
    result = a + b
    return result


@as_function_node
def PassThrough(x):
    return x


class TestSimpleWorkflow(unittest.TestCase):

    def setUp(self):
        self.workflow = Workflow("TestWorkflow")

    def test_workflow_initialization(self):
        self.assertEqual(self.workflow.label, "TestWorkflow")
        self.assertIsInstance(self.workflow._nodes, OrderedDict)
        self.assertIsInstance(self.workflow._edges, list)

    def test_function_node_creation(self):
        node = test_func()
        inp_labels = node.inputs.data['label']
        out_labels = node.outputs.data['label']
        ready = node.inputs.data['ready']
        self.assertIsInstance(node, Node)
        self.assertEqual(node.n_out_labels, 1)
        self.assertEqual(inp_labels, ['a', 'b'])
        self.assertEqual(ready, [False, True])
        self.assertEqual(out_labels, ['result'])
        # self.assertEqual(node._func, test_func)

    def test_make_node_decorator(self):
        def dummy_func():
            pass

        decorator = make_node_decorator(dummy_func, "_postfix", "dummy_node")
        self.assertTrue(callable(decorator))

    def test_connections(self):
        wf = Workflow("single_value")
        wf.upstream = PassThrough(0)
        wf.downstream_by_port = PassThrough(wf.upstream.outputs.x)
        wf.downstream_by_node = PassThrough(wf.upstream)

        con_by_port = wf.downstream_by_port.inputs["x"].connections[0]
        con_by_node = wf.downstream_by_node.inputs["x"].connections[0]

        self.assertEqual(
            wf.upstream,
            con_by_port.owner,
            msg="The connection expects to capture the parent",
        )
        self.assertEqual(
            wf.upstream.outputs.data[PORT_LABEL][0],
            con_by_port.label,
            msg="The connection expects to capture the port label",
        )
        self.assertEqual(
            con_by_port.owner,
            con_by_node.owner,
            msg="Result should not differ using the single-output shortcut",
        )
        self.assertEqual(
            con_by_port.label,
            con_by_node.label,
            msg="Result should not differ using the single-output shortcut",
        )

        self.assertListEqual(
            [],
            wf.upstream.inputs["x"].connections,
            msg="Not-connected ports should have an empty list of connections",
        )

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
