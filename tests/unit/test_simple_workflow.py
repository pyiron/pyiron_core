import os
import time
import unittest
from collections import OrderedDict

import pyiron_database.instance_database as idb

from pyiron_workflow.simple_workflow import (
    Workflow,
    Node,
    make_node_decorator,
    as_function_node,
)


@as_function_node
def test_func(a: int, b: int = 1):
    result = a + b
    return result


@as_function_node("t")
def Time(store=False):
    return time.time()


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

    def test_storage(self):
        with self.subTest("Off"):
            n = Time(store=False)
            t1 = n.run()
            t2 = n.run()
            self.assertNotEqual(
                t1, t2, msg="Without storage, we expect independent runs"
            )

        with self.subTest("On"):
            n = Time(store=True)
            try:
                t1 = n.run()
                t_sleep = 0.1
                time.sleep(t_sleep)  # To make _sure_ they would otherwise be different
                t2 = n.run()
                self.assertAlmostEqual(
                    t1,
                    t2,
                    msg="With storage, we expect to reload the old time",
                    delta=t_sleep / 10.0,
                )
            finally:
                storage_location = idb.store_node_outputs(n)
                os.unlink(storage_location)
                os.rmdir(storage_location.split(os.sep)[0])


if __name__ == "__main__":
    unittest.main()
