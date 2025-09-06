import dataclasses
import os
import time
import unittest
from collections import OrderedDict

import pyiron_core.pyiron_database.instance_database as idb
from pyiron_core.pyiron_workflow.simple_workflow import (
    PORT_LABEL,
    Node,
    Workflow,
    as_function_node,
    extract_dataclass_parameters,
    extract_input_parameters_from_function,
    get_inputs_data,
    make_node_decorator,
    value_to_string,
)

from static.nodes import Identity, IdentityMacro


@as_function_node
def trial_func(a: int, b: int = 1):
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
        node = trial_func()
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

    def test_connections(self):
        wf = Workflow("single_value")
        wf.upstream = Identity(0)
        wf.downstream_by_port = Identity(wf.upstream.outputs.x)
        wf.downstream_by_node = Identity(wf.upstream)

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

    def test_simple_macro(self):
        m = IdentityMacro(x=42)
        out = m.run()
        self.assertTupleEqual(
            (42, 42),
            out,
            msg="the macro should be runnable and should allow channel-based and "
            "node-based (with single-returns) output formats",
        )

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


class TestValueToString(unittest.TestCase):
    def test_int(self):
        self.assertEqual(value_to_string(42), "42")

    def test_float(self):
        self.assertEqual(value_to_string(3.14), "3.14")

    def test_bool(self):
        self.assertEqual(value_to_string(True), "True")
        self.assertEqual(value_to_string(False), "False")

    def test_none(self):
        self.assertEqual(value_to_string(None), "None")

    def test_str(self):
        self.assertEqual(value_to_string("hello"), '"hello"')

    def test_list(self):
        self.assertIsNone(value_to_string([1, 2, 3]))

    def test_custom_object(self):
        class Dummy:
            pass

        self.assertIsNone(value_to_string(Dummy()))


class TestGetInputsData(unittest.TestCase):
    @staticmethod
    def _some_function(v: int, w: None, x: type(None), y: tuple, z):
        whatever = 42
        return whatever

    @dataclasses.dataclass
    class _SomeData:
        w: int = 42
        x: None = None
        y: type(None) = None
        z: tuple = ()

    def test_function_hint_parsing(self):
        fnc_inputs = get_inputs_data(
            self._some_function, extract_input_parameters_from_function
        )
        self.assertListEqual(
            ["int", "None", "None", "NonPrimitive", "NotHinted"],
            fnc_inputs.data["type"],
            msg="Whitelisted hints, non-primitive hints, and no hint at all should all "
            "parse separately and correctly",
        )

    def test_dataclass_hint_parsing(self):
        dc_inputs = get_inputs_data(
            self._SomeData(),
            extract_dataclass_parameters,
        )
        self.assertListEqual(
            ["int", "None", "None", "NonPrimitive"],
            dc_inputs.data["type"],
            msg="Whitelisted hints, non-primitive hints should parse separately and "
            "correctly; dataclasses _can't_ have un-hinted fields, so no worries "
            "there.",
        )


if __name__ == "__main__":
    unittest.main()
