import dataclasses
import os
import time
import unittest
from collections import OrderedDict

from static import nodes, not_nodes

import pyiron_core.pyiron_database.api as pyiron_database
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


@as_function_node
def trial_func(a: int, b: int = 1):
    result = a + b
    return result


@as_function_node("t")
def Time(store=False):
    return time.time()


@as_function_node("result")
def Add(x, y):
    return x + y


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

    def test_make_node_decorator(self):
        def dummy_func():
            pass

        decorator = make_node_decorator(dummy_func, "dummy_node")
        self.assertTrue(callable(decorator))

    def test_connections(self):
        wf = Workflow("single_value")
        wf.upstream = nodes.Identity(0)
        wf.downstream_by_port = nodes.Identity(wf.upstream.outputs.x)
        wf.downstream_by_node = nodes.Identity(wf.upstream)

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
        m = nodes.IdentityMacro(x=42)
        out = m.run()
        self.assertTupleEqual(
            (42, 42),
            out,
            msg="the macro should be runnable and should allow channel-based and "
            "node-based (with single-returns) output formats",
        )

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
                storage_location = pyiron_database.store_node_outputs(n)
                os.unlink(storage_location)

    def test_decorator_name_mangling(self):
        with self.subTest("Node classes should get mangled names"):
            self.assertNotEqual(
                not_nodes.Identity.__name__,
                nodes.Identity.__name__,
                msg="The decorated object should get a different name",
            )
            self.assertTrue(
                nodes.Identity.__name__.startswith(not_nodes.Identity.__name__),
                msg="The node class name should derive from the underlying class name",
            )

        with self.subTest("name/qualname alignment"):
            self.assertEqual(
                nodes.Identity.__name__,
                nodes.Identity.__qualname__,
                msg="For unqualified objects, name and qualname should match",
            )


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


class TestPortDirectionality(unittest.TestCase):
    """Test that port connections are validated to flow output->input."""

    def setUp(self):
        self.wf = Workflow("TestPortDirectionality")
        self.wf.n1 = Add(1, 2)

    def test_input_port_at_initialization_raises(self):
        """Input port cannot be passed as argument during node creation."""

        with self.assertRaises(ValueError) as cm:
            self.wf.n2 = Add(self.wf.n1.inputs.x, 10)

        self.assertIn("Add.x=", str(cm.exception))
        self.assertIn("n1.outputs", str(cm.exception))

    def test_input_port_at_call_time_raises(self):
        """Input port cannot be passed when calling an existing node."""
        self.wf.n2 = Add(2, 10)

        with self.assertRaises(ValueError) as cm:
            self.wf.n2(self.wf.n1.inputs.x, 10)

        self.assertIn("n2.x=", str(cm.exception))
        self.assertIn("n1.outputs", str(cm.exception))

    def test_input_port_at_assignment_raises(self):
        """Input port cannot be assigned to another input port."""
        self.wf.n2 = Add()

        with self.assertRaises(ValueError) as cm:
            self.wf.n2.inputs.x = self.wf.n1.inputs.x

        self.assertIn("n2.x=", str(cm.exception))
        self.assertIn("n1.outputs", str(cm.exception))

    def test_output_to_output_assignment_raises(self):
        """Output port cannot be assigned to another output port."""
        self.wf.n2 = Add(3, 4)

        with self.assertRaises(ValueError) as cm:
            self.wf.n1.outputs.result = self.wf.n2.outputs.result

        self.assertIn("n1.result=", str(cm.exception))
        self.assertIn("n1.inputs", str(cm.exception))

    def test_valid_output_to_input_connection(self):
        """Output port CAN be assigned to input port (valid data flow)."""
        # These should NOT raise
        try:
            self.wf.n2 = Add(self.wf.n1.outputs.result, 10)  # Initialization
            self.wf.n3 = Add()
            self.wf.n3.inputs.x = self.wf.n1.outputs.result  # Assignment
            self.wf.n1.run()
            self.wf.n2.run()
            self.wf.n3(y=self.wf.n2.outputs.result)  # Call passing
            self.assertEqual(
                self.wf.n3.outputs.result.value,
                (1 + 2) + ((1 + 2) + 10),
            )
        except ValueError as e:
            self.fail(f"Valid output->input connection raised ValueError: {e}")

    def test_valid_node_shortcut_connection(self):
        """Single-output nodes can be passed directly (shortcut for .outputs.result)."""

        # This should NOT raise (valid shortcut for single-output nodes)
        try:
            self.wf.n2 = Add(self.wf.n1, 10)
        except ValueError as e:
            self.fail(f"Valid node shortcut raised ValueError: {e}")


if __name__ == "__main__":
    unittest.main()
