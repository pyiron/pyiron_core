import unittest

import pyiron_core.pyiron_workflow.simple_workflow as swf
from pyiron_core.pyiron_workflow.graph import base


@swf.as_function_node
def NonPrimitiveHint(x: tuple):
    y = x[0] if len(x) > 0 else None
    return y


class TestPortToCode(unittest.TestCase):
    @staticmethod
    def make_port(label, type_, default=swf.NotData, value=swf.NotData):
        port_data = {
            swf.PORT_LABEL: [label],
            swf.PORT_TYPE: [type_],
            swf.PORT_DEFAULT: [default],
            swf.PORT_VALUE: [value],
        }
        return swf.Port(port_data, label)

    def test_all_valid_types(self):
        cases = [
            # type, value, default, expectation
            ("int", 42, 0, "port: int = 42"),
            ("float", 3.14, 2.71, "port: float = 3.14"),
            ("bool", True, False, "port: bool = True"),
            ("None", None, None, "port: None = None"),
            ("str", "x", "y", 'port: str = "x"'),
            ("NotHinted", "x", "y", 'port="x"'),
            ("NonPrimitive", {}, {}, "port"),
            ("Node", object(), object(), "port: Node"),
        ]
        for type_, val, dflt, expected in cases:
            with self.subTest(type=type_):
                p = self.make_port("port", type_, default=dflt, value=val)
                print((type_, val, dflt, expected), base.port_to_code(p))
                self.assertEqual(base.port_to_code(p), expected)

    def test_use_default_flag(self):
        p = self.make_port("port", "int", default=42, value=0)
        self.assertEqual(base.port_to_code(p, use_default=False), "port: int = 0")
        self.assertEqual(base.port_to_code(p, use_default=True), "port: int = 42")

    def test_scope_naming(self):
        p = self.make_port("x", "int", default=1, value=2)
        self.assertEqual(base.port_to_code(p, scope="scope"), "scope__x: int = 2")

    def test_scope_combined_permutations(self):
        p = self.make_port("foo", "bool", default=False, value=True)
        self.assertEqual(
            base.port_to_code(p, use_default=False, scope=None), "foo: bool = True"
        )
        self.assertEqual(
            base.port_to_code(p, use_default=True, scope=None), "foo: bool = False"
        )
        self.assertEqual(
            base.port_to_code(p, use_default=True, scope="bar"), "bar__foo: bool = False"
        )


class TestGetCodeFromGraph(unittest.TestCase):
    def test_nonprimitive_hints(self):
        node_label = "nph"
        g = base.Graph(label="my_graph")
        g = base.add_node(g, NonPrimitiveHint(label=node_label))
        code_string = base.get_code_from_graph(g, scope_inputs=False)
        self.assertEqual(
            "NonPrimitive",
            base.get_node_input_port(g.nodes[node_label], "x").type,
            msg="Sanity check that the port holds `NonPrimitive` type",
        )
        print(code_string)
        self.assertTrue(
            code_string.startswith("\ndef my_graph(x):"),
            msg="Non-primitive types should be purged from code representations -- we "
            "can't hint 'NonPrimitive'",
            # Update test if complex types are handled more robustly
            # https://github.com/JNmpi/pyiron_core/issues/78
        )
