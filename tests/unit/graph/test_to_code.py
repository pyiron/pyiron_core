import unittest

import pyiron_workflow.simple_workflow as swf

from pyiron_workflow.graph.to_code import port_to_code


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
                print((type_, val, dflt, expected), port_to_code(p))
                self.assertEqual(port_to_code(p), expected)

    def test_use_default_flag(self):
        p = self.make_port("port", "int", default=42, value=0)
        self.assertEqual(port_to_code(p, use_default=False), "port: int = 0")
        self.assertEqual(port_to_code(p, use_default=True), "port: int = 42")

    def test_scope_naming(self):
        p = self.make_port("x", "int", default=1, value=2)
        self.assertEqual(port_to_code(p, scope="scope"), "scope__x: int = 2")
        self.assertEqual(
            port_to_code(p, scope="mod", scope_delimiter="_"), "mod_x: int = 2"
        )

    def test_scope_combined_permutations(self):
        p = self.make_port("foo", "bool", default=False, value=True)
        self.assertEqual(
            port_to_code(p, use_default=False, scope=None), "foo: bool = True"
        )
        self.assertEqual(
            port_to_code(p, use_default=True, scope=None), "foo: bool = False"
        )
        self.assertEqual(
            port_to_code(p, use_default=True, scope="bar"), "bar__foo: bool = False"
        )
        self.assertEqual(
            port_to_code(p, use_default=False, scope="ns", scope_delimiter="_"),
            "ns_foo: bool = True",
        )
