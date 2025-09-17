import unittest

from pyiron_core.pyiron_workflow import imports
from pyiron_core.pyiron_workflow.api import serial

from static import nodes

def pseudo_subgraph(): ...
pseudo_subgraph._is_subgraph_code = True

def non_subgraph(): ...


class TestImports(unittest.TestCase):
    def test_subgraph_misdirection(self):
        api_path = f"{serial.__name__}.{serial.subgraph.__name__}"

        self.assertEqual(
            api_path,
            imports.get_import_path_from_type(pseudo_subgraph),
            msg="We expect functions tagged with the `_is_subgraph_code` attribute to "
            "get their reference path redirected to the serial API placeholder.",
        )
        self.assertNotEqual(
            api_path,
            imports.get_import_path_from_type(non_subgraph),
            msg="We expect functions not tagged with the `_is_subgraph_code` attribute "
            "to get their reference path unmodified.",
        )

    def test_get_import_path_from_type(self):
        self.assertEqual(
            f"{nodes.__name__}.{nodes.AddOne.__qualname__}",
            imports.get_import_path_from_type(nodes.AddOne),
            msg="Expect typical path for a typical import",
        )

        self.assertEqual(
            f"{int.__module__}.{int.__name__}",
            imports.get_import_path_from_type(int),
            msg="Expect typical path for built-in types"
        )

        nontrivial_qualname = nodes.SomethingNested.the_nested_thing
        self.assertEqual(
            f"{nontrivial_qualname.__module__}.{nontrivial_qualname.__qualname__}",
            imports.get_import_path_from_type(nontrivial_qualname),
            msg="Expect typical path for objects with non-trivial qualnames",
        )

    def test_get_object_from_path(self):
        with self.subTest("Return None on a missing module"):
            self.assertIsNone(
                imports.get_object_from_path("this_doesnt.exist", log=None),
                msg="If the module is not importable module, we should return None",
            )

        with self.subTest("Load objects that do exist"):
            self.assertIs(
                nodes.AddOne,
                imports.get_object_from_path(
                    f"{nodes.__name__}.{nodes.AddOne.__qualname__}"
                ),
                msg="Should be able to load nodes"
            )

            self.assertIs(
                int,
                imports.get_object_from_path(
                    f"{int.__module__}.{int.__qualname__}"
                ),
                msg="Should be able to load built-ins"
            )

            self.assertIs(
                nodes.SomethingNested.the_nested_thing,
                imports.get_object_from_path(
                    f"{nodes.__name__}."
                    f"{nodes.SomethingNested.the_nested_thing.__qualname__}"
                ),
                msg="Should be able to load objects with non-trivial qualnames"
            )
