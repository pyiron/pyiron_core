"""
Wrapper classes for serializations to refer to, so we can move the underlying objects
around inside the source code without breaking the serialized import paths.

We need actual new class definitions, as simply importing them from here still yields
the original paths for the `__module__` of imported objects.
"""

import functools

from pyiron_workflow.graph.base import (
    GraphEdge as _GraphEdge,
    GraphNode as _GraphNode,
    identity as _identity,
)


class GraphEdge(_GraphEdge): ...


class GraphNode(_GraphNode): ...


@functools.wraps(_identity)
def identity(*args, **kwargs):
    return _identity(*args, **kwargs)


identity.__module__ = __name__


def func_dataclass():
    raise NotImplementedError(
        "Prior to leveraging this API, the reference JSON workflows used to test the "
        "project contained references to "
        '`"import_path": "pyiron_workflow.simple_workflow.func_dataclass"`. This '
        "is a locally defined function inside another function definition in that "
        "module, and thus not actually importable. Since it's not actually importable, "
        "the code must not actually ever be importing it and thus this exception "
        "should never be reached."
    )


def subgraph():
    raise NotImplementedError(
        "Prior to leveraging this API, the reference JSON workflows used to test the "
        "project contained references to "
        '`"function": "pyiron_workflow.graph.base.subgraph"`. This is not defined '
        "at the top level of the module, and thus not actually importable. Since it's "
        "not actually importable, the code must not actually ever be importing it and "
        "thus this exception should never be reached."
    )
