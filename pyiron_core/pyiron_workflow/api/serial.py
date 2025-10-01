"""
Wrapper classes for serializations to refer to, so we can move the underlying objects
around inside the source code without breaking the serialized import paths.

We need actual new class definitions, as simply importing them from here still yields
the original paths for the `__module__` of imported objects.
"""

import functools

from pyiron_core.pyiron_workflow.graph.base import (
    GraphEdge as _GraphEdge,
)
from pyiron_core.pyiron_workflow.graph.base import (
    GraphNode as _GraphNode,
)
from pyiron_core.pyiron_workflow.simple_workflow import identity as _identity


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
        '`"import_path": "pyiron_core.pyiron_workflow.simple_workflow.func_dataclass"`. This '
        "is a locally defined function inside another function definition in that "
        "module, and thus not actually importable. Since it's not actually importable, "
        "the code must not actually ever be importing it and thus this exception "
        "should never be reached."
    )


def subgraph():
    raise NotImplementedError(
        "Prior to leveraging this API, the reference JSON workflows used to test the "
        "project contained references to "
        '`"function": "pyiron_core.pyiron_workflow.graph.base.subgraph"`, or whatever '
        "the subgraph label was whenever the node being serialzed was a created group. "
        "These are dynamically defined functions and not actually importable. Now, we "
        "replace those with references to this static object. It never actually gets "
        "imported, and exists merely to provide clarity for what the serialization "
        "means when it puts a function path for a subgraph node."
    )
