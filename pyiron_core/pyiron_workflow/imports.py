import importlib
import types
import typing


def _get_locally_defined_objects(module: types.ModuleType) -> list[object]:
    """
    Get everything defined locally in a module, avoiding things that are accessible
    there but defined elsewhere.
    """
    return [
        name
        for name in dir(module)
        if getattr(getattr(module, name), "__module__", None) == module.__name__
    ]


def get_import_path_from_type(obj):
    from pyiron_core.pyiron_workflow.api import serial

    module = obj.__module__ if hasattr(obj, "__module__") else obj.__class__.__module__
    name = (
        obj.__qualname__ if hasattr(obj, "__qualname__") else obj.__class__.__qualname__
    )

    if name in _get_locally_defined_objects(serial):
        return f"{serial.__name__}.{name}"
    elif hasattr(obj, "_is_subgraph_code"):
        return f"{serial.__name__}.subgraph"

    path = f"{module}.{name}"
    if path == "numpy.ndarray":
        path = "numpy.array"
    return path


def get_object_from_path(import_path, log=None):
    try:
        return _bagofholding_import_from_string(import_path)
    except ModuleNotFoundError as e:
        if log is not None:
            log.append_stderr(e)
        return None


def _bagofholding_import_from_string(library_path: str) -> typing.Any:
    # https://github.com/pyiron/bagofholding/blob/071aa2ebece51de0342f5ebd8d966f2bdc590527/bagofholding/retrieve.py#L13-L22
    split_path = library_path.split(".", 1)
    if len(split_path) == 1:
        module_name, path = split_path[0], ""
    else:
        module_name, path = split_path
    obj = importlib.import_module(module_name)
    for k in path.split("."):
        obj = getattr(obj, k)
    return obj
