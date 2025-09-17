import importlib


def get_import_path_from_type(obj):
    from pyiron_core.pyiron_workflow.api import serial

    module = obj.__module__ if hasattr(obj, "__module__") else obj.__class__.__module__
    name = obj.__qualname__ if hasattr(obj, "__qualname__") else obj.__class__.__qualname__

    if name in dir(serial) and getattr(serial, name).__module__ == serial.__name__:
        return f"{serial.__name__}.{name}"
    elif hasattr(obj, "_is_subgraph_code"):
        return f"{serial.__name__}.subgraph"

    path = f"{module}.{name}"
    if path == "numpy.ndarray":
        path = "numpy.array"
    return path


def get_object_from_path(import_path, log=None):
    # Split the path into module and object part
    module_path, _, name = import_path.rpartition(".")
    # print('module_path: ', module_path)
    # Import the module
    try:
        module = importlib.import_module(module_path)
    except ModuleNotFoundError as e:
        if log is not None:
            log.append_stderr(e)
        return None
    # Get the object
    object_from_path = getattr(module, name)
    return object_from_path
