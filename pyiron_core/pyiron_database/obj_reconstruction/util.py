import codecs
import pickle
from typing import Any


def get_type(cls: Any) -> tuple[str, str, str]:
    # print("get_type", cls)
    if hasattr(cls, "_func"):  # pyiron function node
        # print("pyiron function node")
        module = cls._func.__module__
        qualname = cls._func.__qualname__
    else:
        module = cls.__class__.__module__
        qualname = cls.__class__.__qualname__
    from importlib import import_module

    base_module = import_module(module.split(".")[0])
    version = (
        base_module.__version__
        if hasattr(base_module, "__version__")
        else "not_defined"
    )
    return module, qualname, version


def recreate_type(
    module_name: str, qualname: str, version: str, strict_version_check: bool = False
) -> Any:
    from importlib import import_module

    base_module = import_module(module_name.split(".")[0])
    if strict_version_check:
        actual_version = (
            base_module.__version__
            if hasattr(base_module, "__version__")
            else "not_defined"
        )
        if actual_version != version:
            raise ValueError(f"Version mismatch: {version} != {actual_version}")
    module = import_module(module_name)
    recreated_type = getattr(module, qualname)
    return recreated_type


def recreate_obj(
    module: str, qualname: str, version: str, init_args: dict[str, Any]
) -> Any:
    recreated_type = recreate_type(module, qualname, version)
    obj = recreated_type(**init_args)
    return obj


def serialize_obj(obj: Any) -> dict[str, Any]:
    module, qualname, version = get_type(obj)
    if hasattr(obj, "__getstate__") and obj.__getstate__() is not None:
        if isinstance(obj.__getstate__(), dict):
            print("serialization needed")
            return dict(
                __import_path__={
                    "module": module,
                    "qualname": qualname,
                    "version": version,
                },
                __getstate__=obj.__getstate__(),
            )
    print("serialization not needed")
    return obj


def deserialize_obj(serialized_obj: dict[str, Any]) -> Any:
    """
    Deserialize an object from its serialized form.

    Parameters:
    serialized_obj (dict[str, Any]): The serialized representation of the object.

    Returns:
    Any: The deserialized object.
    """
    from dataclasses import dataclass
    from types import FunctionType as function

    if isinstance(serialized_obj, dict):
        # print("deserialization needed", serialized_obj)
        if "__import_path__" not in serialized_obj:
            # print("return original object")
            return serialized_obj
        # print("recreate object", serialized_obj)
        import_path = serialized_obj["__import_path__"]
        module = import_path["module"]
        qualname = import_path["qualname"]
        version = import_path["version"]

        recreated_type = recreate_type(module, qualname, version)
        # check whether this is a as_out_dataclass_node (should be simplified by providing a tag for the class)
        if isinstance(recreated_type, function):
            obj = recreated_type()
            if hasattr(obj, "dataclass"):
                return obj.dataclass(**serialized_obj["__getstate__"])
            print("return recreated node object")
            return recreated_type(**serialized_obj["__getstate__"])

        # print("recreate object", recreated_type)
        new_obj = recreated_type()
        new_obj.__setstate__(serialized_obj["__getstate__"])
        return new_obj

    else:
        # print("return original object")
        return serialized_obj


def pickle_dump(obj):
    return codecs.encode(pickle.dumps(obj), "base64").decode()


def pickle_load(buf):
    return pickle.loads(codecs.decode(buf.encode(), "base64"))
