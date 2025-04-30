from __future__ import annotations

import abc
from collections.abc import MutableMapping
from typing import Any

from pyiron_database.obj_reconstruction.util import recreate_type


def _save_join(separator, items):
    for it in items:
        if separator in it:
            raise ValueError(f"Separator ({separator}) not allowed in item ({it})")
    return separator.join(items)


class StorageGroup(MutableMapping[str, Any], abc.ABC):
    """A dict like container to store arbitrary stuff."""

    separator: str = "@"
    undefined_version: str = "not_defined"

    @abc.abstractmethod
    def create_group(self, key: str) -> StorageGroup:
        pass

    @abc.abstractmethod
    def require_group(self, key: str) -> StorageGroup:
        pass

    @abc.abstractmethod
    def is_group(self, key: str) -> bool:
        pass

    def _recover_value(self, group: StorageGroup) -> Any:
        type = group.get("_type", "group")
        match type:
            case "group":
                return group
            case "type":
                module, qualname, version = group["_class"].split(self.separator)
                return recreate_type(
                    module,
                    qualname,
                    version,
                )
            case "pickle":
                func = group["func"]
                args = group["args"]
                obj = func(*args)
                state = group["state"]
                if hasattr(obj, "__setstate__"):
                    obj.__setstate__(state)
                else:
                    obj.__dict__.update(**state)
                if group["listitems"] is not None:
                    raise RuntimeError("listitems")
                if group["dictitems"] is not None:
                    raise RuntimeError("dictitems")
                return obj
            case "tuple":
                lst = []
                i = 0
                while f"item_{i}" in group:
                    lst.append(group[f"item_{i}"])
                    i += 1
                return tuple(lst)
            case "global":
                module, qualname, version = group["_class"].split(self.separator)
                return recreate_type(
                    module,
                    qualname,
                    version,
                )
            case "base64":
                from base64 import b64decode

                return b64decode(group["value"].encode("utf8"))
            case _:
                raise TypeError(f"Could not instantiate: {type}")

    def _transform_value(self, key: str, value: Any) -> None:
        def save_reduce(
            group,
            self,
            func,
            args,
            state=None,
            listitems=None,
            dictitems=None,
            state_setter=None,
            *,
            obj=None,
        ):
            group["_type"] = "pickle"
            group["func"] = func
            group["args"] = args
            group["state"] = state
            group["listitems"] = listitems
            group["dictitems"] = dictitems
            group["state_setter"] = state_setter

        if isinstance(value, type) or callable(value):
            module, qualname, version = (
                value.__module__,
                (
                    value.__qualname__
                    if hasattr(value, "__qualname__")
                    else value.__class__.__qualname__
                ),
                self.undefined_version,
            )
            group = self.create_group(key)
            group["_type"] = "type"
            group["_class"] = _save_join(self.separator, [module, qualname, version])
            return

        if isinstance(value, tuple):
            group = self.create_group(key)
            group["_type"] = "tuple"
            for i, v in enumerate(value):
                group[f"item_{i}"] = v
            return

        if isinstance(value, dict):
            group = self.create_group(key)
            for k, v in value.items():
                group[k] = v
            return

        if isinstance(value, bytes):
            from base64 import b64encode

            group = self.create_group(key)
            group["_type"] = "base64"
            group["value"] = b64encode(value).decode("utf8")
            return

        reduce = getattr(value, "__reduce_ex__", None)
        if reduce is None:
            reduce = getattr(value, "__reduce__", None)

        if reduce is not None:
            rv = reduce(4)

            if isinstance(rv, str):
                module, qualname, version = value.__module__, rv, self.undefined_version
                group = self.create_group(key)
                group["_type"] = "global"
                group["_class"] = _save_join(
                    self.separator, [module, qualname, version]
                )
                return

            group = self.create_group(key)
            save_reduce(group, value, *rv)
            return
