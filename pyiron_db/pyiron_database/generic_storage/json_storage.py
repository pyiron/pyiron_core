from __future__ import annotations

import contextlib
import json
from pathlib import Path
from types import NoneType, TracebackType
from typing import Any

from pyiron_database.generic_storage.interface import StorageGroup


class JSONGroup(StorageGroup):
    def __init__(self, data: dict[str, Any]) -> None:
        self.data = data

    def __contains__(self, item: object) -> bool:
        return item in self.data

    def __delitem__(self, key: str) -> None:
        del self.data[key]

    def __getitem__(self, key: str) -> Any:
        if self.is_group(key):
            group = JSONGroup(self.data[key])
            type = group.get("_type", "group")
            match type:
                case "group":
                    return group
                case _:
                    return self._recover_value(group)

        return self.data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        if isinstance(value, int | float | str | list | NoneType):
            self.data[key] = value
            return

        self._transform_value(key, value)

        if key not in self:
            self.data[key] = value

    def __iter__(self):
        return iter(self.data)

    def __len__(self) -> int:
        return len(self.data)

    def create_group(self, key: str) -> JSONGroup:
        if key in self.data:
            raise KeyError(f"{key} already exists")
        self.data[key] = {}
        return JSONGroup(self.data[key])

    def require_group(self, key: str) -> JSONGroup:
        return JSONGroup(self.data[key])

    def is_group(self, key: str) -> bool:
        return isinstance(self.data.get(key, None), dict)


class JSONStorage(contextlib.AbstractContextManager[JSONGroup]):
    def __init__(self, filename: str, mode: str = "r") -> None:
        super().__init__()
        self.filename = Path(filename)
        self.mode = mode
        self.data: dict[str, Any] = {}

    def __enter__(self) -> JSONGroup:
        path = self.filename.parent
        path.mkdir(parents=True, exist_ok=True)
        with open(self.filename, self.mode) as file:
            if file.readable():
                self.data = json.loads(file.read())

        return JSONGroup(self.data)

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        with open(self.filename, self.mode) as file:
            if file.writable():
                file.write(json.dumps(self.data))
