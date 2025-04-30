from __future__ import annotations

import contextlib
import pickle
from pathlib import Path
from types import TracebackType
from typing import Any

from pyiron_database.generic_storage.interface import StorageGroup


class PickleGroup(StorageGroup):
    def __init__(self, data: dict[str, Any]) -> None:
        self.data = data

    def __contains__(self, item: object) -> bool:
        return item in self.data

    def __delitem__(self, key: str) -> None:
        del self.data[key]

    def __getitem__(self, key: str) -> Any:
        return self.data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self.data[key] = value

    def __iter__(self):
        return iter(self.data)

    def __len__(self) -> int:
        return len(self.data)

    def create_group(self, key: str) -> PickleGroup:
        if key in self.data:
            raise KeyError(f"{key} already exists")
        self.data[key] = {}
        return PickleGroup(self.data[key])

    def require_group(self, key: str) -> PickleGroup:
        return PickleGroup(self.data[key])

    def is_group(self, key: str) -> bool:
        return isinstance(self.get(key, None), dict)


class PickleStorage(contextlib.AbstractContextManager[PickleGroup]):
    def __init__(self, filename: str, mode="rb") -> None:
        super().__init__()
        self.filename = Path(filename)
        self.mode = mode
        self.data: dict = {}

    def __enter__(self) -> PickleGroup:
        path = self.filename.parent
        path.mkdir(parents=True, exist_ok=True)
        with open(self.filename, self.mode) as file:
            if file.readable():
                self.data = pickle.load(file)

        return PickleGroup(self.data)

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        with open(self.filename, self.mode) as file:
            if file.writable():
                pickle.dump(self.data, file, pickle.HIGHEST_PROTOCOL)
