from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from _typeshed import DataclassInstance

from dataclasses import fields

from pyiron_database.generic_storage.interface import StorageGroup


def unwrap_dataclass(storage_group: StorageGroup, dataclass: DataclassInstance) -> None:
    for field in fields(dataclass):
        storage_group[field.name] = getattr(dataclass, field.name)
