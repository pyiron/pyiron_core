from __future__ import annotations

import abc
from dataclasses import dataclass
from datetime import datetime


class InstanceDatabase(abc.ABC):
    @dataclass
    class NodeData:
        hash: str
        qualname: str
        module: str
        version: str
        connected_inputs: list[str]
        inputs: dict[str, str]
        outputs: list[str]
        output_path: str | None

        # add metadata for job execution
        master_hash: str | None = None
        start_time: datetime | None = None
        cpu_time: float | None = None   # in seconds
        executor: str | None = None     # server info including number of cores
        user: str | None = None

    @abc.abstractmethod
    def init(self) -> None:
        """Initialize what this database backend needs.

        It is possible to call this function multiple times.
        If called multiple times it will skip the initialization.
        """
        pass

    @abc.abstractmethod
    def drop(self) -> None:
        """Drop anything related to this database backend.

        Can be called mutiple times.
        """
        pass

    @abc.abstractmethod
    def create(self, node: NodeData) -> str:
        """
        Create a new entry in the database.

        This method stores a new node in the database using the provided node data.

        Args:
            node (NodeData): The data of the node to be created in the database.

        Returns:
            str: The hash of the newly created node.
        """
        pass

    @abc.abstractmethod
    def read(self, hash: str) -> NodeData | None:
        """
        Read the node data from the database.

        Args:
            hash (str): The hash of the node to be read.

        Returns:
            NodeData | None: The node data if the node exists, otherwise None.
        """
        pass

    @abc.abstractmethod
    def update(self, hash: str, **kwargs) -> None:
        """
        Update a node in the database.

        This method updates the node with the given hash in the database using the
        provided key-value pairs.
        If a key is already present in the node and its value is different from the
        provided value, the value is updated.

        Args:
            hash (str): The hash of the node to be updated.
            **kwargs: The key-value pairs to update the node with.
        """
        pass

    @abc.abstractmethod
    def delete(self, hash: str) -> None:
        """
        Delete the node with the given hash from the database.

        Args:
            hash (str): The hash of the node to be deleted.
        """
        pass
