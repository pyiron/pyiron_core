from __future__ import annotations

try:
    from sqlalchemy import Column, MetaData, String, DateTime, Float, Table, create_engine
    from sqlalchemy.dialects.postgresql import JSONB, insert

    FAILED_IMPORT = None
except ImportError as err:
    FAILED_IMPORT = err.name


from .InstanceDatabase import InstanceDatabase


class PostgreSQLInstanceDatabase(InstanceDatabase):
    def __init__(self, connection_string: str, echo: bool = False) -> None:
        if FAILED_IMPORT is not None:
            raise ImportError(
                f"{type(self)} requires '{FAILED_IMPORT}' to be installed."
            )

        self.metadata = MetaData()
        self.table = Table(
            "nodes",
            self.metadata,
            Column("hash", String, primary_key=True),
            Column("qualname", String, nullable=True),
            Column("module", String, nullable=True),
            Column("version", String, nullable=True),
            Column("connected_inputs", JSONB, nullable=True),
            Column("inputs", JSONB, nullable=True),
            Column("outputs", JSONB, nullable=True),
            Column("output_path", String, nullable=True),
            # add metadata for job execution
            Column("master_hash", String, nullable=True),
            Column("start_time", DateTime, nullable=True),
            Column("cpu_time", Float, nullable=True),  # in seconds
            Column("executor", String, nullable=True),  # server info including number of cores
            Column("user", String, nullable=True),         
        )

        self.engine = create_engine(connection_string, echo=echo)

    def init(self) -> None:
        self.metadata.create_all(self.engine)

    def drop(self) -> None:
        self.metadata.drop_all(self.engine)

    def create(
        self,
        node: InstanceDatabase.NodeData,
    ) -> str:
        with self.engine.connect() as connection:
            stmt = (
                insert(self.table)
                .values(
                    hash=node.hash,
                    qualname=node.qualname,
                    module=node.module,
                    version=node.version,
                    connected_inputs=node.connected_inputs,
                    inputs=node.inputs,
                    outputs=node.outputs,
                    output_path=node.output_path,
                    master_hash=node.master_hash,
                    start_time=node.start_time,
                    cpu_time=node.cpu_time,
                    executor=node.executor,
                    user=node.user,
                )
                .on_conflict_do_nothing()
            )
            result = connection.execute(stmt)
            connection.commit()
            return result.inserted_primary_key[0]

    def read(self, hash: str) -> InstanceDatabase.NodeData | None:
        with self.engine.connect() as connection:
            stmt = self.table.select().where(self.table.c.hash == hash)
            result = connection.execute(stmt).first()
            return None if result is None else self.NodeData(**result._mapping)

    def update(self, hash: str, **kwargs) -> None:
        with self.engine.connect() as connection:
            stmt = self.table.update().where(self.table.c.hash == hash).values(**kwargs)
            connection.execute(stmt)

    def delete(self, hash: str) -> None:
        with self.engine.connect() as connection:
            stmt = self.table.delete().where(self.table.c.hash == hash)
            connection.execute(stmt)
