from pyiron_workflow import as_function_node
from typing import Optional


@as_function_node
def CreateDB(
    user: str = "joerg",
    password: str = "none",
    host: str = "localhost",
    port: int = 5432,
    database: str = "none",
):
    import pyiron_database.instance_database as idb

    if database == "none":
        database = user

    connection_str = f"postgresql://{user}:{password}@{host}:{port}/{database}"

    db = idb.PostgreSQLInstanceDatabase(connection_str)
    db.init()

    return db


@as_function_node
def DeleteDB(db):
    """
    Delete the database and all its contents!
    """

    db.drop()
    return db


# @as_function_node
# def remove_row(db, index: int):
#     from pyiron_nodes.development import hash_based_storage as hs

#     hs.remove_nodes_from_db(db, indices=[index])
#     db = hs.create_nodes_table(echo=False)
#     return db


# @as_function_node
# def query_db(db, query: str):
#     from pyiron_nodes.development import hash_based_storage as hs
#     import json

#     res = hs.db_query_dict(db, json.loads(query))
#     return res


@as_function_node
def ShowTable(db):
    from sqlalchemy.orm import sessionmaker
    import pandas as pd

    Session = sessionmaker(bind=db.engine)
    session = Session()

    df = pd.read_sql(session.query(db.table).statement, session.bind)

    session.close()

    return df
