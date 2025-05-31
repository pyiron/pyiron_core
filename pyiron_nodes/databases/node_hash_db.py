from pyiron_workflow import as_function_node, Node
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


@as_function_node
def GetGraph(db, node_id: int):
    """
    Get the graph of a node with id *node_id from the database.
    """
    import pyiron_database.instance_database as idb
    from pyiron_workflow.api import GuiGraph
    from sqlalchemy.orm import sessionmaker
    import pandas as pd

    Session = sessionmaker(bind=db.engine)
    session = Session()

    df = pd.read_sql(session.query(db.table).statement, session.bind)

    session.close()

    _, graph = idb.restore_node_from_database(db=db, node_hash=df.hash.iloc[node_id])

    gui_graph = GuiGraph(graph)

    return gui_graph


@as_function_node
def GetHash(node: Node):
    """
    Get the hash of a node
    """
    import pyiron_database.instance_database as idb

    print("inputs: ", node.inputs)
    hash = idb.get_hash(node)
    return hash
