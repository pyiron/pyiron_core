from pyiron_core.pyiron_workflow import Node, as_function_node


@as_function_node
def CreateDB(
    user: str = "joerg",
    password: str = "none",
    host: str = "localhost",
    port: int = 5432,
    database: str = "none",
):
    import pyiron_core.pyiron_database.api as pyiron_database

    if database == "none":
        database = user

    connection_str = f"postgresql://{user}:{password}@{host}:{port}/{database}"

    db = pyiron_database.PostgreSQLInstanceDatabase(connection_str)
    db.init()

    return db


@as_function_node
def DeleteDB(db):
    """
    Delete the database and all its contents!
    """

    db.drop()
    return db


@as_function_node
def ShowTable(db):
    import pandas as pd
    from sqlalchemy.orm import sessionmaker

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
    import pandas as pd
    from sqlalchemy.orm import sessionmaker

    import pyiron_core.pyiron_database.api as pyiron_database
    from pyiron_core.pyiron_workflow.api.gui import GuiGraph

    Session = sessionmaker(bind=db.engine)
    session = Session()

    df = pd.read_sql(session.query(db.table).statement, session.bind)

    session.close()

    _, graph = pyiron_database.restore_node_from_database(
        db=db, node_hash=df.hash.iloc[node_id]
    )

    gui_graph = GuiGraph(graph)

    return gui_graph


@as_function_node
def GetHash(node: Node):
    """
    Get the hash of a node
    """
    import pyiron_core.pyiron_database.api as pyiron_database

    print("inputs: ", node.inputs)
    hash = pyiron_database.get_hash(node)
    return hash
