from pyiron_workflow import as_function_node
from typing import Optional


@as_function_node
def create_db(db_path: Optional[str] = None):
    from pyiron_nodes.development import hash_based_storage as hs

    db = hs.create_nodes_table(echo=False)
    return db


@as_function_node
def remove_row(db, index: int):
    from pyiron_nodes.development import hash_based_storage as hs

    hs.remove_nodes_from_db(db, indices=[index]);
    db = hs.create_nodes_table(echo=False)
    return db


@as_function_node
def query_db(db, query: str):
    from pyiron_nodes.development import hash_based_storage as hs
    import json

    res = hs.db_query_dict(db, json.loads(query))
    return res


@as_function_node
def list_db(db):
    from pyiron_nodes.development import hash_based_storage as hs

    df = hs.list_table(db)
    return df
