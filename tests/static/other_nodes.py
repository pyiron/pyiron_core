import pyiron_core.pyiron_workflow as pwf


@pwf.as_function_node
def Identity(x):
    return x
