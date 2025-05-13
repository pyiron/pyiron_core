import pyiron_workflow as pwf


@pwf.as_function_node
def PassThrough(x):
    return x
