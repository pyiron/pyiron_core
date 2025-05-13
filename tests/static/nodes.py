import pyiron_workflow as pwf


@pwf.as_function_node
def PassThrough(x):
    return x


@pwf.as_macro_node(["by_channel", "by_node"])
def PassThroughMacro(x):
    wf = pwf.Workflow("subgraph")
    wf.p1 = PassThrough(x)
    wf.p2 = PassThrough(wf.p1)
    return wf.p1.outputs.x, wf.p2

