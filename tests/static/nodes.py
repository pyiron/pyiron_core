import pyiron_core.pyiron_workflow as pwf


@pwf.as_function_node
def Identity(x):
    return x


@pwf.as_macro_node(["by_channel", "by_node"])
def IdentityMacro(x):
    wf = pwf.Workflow("subgraph")
    wf.p1 = Identity(x)
    wf.p2 = Identity(wf.p1)
    return wf.p1.outputs.x, wf.p2


@pwf.as_macro_node(["o1", "o2"])
def ParallelIdentityMacro(x):
    wf = pwf.Workflow("subgraph")
    wf.p1 = Identity(x)
    wf.p2 = Identity(x)
    return wf.p1.outputs.x, wf.p2


@pwf.as_function_node
def AddOne(x):
    y = x + 1
    return y
