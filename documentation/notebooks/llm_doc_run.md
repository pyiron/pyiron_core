Create a workflow that has the following nodes: a vector x (using linspace), nodes for trigonometric functions and a plotting node. Use the following guidelines to create the workflow:
Summary for the LLM

When you are asked to create a workflow:

Import Workflow and as_function_node.

Write each computational step as a plain function:

annotate every argument and the return type,
give sensible defaults for optional parameters,
decorate with @as_function_node("out_label") or @as_function_node(["out_label_1", "out_labe_2"], ...) for multiple output labels
names for labels should be instructive and behave like variable names (i.e., no dots, etc.)
Instantiate a Workflow with a descriptive name.

Add nodes by assigning them to attributes of the workflow:

wf.<node_name> = <NodeFunction>(<non_default_kwargs>)
copy
Only include arguments whose values differ from the defaults.

Connect nodes simply by passing other workflow attributes as keyword arguments to the node call. The workflow automatically creates the corresponding edges.

Execute with wf.run() and retrieve results from the returned dictionary.

Minimal Template the LLM can fill in
# 1️⃣ Imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from pyiron_workflow import Workflow, as_function_node

# 2️⃣ Node definitions (example)
@as_function_node("my_node")
def MyNode(arg1: float = 1.0, arg2: int = 10) -> np.ndarray:
    ...

# 3️⃣ Build the workflow
wf = Workflow('my_workflow')
wf.first  = MyNode(arg2=20)               # only non‑default arg2 is set
wf.second = AnotherNode(x=wf.first)       # connect first → second
wf.last   = PlotNode(x=wf.first, y=wf.second, title="Result")

# 4️⃣ Run
out = wf.run()
out['last']   # Figure, array, etc.
copy
Follow this pattern and you will obtain a clean, reproducible, and fully typed workflow that can be exported, visualised, or shared with collaborators.