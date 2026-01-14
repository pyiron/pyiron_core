# Advanced Concepts & Best‑Practice Rules

This guide summarises the key conventions for writing clean `pyiron_core`
workflows. It builds on the notebook `notebooks/llm_doc.md` and provides
concise, copy‑paste‑ready examples.

---

## 📦 Allowed Imports

Only the modules listed below may be imported inside a workflow.  If you need
additional utilities, generate them **inside a node** (see the Function‑Node Rules).

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from pyiron_workflow import (
    Workflow,
    as_function_node,
    as_out_dataclass_node,
    as_inp_dataclass_node,
)
from pyiron_workflow.data_fields import DataArray, EmptyArrayField
from typing import Optional, Literal, TypeAlias, Protocol
# For atomistic use‑cases:
from ase import Atoms
```

---

## 🧩 Function‑Node Rules

*All arguments must have type annotations.* The engine uses these signatures to
create input ports.

```python
@as_function_node
def ComputeEnergy(rho: np.ndarray) -\u003e float:
    return float(np.sum(rho))
```

When returning a custom dataclass, annotate the return type with the class name
and return `ClassName().dataclass()`.

---

## 📊 Custom Dataclass Nodes

Dataclasses are used to group related data into a single port.

```python
@as_out_dataclass_node
class Mesh2D:
    array: DataArray = EmptyArrayField
    x:     DataArray = EmptyArrayField
    y:     DataArray = EmptyArrayField
```

---

## 📈 Multi‑Output & `.outputs`

Nodes that return multiple values expose them via the `.outputs` attribute with
named labels only.

```python
@as_function_node
def MeshgridNode(x: np.ndarray, y: np.ndarray) -\u003e tuple[np.ndarray, np.ndarray]:
    X, Y = np.meshgrid(x, y, indexing="ij")
    return X, Y
```

Use `wf.node.outputs.X` and `wf.node.outputs.Y`; numeric indexing is banned.

---

## 🖋️ Plotting – MathText‑Safe Labels

Labels must be compatible with Matplotlib's default *mathtext*.

```python
label = r"$\\rho(\\mathrm{r})$"
```

---

## 🎉 Minimal Self‑Contained Example

```python
wf = Workflow("Demo")
wf.grid = CreateGrid(spec=GridSpec().dataclass())
wf.plot = PlotDensity(mesh=wf.grid)
wf.run()
wf.plot.show()
```

---

For the full list of rules see the original notebook and the detailed
documentation in `docs/guide/`.
