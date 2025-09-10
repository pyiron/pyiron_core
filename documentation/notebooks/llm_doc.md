## 📜 Workflow Creation Documentation (Final Version with Tuple-Splitting in Template)

### **🎯 Task**
Create a **Pyiron Workflow** with function nodes that:
1. Generate or transform data.
2. Optionally process data in multiple steps.
3. Plot or output the results.

---

### **📝 Rules for Node Implementation**

#### 1. **Imports**
```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from pyiron_workflow import Workflow, as_function_node
# Optional for atomistic cases
from ase import Atoms
```

---

#### 2. **Function Node Rules**
- Write each computational block as a **stand-alone function**.
- Annotate **all arguments** with types (`int`, `float`, `bool`, `str`, `np.ndarray`, `Atoms`, `Figure`, etc.).
- Add **sensible defaults** for optional parameters.
- Decorate with:
  - `@as_function_node("output_label")` for single output.
  - `@as_function_node(["label1", "label2"])` for multiple outputs.
- **Labels**:
  - Must be variable-like names (only letters, numbers, and `_`).
  - No spaces, dots, hyphens, or special symbols.

---

#### 3. **🚫 No Tuples Rule (Tuple-Splitting)**
LLMs **must NOT** produce tuple-typed arguments like `Tuple[int, int, int]` or `Tuple[bool, bool, bool]`.

Instead:
- **Break them into separate parameters** with primitive types.
- Give each parameter a descriptive suffix.

**Tuple Examples → Split:**

Miller indices:
```python
miller_h: int = 1,
miller_k: int = 0,
miller_l: int = 0
```

Periodicity:
```python
periodic_x: bool = True,
periodic_y: bool = True,
periodic_z: bool = False
```

---

#### 4. **Connecting Nodes**
- Create the workflow:
```python
wf = Workflow("descriptive_name")
```
- Add nodes as attributes:
```python
wf.node_name = NodeFunction(arg=..., arg2=...)
```
  - Only override non-default arguments.
- Connect outputs by passing workflow attributes:
```python
wf.next = NextNode(input_name=wf.previous)
```

---

### **📦 Minimal Template (Example-Based)**

```python
# 1️⃣ Imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from pyiron_workflow import Workflow, as_function_node
from ase import Atoms

# 2️⃣ Node definitions

@as_function_node("x")
def LinspaceNode(start: float = 0.0, stop: float = np.pi, num: int = 100) -> np.ndarray:
    """Generate equally spaced values between start and stop."""
    return np.linspace(start, stop, num)

@as_function_node(["sin_x", "cos_x"])
def TrigNode(x: np.ndarray):
    """Return sine and cosine of x."""
    return np.sin(x), np.cos(x)

@as_function_node("figure")
def PlotNode(
    x: np.ndarray,
    y1: np.ndarray,
    y2: np.ndarray,
    title: str = "Sine and Cosine"
) -> Figure:
    """Plot two curves on the same axes."""
    fig, ax = plt.subplots()
    ax.plot(x, y1, label="sin(x)")
    ax.plot(x, y2, label="cos(x)")
    ax.legend()
    ax.set_title(title)
    return fig

# Example of atomistic function with tuple-splitting
@as_function_node("slab")
def SlabNode(
    bulk: Atoms,
    miller_h: int = 1,
    miller_k: int = 0,
    miller_l: int = 0,
    layers: int = 5,
    vacuum: float = 10.0,
    periodic_x: bool = True,
    periodic_y: bool = True,
    periodic_z: bool = False
) -> Atoms:
    """Generate a slab from a bulk structure."""
    # Implementation of slab creation would go here
    return bulk

# 3️⃣ Workflow assembly
wf = Workflow("my_workflow")
wf.x_vals  = LinspaceNode(stop=2*np.pi)
wf.sinvals, wf.cosvals = TrigNode(x=wf.x_vals)
wf.plotter = PlotNode(x=wf.x_vals, y1=wf.sinvals, y2=wf.cosvals)
```

---

### **🚫 Don’ts**
- No `Tuple[...]` for arguments → always separate.
- No code execution (`wf.run()`, `if __name__ == "__main__"`).
- No invalid label names.

---

### ✅ How this helps:
- The **"no tuples"** rule is taught twice: in writing and in code.
- The template shows both **numeric** and **atomistic** nodes.
- Explicit examples for `periodic_x`, `miller_h` ensure LLM won’t accidentally revert to tuples.