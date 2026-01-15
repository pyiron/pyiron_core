## 🎯 Goal  
Create a **Pyiron Workflow** that is **self‑contained**, **executable after copy‑paste**, and **easy to use**.  
The workflow should:

1. **Generate / transform** data.  
2. **Process** the data in one or more steps.  
3. **Plot / output** the results.  

All data should travel through **typed dataclass containers** (input‑ and output‑dataclasses) to keep the graph tidy and to minimise the number of individual ports.  

---  

## 📦 Allowed Imports  

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

*Only the modules listed above may be imported.*  
If you need additional utilities, generate them **inside a node** (see Rule 5).  

---  

## 🧩 1️⃣ Function‑Node Rules  

### 1.1 All arguments must have **type annotations**  

| Allowed types | Description |
|---------------|-------------|
| `int`, `float`, `bool`, `str` | Primitive literals. |
| `np.ndarray` | Numpy arrays. |
| `Atoms` | ASE atomic structures. |
| `Figure` | Matplotlib figure objects. |
| **Custom dataclasses** (see § 2) | Use the **decorated class name** (`Mesh2D`, `GridSpec`, …) as the type hint. **Never** place a call (`()`) in the annotation. |
| `Optional[Literal["opt1","opt2",…]]` | Finite‑choice inputs, possibly optional. |
| `Literal["val1","val2",…]` | Mandatory finite‑choice inputs. |

> **Why the restriction?**  
> The workflow engine inspects signatures to generate node‑output containers and to validate connections. Calls such as `Mesh2D().dataclass()` are *instances*, not *types*, and therefore illegal in a type hint.

### 1.2 Return‑type annotation must **match the actual returned object**  

* If a node returns a **custom dataclass**, annotate the return type with the **class name** (`-> Mesh2D`).  
* If a node returns a primitive or `np.ndarray`, annotate accordingly (`-> np.ndarray`).  

### 1.3 Expose **all useful parameters** of the underlying function/code  

* Every parameter that a user might want to control should appear as a **node argument**.  
* For parameters that have a **finite set of valid values**, use `Optional[Literal[...]]` (or `Literal[...]` if mandatory).  
* Example (plotting colormap):  

  ```python
  CmapChoice = Optional[Literal["viridis", "plasma", "inferno", "magma", "cividis"]]
  def PlotDensity(rho: Mesh2D, cmap: CmapChoice = "viridis") -> Figure: ...
  ```

### 1.4 Physical constants must be **defined inside the node** that uses them.  

* Example (Boltzmann constant in eV/K):  

  ```python
  def ComputeDensity(...):
      kB_eV_per_K = 8.617333262e-5   # defined inside the node
      ...
  ```

### 1.5 Node granularity – **avoid over‑splitting** related logic.  

A node should perform a **coherent computational step** (e.g., “create potential”, “compute density”, “plot result”).  

### 1.6 **No external file assumptions**  

* Do **not** read/write files unless the user explicitly requests a file‑based node.  

### 1.7 **No numeric indexing** of multi‑output nodes (see § 3).  

---  

## 📊 2️⃣ Custom Dataclass Nodes (Input & Output)  

### 2.1 Definition  

```python
@as_out_dataclass_node
class Mesh2D:
    array: DataArray = EmptyArrayField   # 2‑D data array
    x:     DataArray = EmptyArrayField   # 1‑D x‑coordinates
    y:     DataArray = EmptyArrayField   # 1‑D y‑coordinates
```

```python
@as_inp_dataclass_node
class GridSpec:
    x_min: float = 0.0
    x_max: float = 2 * np.pi
    y_min: float = 0.0
    y_max: float = 2 * np.pi
    nx:    int   = 100
    ny:    int   = 100
```

*The decorator does **not** turn the class into a normal Python class; it creates a **factory function**.*  

### 2.2 How to use them in type hints  

| Situation | Correct syntax | Reason |
|-----------|----------------|--------|
| Argument annotation | `def foo(data: Mesh2D):` | `Mesh2D` (the factory) is the **type descriptor** recognized by the engine. |
| Return annotation | `def foo(...) -> Mesh2D:` | Same – the node must return `Mesh2D().dataclass()`. |
| Creating the concrete object inside a node | `obj = Mesh2D().dataclass()` | This call produces the actual container with mutable fields. |
| Passing the object to another node | `next_node(V_ext=wf.V_ext, ...)` | `wf.V_ext` already stores the concrete instance; the engine forwards it. |

### 2.3 Why use **input‑** and **output‑dataclasses**?  

* **Cleaner graphs** – a single port carries a whole structured object instead of many scalar ports.  
* **Reduced wiring effort** – only one connection per dataclass is needed.  
* **Self‑documenting** – the fields of the dataclass describe the data that flows through the workflow.  

### 2.4 Optional: Making static type‑checkers happy  

```python
# Type alias (helps mypy, pyright, etc.)
Mesh2DType: TypeAlias = Mesh2D
GridSpecType: TypeAlias = GridSpec

# Or a Protocol if you only need the fields
class Mesh2DProtocol(Protocol):
    array: np.ndarray
    x: np.ndarray
    y: np.ndarray
```

These are **optional** and have no effect at runtime.  

### 2.5 Allowed imports for custom containers  

`as_out_dataclass_node` and `as_inp_dataclass_node` are **explicitly permitted** because they enable structured data flow.  

---  

## 📈 3️⃣ Multi‑Output & `.outputs` Rules (Hard‑Ban)  

If a node returns **multiple** outputs, they must be accessed **only** via the `.outputs` attribute with **named labels**.

```python
@as_function_node
def MeshgridNode(x, y):
    X, Y = np.meshgrid(x, y, indexing="ij")
    return X, Y          # labels are "X" and "Y"
```

**Correct usage**

```python
wf.grid = MeshgridNode(x=xs, y=ys)
wf.plot = PlotDensity(
    X=wf.grid.outputs.X,
    Y=wf.grid.outputs.Y,
    rho=wf.rho,
)
```

**Forbidden**

* `wf.X = wf.grid.outputs.X` (direct assignment)  
* `wf.plot = PlotDensity(X=wf.grid.outputs[0], Y=wf.grid.outputs[1])` (numeric indexing)  

If you need a value repeatedly, create a **dedicated node** that returns that single output.  

---  

## 🖋️ 4️⃣ Plotting Rule – MathText‑Safe Labels  

* Use only **matplotlib‑compatible mathtext** unless the user explicitly requests full LaTeX (`mpl.rcParams['text.usetex'] = True`).  

Supported example  

```python
label = r"$\rho(\mathrm{r})$"          # ✅
title = r"$E_{\mathrm{kin}}$"          # ✅
```

Unsupported in default mathtext  

```python
r"$\rho(\mathbf{r})$"   # ❌ (fails without usetex)
```

---  

## 🛠️ 5️⃣ Workflow Construction Rules  

1. **All computation must happen inside function nodes** – never inline in the workflow assembly.  
2. **Inputs at assembly time** may only be **basic literals** (`int`, `float`, `str`, `bool`, `Optional[...]`, `Literal[...]`) **or** outputs of other nodes.  
3. **Never assign `None` or placeholder arrays** to workflow attributes.  
4. **No external file reads/writes** unless a node explicitly does so (user‑requested).  

---  

## 📚 6️⃣ Minimal Compliant Example (Shows Input‑/Output‑Dataclasses & `Optional[Literal]`)  

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from typing import Optional, Literal

from pyiron_workflow import (
    Workflow,
    as_function_node,
    as_out_dataclass_node,
    as_inp_dataclass_node,
)
from pyiron_workflow.data_fields import DataArray, EmptyArrayField

# ----------------------------------------------------------------------
# 1️⃣  Output dataclass – travels through the workflow
# ----------------------------------------------------------------------
@as_out_dataclass_node
class Mesh2D:
    """Container for a 2‑D field and its grid vectors."""
    array: DataArray = EmptyArrayField
    x:     DataArray = EmptyArrayField
    y:     DataArray = EmptyArrayField

# ----------------------------------------------------------------------
# 2️⃣  Input dataclass – gathers grid parameters from the user
# ----------------------------------------------------------------------
@as_inp_dataclass_node
class GridSpec:
    """Rectangular grid specification."""
    x_min: float = 0.0
    x_max: float = 2 * np.pi
    y_min: float = 0.0
    y_max: float = 2 * np.pi
    nx:    int   = 100
    ny:    int   = 100

# ----------------------------------------------------------------------
# 3️⃣  Nodes
# ----------------------------------------------------------------------
@as_function_node
def CreateExamplePotential(grid: GridSpec, amplitude: float = 1.0) -> Mesh2D:
    """Generate a sinusoidal external potential on the supplied grid."""
    x = np.linspace(grid.x_min, grid.x_max, grid.nx)
    y = np.linspace(grid.y_min, grid.y_max, grid.ny)
    X, Y = np.meshgrid(x, y, indexing="ij")

    pot = Mesh2D().dataclass()
    pot.array = amplitude * np.sin(X) * np.cos(Y)
    pot.x = X
    pot.y = Y
    return pot

@as_function_node("density")
def ComputeDensity(V_ext: Mesh2D, T: float = 300.0, rho_bulk: float = 0.01) -> Mesh2D:
    """Boltzmann‑weighted density from an external potential."""
    kB_eV_per_K = 8.617333262e-5
    beta = 1.0 / (kB_eV_per_K * T)

    rho = Mesh2D().dataclass()
    rho.array = rho_bulk * np.exp(-beta * V_ext.array)
    rho.x = V_ext.x
    rho.y = V_ext.y
    return rho

# ----------------------------------------------------------------------
# 4️⃣  Plotting node – expose selectable colormap and shading
# ----------------------------------------------------------------------
CmapChoice = Optional[Literal["viridis", "plasma", "inferno", "magma", "cividis"]]

@as_function_node
def PlotDensity(
    rho: Mesh2D,
    cmap: CmapChoice = "viridis",
    shading: Literal["auto", "nearest"] = "auto",
) -> Figure:
    """Plot the 2‑D density field."""
    fig, ax = plt.subplots()
    pcm = ax.pcolormesh(
        rho.x,
        rho.y,
        rho.array,
        shading=shading,
        cmap=cmap,
    )
    fig.colorbar(pcm, ax=ax, label=r"$\rho(\mathrm{r})$")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Density")
    return fig

# ----------------------------------------------------------------------
# 5️⃣  Assemble the workflow – only literals + node outputs
# ----------------------------------------------------------------------
wf = Workflow("classical_dft_2d_density")

# 5.1  Provide grid definition (input‑dataclass node)
wf.grid = GridSpec(
    x_min=0.0,
    x_max=2 * np.pi,
    y_min=0.0,
    y_max=2 * np.pi,
    nx=200,
    ny=200,
)

# 5.2  Create external potential using the grid object
wf.V_ext = CreateExamplePotential(grid=wf.grid, amplitude=0.5)

# 5.3  Compute Boltzmann density
wf.rho = ComputeDensity(V_ext=wf.V_ext, T=300.0, rho_bulk=0.01)

# 5.4  Plot – user can change colormap or shading here
wf.fig = PlotDensity(
    rho=wf.rho,
    cmap="plasma",          # any of the allowed literals, or omit for default
    shading="auto",
)

# ----------------------------------------------------------------------
# 6️⃣  (Optional) Display the figure in a Jupyter notebook
# ----------------------------------------------------------------------
# from IPython.display import display
# display(wf.fig)
```

---  

## 📌 Common Pitfalls (Recap)  

| Issue | ❌ Bad | ✅ Good |
|-------|-------|--------|
| Using a call in a type hint | `def foo(p: Mesh2D().dataclass())` | `def foo(p: Mesh2D)` |
| Returning wrong type | `def foo(...) -> np.ndarray` but returns `Mesh2D` | Align annotation with return (`-> Mesh2D`) |
| Numeric indexing of multi‑output | `wf.node.outputs[0]` | `wf.node.outputs.X` |
| Unsupported mathtext | `label=r"$\rho(\mathbf{r})$"` | `label=r"$\rho(\mathrm{r})$"` |
| External file read in workflow assembly | `wf.V = np.loadtxt("file.dat")` | `wf.V = LoadPotential(filename="file.dat")` (only if user asks) |
| Not exposing useful parameters | Fixed colormap inside `PlotDensity` | `cmap: Optional[Literal[...]] = "viridis"` |

