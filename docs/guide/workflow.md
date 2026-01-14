# Workflow

A Pyiron Workflow represents a complete scientific pipeline composed of interconnected computational steps. While nodes define individual operations, the workflow orchestrates how these operations connect and execute in sequence to transform inputs into meaningful scientific results. This section explains how to assemble nodes into robust, reproducible workflows that capture your scientific process with precision.

## The Conceptual Foundation

Workflows in Pyiron embody the principle of **executable scientific methodology** - transforming research procedures into structured, verifiable computational processes. At their core, workflows address critical challenges in computational science:

1. **Process Documentation**: Scientific workflows should serve as executable documentation of research methods, making procedures transparent and shareable.

2. **Dependency Management**: Complex analyses involve numerous interdependent steps; workflows explicitly map these dependencies to ensure correct execution order.

3. **Parameter Propagation**: Scientific studies often require systematic parameter variations; workflows provide structured mechanisms for managing these variations.

4. **Result Traceability**: Understanding how final results derive from initial inputs is essential for validation; workflows maintain this lineage throughout execution.

## Workflow Construction Principles

A Pyiron Workflow is created by connecting nodes according to specific principles that ensure scientific rigor and computational reliability.

### Basic Workflow Structure

First, let's define the necessary nodes:

```python
import numpy as np
from matplotlib.figure import Figure
from typing import Literal

from pyiron_core.pyiron_workflow import (
    as_function_node,
    as_out_dataclass_node,
    as_inp_dataclass_node,
)
from pyiron_core.pyiron_workflow.data_fields import DataArray, EmptyArrayField

# Define dataclass nodes
@as_inp_dataclass_node
class GridParams:
    """Parameters for creating a computational grid"""
    x_min: float = 0.0
    x_max: float = 10.0
    y_min: float = 0.0
    y_max: float = 10.0
    nx: int = 100
    ny: int = 100

@as_out_dataclass_node
class Grid2D:
    """Holds a 2D grid structure"""
    x: DataArray = EmptyArrayField
    y: DataArray = EmptyArrayField
    X: DataArray = EmptyArrayField
    Y: DataArray = EmptyArrayField

@as_out_dataclass_node
class DensityResults:
    """Holds density calculation results"""
    density: DataArray = EmptyArrayField
    grid: Grid2D = EmptyArrayField

# Define function nodes
@as_function_node
def CreateGrid(params: GridParams) -> Grid2D:
    """Creates a 2D grid based on parameters"""
    x = np.linspace(params.x_min, params.x_max, params.nx)
    y = np.linspace(params.y_min, params.y_max, params.ny)
    X, Y = np.meshgrid(x, y, indexing="ij")
    
    grid = Grid2D().dataclass()
    grid.x = x
    grid.y = y
    grid.X = X
    grid.Y = Y
    return grid  # Output port will be named "grid"

@as_function_node
def ComputeDensity(grid: Grid2D, amplitude: float = 1.0) -> DensityResults:
    """Computes a sample density field on the grid"""
    density = amplitude * np.sin(grid.X) * np.cos(grid.Y)
    
    results = DensityResults().dataclass()
    results.density = density
    results.grid = grid
    return results  # Output port will be named "results"

@as_function_node
def PlotResults(results: DensityResults, cmap: Literal["viridis", "plasma", "coolwarm"] = "viridis") -> Figure:
    """Creates a visualization of the density results"""
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots()
    pcm = ax.pcolormesh(
        results.grid.X, 
        results.grid.Y, 
        results.density, 
        cmap=cmap
    )
    fig.colorbar(pcm, ax=ax, label="Density")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Density Field")
    return fig  # Output port will be named "fig"
```

Now we can assemble the workflow:

```python
from pyiron_core.pyiron_workflow import Workflow

# 1. Create a workflow instance with a descriptive name
wf = Workflow("density_analysis_workflow")

# 2. Add nodes to the workflow with meaningful attribute names
wf.grid_params = GridParams(
    x_min=0.0,
    x_max=2 * np.pi,
    nx=200  # Only specify non-default parameters
)
wf.grid = CreateGrid(params=wf.grid_params)
wf.density = ComputeDensity(grid=wf.grid, amplitude=np.sqrt(2.0))
wf.figure = PlotResults(results=wf.density, cmap="plasma")
```

### Understanding Workflow Syntax

In Pyiron Workflow, the syntax has specific meaning that's crucial to understand:

```python
wf.grid_1 = CreateGrid(params=wf.grid_params)
wf.grid_2 = CreateGrid(params=wf.grid_params)
```

- **Left side (`wf.grid_1`)**: This assigns a **label** (`grid_1`) to the node instance within the workflow. This label becomes how you reference this specific node instance later in your workflow.

- **Right side (`CreateGrid(params=wf.grid_params)`)**: This creates a new instance of the `CreateGrid` node type, with its `params` input connected to the output of the `grid_params` node.

When you create multiple instances of the same node type (like `grid_1` and `grid_2` above), you're creating **distinct computational steps** that can be executed independently, even if they use the same input parameters.

#### Output Port Naming Rules

**Critical Insight**: Output port names are determined by the **variable name used in the return statement**, not by the function name or return type.

```python
@as_function_node
def CreateGrid(params: GridParams) -> Grid2D:
    # ... computation ...
    grid = Grid2D().dataclass()
    # ... populate grid ...
    return grid  # Output port will be named "grid"
```

In this example, the output port is named `grid` because that's the variable name in the return statement.

Similarly:

```python
@as_function_node
def ComputeDensity(grid: Grid2D, amplitude: float = 1.0) -> DensityResults:
    # ... computation ...
    results = DensityResults().dataclass()
    # ... populate results ...
    return results  # Output port will be named "results"
```

Here, the output port is named `results`.

#### Single-Output vs. Multi-Output Node Connections

**For single-output nodes** (most common case):
```python
wf.grid = CreateGrid(params=wf.grid_params)
```
This syntax works because the `grid_params` node has only one output, so Pyiron automatically connects to that output. This is the recommended approach for single-output nodes.

**For multi-output nodes**:
```python
@as_function_node
def SplitData(array: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Splits an array into even and odd indexed elements"""
    even_elements = array[::2]  # Named variable for return
    odd_elements = array[1::2]  # Named variable for return
    return even_elements, odd_elements  # Output ports will be "even_elements" and "odd_elements"

# Create a node with multiple outputs
wf.split = SplitData(array=np.arange(10))  # Direct literal assignment

# Connect to specific outputs using named labels
wf.process_even = ProcessData(data=wf.split.outputs.even_elements)
wf.process_odd = ProcessData(data=wf.split.outputs.odd_elements)
```
When a node produces multiple outputs, you **must** specify which output port you want to connect to using the `.outputs` attribute followed by the specific output label.

This distinction ensures clarity about which part of a node's output is being used, especially when nodes produce multiple related values.

**Key advantage of input dataclasses**: When creating input dataclass nodes, you only need to specify parameters that differ from their defaults. In the example above, `y_min`, `y_max`, and `ny` use their default values, making the workflow definition more concise and readable.

### Running Your Workflow

Once you've defined your workflow, you can execute it in several ways:

#### Full Execution
```python
# Execute the entire workflow
wf.run()

# Now you can access results
density_result = wf.density
figure = wf.figure
```

#### Partial Execution (For Debugging)
```python
# Execute only up to a specific node
wf.grid.pull()  # Executes all nodes needed to compute wf.grid

# Now wf.grid is available, but wf.density and wf.figure are not yet computed
print(wf.grid.outputs.grid.value.X.shape)  # Access the actual grid data

# Continue execution to the next node
wf.density.pull()
```

**Why partial execution matters for debugging:**
- You can inspect intermediate results without running the entire workflow
- If an error occurs, you can pinpoint exactly which node caused it
- You can verify that each step produces the expected output before proceeding
- It's much faster to test individual components than the entire workflow

#### On-Demand Execution
```python
# Accessing a node's output automatically triggers execution
density_result = wf.density  # Executes all necessary nodes

# Subsequent accesses are fast (results are cached)
density_result_again = wf.density
```

### Understanding Node Output Structure

In Pyiron Workflow, when you create a node like `wf.grid = CreateGrid(...)`, `wf.grid` is not the actual data but a **node object** that manages the computation. To access the actual data, you need to navigate through:

```python
# Get the actual Grid2D object produced by the node
grid_data = wf.grid.outputs.grid.value

# Now you can access its fields
print(grid_data.X.shape)
```

The structure is:
- `wf.grid` - The node object
- `.outputs` - Contains all output ports
- `.grid` - The specific output port (named after the return variable)
- `.value` - The actual data value produced by the node

**Important:** When connecting nodes in a workflow, you should connect the node objects directly, not their internal data:

```python
# ✅ CORRECT - connecting node objects
wf.density = ComputeDensity(grid=wf.grid, amplitude=0.5)

# ❌ INCORRECT - trying to connect internal data
wf.density = ComputeDensity(grid=wf.grid.outputs.grid.value, amplitude=0.5)
```

The workflow system automatically handles the data flow between nodes. Only when you need to access results for analysis or visualization should you use the `.outputs.port_name.value` pattern.

### Key Construction Rules

#### 1. All Computation Must Occur Within Nodes

The workflow assembly code itself should contain **no computations** - only node connections:

```python
# First define the necessary nodes
@as_function_node
def ComputeDensity(grid: Grid2D, amplitude: float) -> DensityResults:
    """Computes density with the given amplitude"""
    density = amplitude * np.sin(grid.X) * np.cos(grid.Y)
    
    results = DensityResults().dataclass()
    results.density = density
    results.grid = grid
    return results  # Output port will be "results"

# ✅ Correct - computation that evaluates to a basic type
wf.density = ComputeDensity(amplitude=np.sqrt(2.0))  # This is acceptable

# ❌ Worse practice - magic number
wf.density = ComputeDensity(amplitude=1.41421356237)  # Hard to understand what this number represents

# ✅ Best practice - create a node for the computation if you want to track the recipe
@as_function_node
def CalculateAmplitude(base: float) -> float:
    result = np.sqrt(base)  # Named variable for return
    return result  # Output port will be "result"

wf.amplitude = CalculateAmplitude(base=2.0)
wf.density = ComputeDensity(grid=wf.grid, amplitude=wf.amplitude)
```

**Important clarification**: Expressions like `np.sqrt(2.0)` are acceptable in workflow assembly because they evaluate directly to a basic type (`float`). This is preferable to using magic numbers because:
1. It's self-documenting (you can see where the value comes from)
2. It's more precise (avoids rounding errors from manual calculation)
3. It's more maintainable (changing the base value is straightforward)

However, if you want the workflow to track *how* a parameter was derived (for provenance and reproducibility), create a dedicated node as shown in the best practice example.

#### 2. Inputs Must Be Basic Literals or Node Outputs

Workflow connections should only use:
- Basic literals (`int`, `float`, `str`, `bool`)
- `Literal` or `Optional[Literal]` values
- Outputs from other nodes in the workflow

```python
# First define the necessary nodes
@as_function_node
def ComputeDensity(grid: Grid2D, amplitude: float) -> DensityResults:
    """Computes density with the given amplitude"""
    density = amplitude * np.sin(grid.X) * np.cos(grid.Y)
    
    results = DensityResults().dataclass()
    results.density = density
    results.grid = grid
    return results  # Output port will be "results"

# ✅ Correct - expression that evaluates to a basic type
wf.density = ComputeDensity(amplitude=1.0 / np.sqrt(2.0))

# ❌ Wrong - complex object creation
wf.density = ComputeDensity(amplitude=np.array([1.0, 2.0, 3.0]))  # Should be in a node
```

#### 3. Explicit Connection Semantics

Each workflow attribute assignment represents a connection between nodes:

```python
# First define the necessary nodes
@as_function_node
def CreateGrid(params: GridParams) -> Grid2D:
    """Creates a 2D grid based on parameters"""
    x = np.linspace(params.x_min, params.x_max, params.nx)
    y = np.linspace(params.y_min, params.y_max, params.ny)
    X, Y = np.meshgrid(x, y, indexing="ij")
    
    grid = Grid2D().dataclass()
    grid.x = x
    grid.y = y
    grid.X = X
    grid.Y = Y
    return grid  # Output port will be "grid"

# This creates a connection where:
# - The output of CreateGrid becomes available as wf.grid
# - The params input of CreateGrid receives wf.grid_params
wf.grid_params = GridParams(x_min=0.0, x_max=10.0, nx=100)
wf.grid = CreateGrid(params=wf.grid_params)
```

### Dataclass Handling in Workflows

When working with dataclass nodes, follow these specific patterns:

#### Input Dataclasses (Configuration)
```python
# First define the necessary nodes
@as_inp_dataclass_node
class GridParams:
    """Parameters for creating a computational grid"""
    x_min: float = 0.0
    x_max: float = 10.0
    nx: int = 100
    ny: int = 100

# Create input dataclass with only non-default parameters
wf.grid_params = GridParams(
    x_min=0.0,
    x_max=2 * np.pi,
    nx=200  # Only specify values that differ from defaults
)
```

#### Output Dataclasses (Results)
```python
# First define the necessary nodes
@as_out_dataclass_node
class Grid2D:
    """Holds a 2D grid structure"""
    x: DataArray = EmptyArrayField
    y: DataArray = EmptyArrayField
    X: DataArray = EmptyArrayField
    Y: DataArray = EmptyArrayField

@as_function_node
def CreateGrid(params: GridParams) -> Grid2D:
    """Creates a 2D grid based on parameters"""
    x = np.linspace(params.x_min, params.x_max, params.nx)
    y = np.linspace(params.y_min, params.y_max, params.ny)
    X, Y = np.meshgrid(x, y, indexing="ij")
    
    grid = Grid2D().dataclass()
    grid.x = x
    grid.y = y
    grid.X = X
    grid.Y = Y
    return grid  # Output port will be "grid"

@as_out_dataclass_node
class DensityResults:
    """Holds density calculation results"""
    density: DataArray = EmptyArrayField
    grid: Grid2D = EmptyArrayField

@as_function_node
def ComputeDensity(grid: Grid2D, amplitude: float = 1.0) -> DensityResults:
    """Computes a sample density field on the grid"""
    density = amplitude * np.sin(grid.X) * np.cos(grid.Y)
    
    results = DensityResults().dataclass()
    results.density = density
    results.grid = grid
    return results  # Output port will be "results"

# Connect nodes using dataclass outputs
wf.grid = CreateGrid(params=wf.grid_params)
wf.density = ComputeDensity(grid=wf.grid, amplitude=0.5)
```

**Important:** Always connect the entire dataclass object, not individual fields:

```python
# ❌ Wrong - breaking the Grid2D dataclass by extracting individual fields
wf.plot = PlotDensity(x=wf.grid.x, y=wf.grid.y, density=wf.density.density)

# ✅ Correct - passing the entire Grid2D dataclass
wf.plot = PlotDensity(grid=wf.grid, density=wf.density)
```

### Multiple Output Handling

When working with nodes that produce multiple outputs, access them through named labels:

```python
# First define the necessary nodes
@as_function_node
def SplitData(array: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Splits an array into even and odd indexed elements"""
    even_elements = array[::2]  # Named variable for return
    odd_elements = array[1::2]  # Named variable for return
    return even_elements, odd_elements  # Output ports will be "even_elements" and "odd_elements"

@as_function_node
def ProcessData(data: np.ndarray) -> np.ndarray:
    """Processes the data array"""
    processed = data * 2  # Named variable for return
    return processed  # Output port will be "processed"

# Create a node with multiple outputs
wf.split = SplitData(array=np.arange(10))  # Direct literal assignment

# Connect to specific outputs using named labels
wf.process_even = ProcessData(data=wf.split.outputs.even_elements)
wf.process_odd = ProcessData(data=wf.split.outputs.odd_elements)
```

**Never use numeric indexing** to access multiple outputs:

```python
# ❌ Wrong - numeric indexing is forbidden
wf.process = ProcessData(data=wf.split.outputs[0])

# ✅ Correct - named access
wf.process = ProcessData(data=wf.split.outputs.even_elements)
```

## Workflow Execution Model

Pyiron Workflows follow a **lazy evaluation** model with **explicit execution control**:

### Execution Process

1. **Workflow Assembly**: Define all nodes and connections (no computation occurs)
2. **Validation**: The system verifies all connections are valid
3. **Execution**: When results are requested, the system executes only necessary nodes
4. **Result Delivery**: Final results are returned to the user

### Selective Execution

Pyiron intelligently executes only the nodes necessary to produce requested results:

```python
# If you only need the grid, only CreateGrid executes
grid_data = wf.grid

# If you need the density, both CreateGrid and ComputeDensity execute
density_data = wf.density
```

## Common Workflow Patterns

### Parameter Sweep Pattern

```python
import numpy as np
from pyiron_core.pyiron_workflow import Workflow, as_function_node, as_inp_dataclass_node
from pyiron_core.pyiron_workflow.data_fields import DataArray, EmptyArrayField

# Define dataclass nodes
@as_inp_dataclass_node
class GridParams:
    """Parameters for creating a computational grid"""
    x_min: float = 0.0
    x_max: float = 10.0
    nx: int = 100
    ny: int = 100

@as_out_dataclass_node
class Grid2D:
    """Holds a 2D grid structure"""
    x: DataArray = EmptyArrayField
    y: DataArray = EmptyArrayField
    X: DataArray = EmptyArrayField
    Y: DataArray = EmptyArrayField

@as_out_dataclass_node
class DensityResults:
    """Holds density calculation results"""
    density: DataArray = EmptyArrayField
    grid: Grid2D = EmptyArrayField
    amplitude: float = 0.0

# Define function nodes
@as_function_node
def CreateGrid(params: GridParams) -> Grid2D:
    """Creates a 2D grid based on parameters"""
    x = np.linspace(params.x_min, params.x_max, params.nx)
    y = np.linspace(params.y_min, params.y_max, params.ny)
    X, Y = np.meshgrid(x, y, indexing="ij")
    
    grid = Grid2D().dataclass()
    grid.x = x
    grid.y = y
    grid.X = X
    grid.Y = Y
    return grid  # Output port will be "grid"

@as_function_node
def ComputeDensity(grid: Grid2D, amplitude: float) -> DensityResults:
    """Computes a sample density field on the grid"""
    density = amplitude * np.sin(grid.X) * np.cos(grid.Y)
    
    results = DensityResults().dataclass()
    results.density = density
    results.grid = grid
    results.amplitude = amplitude
    return results  # Output port will be "results"

# Create the workflow
wf = Workflow("parameter_sweep")

# Base configuration (only non-default parameters)
wf.base_params = GridParams(x_max=2 * np.pi, nx=200)

# Multiple runs with different amplitudes
amplitudes = [0.1, 0.5, 1.0, 2.0]
for i, amp in enumerate(amplitudes):
    # Create unique attribute names for each run
    grid_attr = f"grid_{i}"
    density_attr = f"density_{i}"
    
    # Connect nodes for this parameter value
    setattr(wf, grid_attr, CreateGrid(params=wf.base_params))
    setattr(wf, density_attr, ComputeDensity(
        grid=getattr(wf, grid_attr), 
        amplitude=amp
    ))

# Execute only up to grid_1 for debugging
wf.grid_1.pull()
print(f"Grid shape: {wf.grid_1.outputs.grid.value.X.shape}")

# Execute the full workflow for amplitude 1
wf.density_1.pull()
```

### Conditional Workflow Pattern

```python
import numpy as np
from typing import Literal
from pyiron_core.pyiron_workflow import Workflow, as_function_node, as_inp_dataclass_node, as_out_dataclass_node
from pyiron_core.pyiron_workflow.data_fields import DataArray, EmptyArrayField

# Define dataclass nodes
@as_inp_dataclass_node
class AnalysisParams:
    """Parameters controlling the analysis method"""
    method: Literal["simple", "advanced"] = "simple"

@as_out_dataclass_node
class Grid2D:
    """Holds a 2D grid structure"""
    x: DataArray = EmptyArrayField
    y: DataArray = EmptyArrayField
    X: DataArray = EmptyArrayField
    Y: DataArray = EmptyArrayField

@as_out_dataclass_node
class Results:
    """Holds analysis results"""
    data: DataArray = EmptyArrayField
    method: str = ""

# Define function nodes
@as_function_node
def CreateGrid(nx: int = 100, ny: int = 100) -> Grid2D:
    """Creates a standard 2D grid"""
    x = np.linspace(0, 10, nx)
    y = np.linspace(0, 10, ny)
    X, Y = np.meshgrid(x, y, indexing="ij")
    
    grid = Grid2D().dataclass()
    grid.x = x
    grid.y = y
    grid.X = X
    grid.Y = Y
    return grid  # Output port will be "grid"

@as_function_node
def SimpleAnalysis(grid: Grid2D) -> Results:
    """Performs a simple analysis on the grid"""
    result = np.mean(grid.X, axis=1)  # Named variable for return
    
    results = Results().dataclass()
    results.data = result
    results.method = "simple"
    return results  # Output port will be "results"

@as_function_node
def AdvancedAnalysis(grid: Grid2D) -> Results:
    """Performs an advanced analysis on the grid"""
    result = np.sin(grid.X) * np.cos(grid.Y)  # Named variable for return
    
    results = Results().dataclass()
    results.data = result
    results.method = "advanced"
    return results  # Output port will be "results"

@as_function_node
def SelectResults(simple: Results, advanced: Results, method: str) -> Results:
    """Selects the appropriate results based on method"""
    selected = advanced if method == "advanced" else simple  # Named variable for return
    return selected  # Output port will be "selected"

# Create the workflow
wf = Workflow("conditional_analysis")

# Base configuration
wf.params = AnalysisParams(method="advanced")

# Create different processing paths
wf.grid = CreateGrid(nx=100)
wf.simple_results = SimpleAnalysis(grid=wf.grid)
wf.advanced_results = AdvancedAnalysis(grid=wf.grid)

# Select output based on parameter
wf.final_results = SelectResults(
    simple=wf.simple_results,
    advanced=wf.advanced_results,
    method=wf.params.method
)

# Debugging: Check intermediate results
wf.grid.pull()  # Execute only the grid creation
print(f"Grid shape: {wf.grid.outputs.grid.value.X.shape}")

wf.simple_results.pull()  # Execute simple analysis path
print(f"Simple results shape: {wf.simple_results.outputs.results.value.data.shape}")
```

## Common Mistakes to Avoid

### Incorrect Output Port Access
```python
# First define the necessary nodes
@as_out_dataclass_node
class Grid2D:
    """Holds a 2D grid structure"""
    x: DataArray = EmptyArrayField
    y: DataArray = EmptyArrayField
    X: DataArray = EmptyArrayField
    Y: DataArray = EmptyArrayField

@as_function_node
def CreateGrid(params: GridParams) -> Grid2D:
    """Creates a 2D grid based on parameters"""
    x = np.linspace(params.x_min, params.x_max, params.nx)
    y = np.linspace(params.y_min, params.y_max, params.ny)
    X, Y = np.meshgrid(x, y, indexing="ij")
    
    grid = Grid2D().dataclass()
    grid.x = x
    grid.y = y
    grid.X = X
    grid.Y = Y
    return grid  # Output port will be "grid"

# ❌ Wrong - incorrect output port name
wf.grid.pull()
grid_data = wf.grid.outputs.create_grid.value  # No such port

# ✅ Correct - using the return variable name as port name
wf.grid.pull()
grid_data = wf.grid.outputs.grid.value
```

### Numeric Indexing of Multiple Outputs
```python
# First define the necessary nodes
@as_function_node
def SplitData(array: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Splits an array into even and odd indexed elements"""
    even_elements = array[::2]  # Named variable for return
    odd_elements = array[1::2]  # Named variable for return
    return even_elements, odd_elements  # Output ports will be "even_elements" and "odd_elements"

@as_function_node
def ProcessData(data: np.ndarray) -> np.ndarray:
    """Processes the data array"""
    processed = data * 2  # Named variable for return
    return processed  # Output port will be "processed"

# ❌ Wrong - numeric indexing
wf.process = ProcessData(data=wf.split.outputs[0])

# ✅ Correct - named access
wf.process = ProcessData(data=wf.split.outputs.even_elements)
```

### Breaking Dataclass Integrity
```python
# First define the necessary nodes
@as_out_dataclass_node
class Grid2D:
    """Holds a 2D grid structure"""
    x: DataArray = EmptyArrayField
    y: DataArray = EmptyArrayField
    X: DataArray = EmptyArrayField
    Y: DataArray = EmptyArrayField

@as_out_dataclass_node
class DensityResults:
    """Holds density calculation results"""
    density: DataArray = EmptyArrayField
    grid: Grid2D = EmptyArrayField

@as_function_node
def CreateGrid(params: GridParams) -> Grid2D:
    """Creates a 2D grid based on parameters"""
    x = np.linspace(params.x_min, params.x_max, params.nx)
    y = np.linspace(params.y_min, params.y_max, params.ny)
    X, Y = np.meshgrid(x, y, indexing="ij")
    
    grid = Grid2D().dataclass()
    grid.x = x
    grid.y = y
    grid.X = X
    grid.Y = Y
    return grid  # Output port will be "grid"

@as_function_node
def ComputeDensity(grid: Grid2D, amplitude: float = 1.0) -> DensityResults:
    """Computes a sample density field on the grid"""
    density = amplitude * np.sin(grid.X) * np.cos(grid.Y)
    
    results = DensityResults().dataclass()
    results.density = density
    results.grid = grid
    return results  # Output port will be "results"

@as_function_node
def PlotDensity(grid: Grid2D, density: np.ndarray) -> Figure:
    """Plots the density field"""
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots()
    pcm = ax.pcolormesh(grid.X, grid.Y, density)
    return fig  # Output port will be "fig"

# Correct workflow setup
wf = Workflow("example")
wf.grid_params = GridParams()
wf.grid = CreateGrid(params=wf.grid_params)
wf.density = ComputeDensity(grid=wf.grid, amplitude=0.5)

# ❌ Wrong - breaking the Grid2D dataclass by extracting individual fields
wf.plot = PlotDensity(x=wf.grid.x, y=wf.grid.y, density=wf.density.density)

# ✅ Correct - passing the entire Grid2D dataclass
wf.plot = PlotDensity(grid=wf.grid, density=wf.density.outputs.results.value.density)
```

### External File Operations
```python
# First define the necessary nodes
@as_function_node
def LoadData(filename: str) -> np.ndarray:
    """Loads data from a file"""
    import numpy as np
    data = np.loadtxt(filename)  # Named variable for return
    return data  # Output port will be "data"

@as_function_node
def CreateGrid(data: np.ndarray) -> Grid2D:
    """Creates a grid from loaded data"""
    # Implementation would use the loaded data
    grid = Grid2D().dataclass()
    # ...
    return grid  # Output port will be "grid"

# ❌ Wrong - direct file access
import numpy as np
data = np.loadtxt("input.dat")
wf.grid = CreateGrid(data=data)

# ✅ Correct - file operations in dedicated nodes
wf.loader = LoadData(filename="input.dat")
wf.grid = CreateGrid(data=wf.loader)
```

## Why This Workflow Model Matters

The Pyiron Workflow approach delivers significant advantages for scientific computing:

- **Reproducibility**: The complete computational procedure is captured in an executable form, making studies reproducible by others.

- **Modifiability**: Individual components can be updated without rewriting the entire workflow, supporting method evolution.

- **Transparency**: The workflow structure makes data dependencies explicit, enhancing scientific reviewability.

- **Error Reduction**: Strict connection rules prevent common errors like incompatible data connections.

- **Collaboration Support**: Well-structured workflows are easier to share and understand among research teams.

- **Provenance Tracking**: The system automatically tracks how results were generated, supporting scientific validation.

- **Debugging Efficiency**: The ability to execute workflows partially (`wf.grid.pull()`) makes identifying and fixing issues significantly easier.

By following these workflow construction principles, researchers can create computational pipelines that are not just functional, but also scientifically rigorous, transparent, and shareable - transforming code into a meaningful part of the scientific record.