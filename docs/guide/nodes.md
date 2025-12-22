# Nodes

Nodes are the fundamental building blocks of a Pyiron Workflow. They represent individual computational steps that can be connected to form a complete scientific workflow. Unlike regular functions, nodes are specialized classes that must be instantiated and executed following a specific pattern.

## Node Execution Model

In Pyiron Workflow, **nodes are not regular functions** but specialized classes that must be instantiated and executed. This is a critical distinction that affects how you use nodes:

```python
# Create a node instance
square = SquareNumber(x=2.0)

# Execute the node
result = square.run()

# Access the output
output_value = square.outputs.result.value
```

The complete workflow is:
1. **Instantiate** the node with parameters (`square = SquareNumber(x=2.0)`)
2. **Execute** the node (`result = square.run()`)
3. **Access** outputs through the structured output system (`square.outputs.result.value`)

**Important:** Simply calling a node like a function (`SquareNumber(x=2.0)`) does not execute it - it only creates a node instance.

## Function Nodes

Function nodes perform computations and are created using the `@as_function_node` decorator. Each function node must follow these key requirements:

### Output Naming Requirement

Every function node **must have named outputs**. There are two primary ways to provide output names:

#### Option 1: Return named variables (recommended)
```python
@as_function_node
def SquareNumber(x: float) -> float:
    result = x * x  # The variable name "result" becomes the output name
    return result
```

#### Option 2: Specify output name in the decorator
```python
@as_function_node("square")
def SquareNumber(x: float) -> float:
    return x * x
```

**Important:** Simply returning an expression like `x * x` without assigning it to a variable will not work, as the system needs a valid name for the output port.

### Single Return Statement Requirement

Each function node **must have exactly one return statement**. This ensures consistent output port naming and prevents potential conflicts.

```python
# ❌ Wrong - multiple return statements
@as_function_node
def ProcessValue(x: float) -> float:
    if x > 0:
        return x * 2
    else:
        return x / 2

# ✅ Correct - single return statement
@as_function_node
def ProcessValue(x: float) -> float:
    result = x * 2 if x > 0 else x / 2
    return result
```

### Type Annotations

Function nodes can use any valid Python type annotation, but there are important guidelines for different contexts:

#### For User-Facing Inputs (Workflow Endpoints)
Only these types should be used for parameters that users will directly set:
- Primitive types: `int`, `float`, `bool`, `str`
- `Literal` types for constrained choices: `Literal["option1", "option2"]`
- `Optional[Literal[...]]` for optional constrained choices

#### For Internal Connections
More complex types can be used when connecting to other nodes:
- `np.ndarray`
- Custom dataclasses (e.g., `Mesh2D`)
- Other complex types that flow between nodes

### Multiple Output Ports

Function nodes can produce multiple outputs in two ways:

#### Option 1: Return multiple named variables
```python
@as_function_node
def SplitData(array: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Splits an array into even and odd indexed elements"""
    even_elements = array[::2]
    odd_elements = array[1::2]
    return even_elements, odd_elements
```
In this case, the variable names (`even_elements`, `odd_elements`) become the output port names.

#### Option 2: Specify output labels in the decorator
```python
@as_function_node(labels=["even", "odd"])
def SplitData(array: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return array[::2], array[1::2]
```
Or using the shorthand syntax:
```python
@as_function_node(["even", "odd"])
def SplitData(array: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return array[::2], array[1::2]
```

**Important:** When accessing multiple outputs, they must be referenced by their named labels (not numeric indices).

**Critical Note:** The parameter name is `labels` (not `output_labels` as previously documented).

## Dataclass Nodes

Dataclass nodes provide structured containers for passing multiple related values between function nodes. There are two distinct types with different purposes:

### Why Separate Decorators for Input and Output Dataclasses?

The separation between `@as_inp_dataclass_node` and `@as_out_dataclass_node` serves important conceptual and technical purposes:

1. **Different Roles in the Workflow**:
   - Input dataclasses represent **configuration parameters** that users set before execution
   - Output dataclasses represent **computation results** that flow between nodes during execution

2. **Workflow Engine Requirements**:
   - The engine treats inputs and outputs differently for dependency tracking
   - Input dataclasses are typically "frozen" after workflow creation
   - Output dataclasses are generated dynamically during workflow execution

3. **User Experience**:
   - Separating inputs from outputs makes workflows more intuitive to understand
   - GUI tools can present input parameters differently from results

4. **Provenance Tracking**:
   - Distinguishing between configuration (inputs) and results (outputs) is critical for scientific reproducibility

### Input Dataclass Nodes (`@as_inp_dataclass_node`)

Use this decorator when:

- You're defining **user-configurable parameters** (not calculation results)
- The data represents **settings** that control how calculations should run
- You want values to be easily adjustable by users before workflow execution
- The data doesn't depend on other nodes' calculations
- Examples: grid specifications, simulation parameters, file paths

```python
@as_inp_dataclass_node
class GridParams:
    """Parameters for creating a computational grid"""
    x_min: float = 0.0
    x_max: float = 10.0
    y_min: float = 0.0
    y_max: float = 10.0
    nx: int = 100
    ny: int = 100
```

**Usage:**
```python
params = GridParams().run()  # Execute to get the dataclass instance
assert params.nx == 100
```

**Important:** Input dataclass nodes must be executed with `.run()`, not `.dataclass()`.

### Output Dataclass Nodes (`@as_out_dataclass_node`)

Use this decorator when:

- You're bundling **results from a calculation**
- The data will be **generated by a node** and passed to other nodes
- The data represents **intermediate or final results** of computations
- The data **depends on other nodes' outputs** or calculations
- Examples: simulation results, processed data, visualization elements

```python
@as_out_dataclass_node
class SimulationResults:
    """Holds results from a simulation run"""
    density: np.ndarray = EmptyArrayField
    potential: np.ndarray = EmptyArrayField
    energy: float = 0.0
```

### Using Dataclasses in Function Nodes

#### Creating a dataclass instance:
```python
@as_function_node
def CreateGrid(params: GridParams) -> SimulationResults:
    x = np.linspace(params.x_min, params.x_max, params.nx)
    y = np.linspace(params.y_min, params.y_max, params.ny)
    
    results = SimulationResults().dataclass()
    results.density = np.zeros((params.nx, params.ny))
    # ... other calculations
    return results
```

#### Using a dataclass as input:
```python
@as_function_node
def AnalyzeResults(results: SimulationResults) -> float:
    average_density = np.mean(results.density)
    return average_density
```

### Dataclass Usage Rules

| Context | Correct Syntax | Reason |
|---------|----------------|--------|
| Type hint | `def foo(data: SimulationResults):` | `SimulationResults` is the type descriptor |
| Return annotation | `-> SimulationResults` | Matches the factory type |
| Creating instance (inside function node) | `obj = SimulationResults().dataclass()` | Produces the actual container |
| Executing input dataclass | `params = GridParams().run()` | Gets the dataclass instance |

## Node Definition Location

**Critical Requirement:** Node definitions (both dataclass nodes and function nodes) **must be defined at the module level**, not inside functions or methods.

✅ **CORRECT** - Module-level definition:
```python
@as_function_node
def SquareNumber(x: float) -> float:
    result = x * x
    return result

# Later in your code...
square = SquareNumber(x=2.0)
square.run()
result = square.outputs.result.value
```

❌ **INCORRECT** - Definition inside a function:
```python
def process_data():
    @as_function_node
    def SquareNumber(x: float) -> float:
        result = x * x
        return result
    
    square = SquareNumber(x=2.0)
    square.run()
    return square.outputs.result.value
```

Defining nodes inside functions causes:
- Inconsistent output port naming
- Workflow execution failures
- Difficulty tracking node types
- Problems with workflow serialization

## Output Access Pattern

After executing a node, you can access its outputs through the structured `.outputs` attribute:

```python
# For a node that returns a single value
square = SquareNumber(x=2.0)
square.run()
result = square.outputs.result.value  # Access the output value

# For a node that returns multiple values
split_data = SplitData(array=np.array([1, 2, 3, 4, 5, 6]))
split_data.run()
even = split_data.outputs.even_elements.value
odd = split_data.outputs.odd_elements.value
```

The `.outputs` attribute provides:
- Named access to all output ports
- A `.value` property for each port that contains the actual data
- Consistent structure regardless of node type

**Important:** The `.run()` method returns the primary output value, but to access all outputs you must use the `.outputs.port_name.value` pattern.

## Common Mistakes to Avoid

### Incorrect Node Execution
```python
# ❌ Wrong - treating node as a regular function
result = SquareNumber(x=2.0)

# ✅ Correct - proper node execution
square = SquareNumber(x=2.0)
result = square.run()
```

### Incorrect Input Dataclass Usage
```python
# ❌ Wrong - using .dataclass() on input dataclass
params = GridParams().dataclass()

# ✅ Correct - using .run() on input dataclass
params = GridParams().run()
```

### Invalid Output Naming
```python
# ❌ Wrong - no output name
@as_function_node
def Square(x: float) -> float:
    return x * x

# ✅ Correct - named variable
@as_function_node
def Square(x: float) -> float:
    result = x * x
    return result
```

### Multiple Return Statements
```python
# ❌ Wrong - multiple return statements
@as_function_node
def Process(x: float) -> float:
    if x > 0:
        return x * 2
    return x / 2

# ✅ Correct - single return statement
@as_function_node
def Process(x: float) -> float:
    result = x * 2 if x > 0 else x / 2
    return result
```

### Incorrect Dataclass Usage
```python
# ❌ Wrong - using () in type hints
def Process(data: SimulationResults().dataclass()):

# ✅ Correct - just use the class name
def Process(data: SimulationResults):
```

### Missing Dataclass Instance Creation
```python
# ❌ Wrong - won't work
results = SimulationResults()

# ✅ Correct - must use .dataclass()
results = SimulationResults().dataclass()
```

## Why This Distinction Matters

The separation between input and output dataclasses creates a clear boundary between:

- **What the user controls** (input dataclasses)
- **What the workflow computes** (output dataclasses)

This distinction:

- Makes workflows more understandable and maintainable
- Enables better tooling support (e.g., GUIs that highlight configurable parameters)
- Improves scientific reproducibility by clearly separating configuration from results
- Helps prevent common errors like accidentally modifying input parameters during execution

For information on connecting nodes into workflows, see the [Workflow Construction](workflow.md) section.