# Higher-Order Nodes: Functional Programming in Pyiron Workflow

Pyiron Workflow supports a powerful functional programming paradigm where nodes can operate on other nodes, not just data. This advanced feature enables the creation of dynamic, reusable workflow patterns that can adapt to different computational scenarios, going beyond simple linear data flows.

## What Are Higher-Order Nodes?

In functional programming, a higher-order function is one that:
1. Takes one or more functions as arguments, or
2. Returns a function as its result

Similarly, in Pyiron Workflow, a higher-order node:
1. Takes one or more **nodes** as inputs, or
2. Returns a **node** as its output

This capability allows for sophisticated workflow patterns like:
- Looping constructs
- Conditional execution paths
- Workflow templates
- Dynamic workflow generation
- Recursive computations

## Defining Higher-Order Nodes

To define a higher-order node, you need to:

1. Import the `Node` type for type annotations
2. Use `Node` in your type hints
3. Work with the node objects in your function body

Here's the basic structure:

```python
from pyiron_core.pyiron_workflow import Node  # Import the Node type

@as_function_node
def MyHigherOrderNode(input_node: Node, parameter: float) -> Node:
    # Function body that operates on input_node
    # ...
    return modified_node
```

## Output Port Naming in Higher-Order Nodes

**Critical Insight:** When working with higher-order nodes, output port names follow specific rules that are essential for proper connections:

1. **For regular function nodes**, output ports are named after:
   - The return variable name (e.g., `return result` → port name "result")
   - An explicit name in the decorator (e.g., `@as_function_node("value")` → port name "value")

2. **When passing nodes to higher-order nodes**, you must specify the exact output port:
   ```python
   # CORRECT - specifying the output port
   wf.result = loop_until(
       recursive_function=wf.recursive_func.outputs.new_x,
       max_steps=40
   )
   
   # INCORRECT - missing output port specification
   wf.result = loop_until(
       recursive_function=wf.recursive_func,  # Missing .outputs.port_name
       max_steps=40
   )
   ```

3. **Higher-order node outputs** follow the same naming rules as regular nodes:
   - Based on return variable name or decorator specification
   - Must be accessed using the standard output structure

## Example: Looping Construct

The `loop_until` function is a perfect example of a higher-order node that implements a looping construct:

```python
from pyiron_core.pyiron_workflow import Node

@as_function_node
def loop_until(recursive_function: Node, max_steps: int = 10) -> float:
    """Executes a recursive function until a condition is met"""
    # Get the initial value from the recursive function's inputs
    x = recursive_function.inputs.x.value
    
    for i in range(max_steps):
        # Execute the recursive function with current value
        x, break_condition = recursive_function(x)
        
        if break_condition:
            break
    
    return x  # Output port will be named "x" (return variable name)
```

### Using the Looping Construct

Here's how to use this higher-order node in a workflow:

```python
# Define a recursive function node
@as_function_node
def recursive_step(x: float) -> tuple[float, bool]:
    """Computes next value and determines if we should stop"""
    new_x = x * 0.9  # Reduce by 10% each step
    should_stop = new_x < 0.1  # Stop when value drops below 0.1
    return new_x, should_stop  # Output ports: "new_x" and "should_stop"

# Create workflow
wf = Workflow("looping_example")

# Set up the recursive function with initial value
wf.recursive_func = recursive_step(x=1.0)

# Use the loop_until higher-order node
wf.result = loop_until(
    recursive_function=wf.recursive_func.outputs.new_x,  # Must specify output port
    max_steps=40
)

# Execute workflow
wf.run()

# Access the result
result_value = wf.result.outputs.x.value  # Output port "x" from loop_until
```

**Key Points:**
- The recursive function returns two values, creating output ports "new_x" and "should_stop"
- When connecting to the higher-order node, we must specify which output port to use
- The higher-order node's output port is named "x" (from the return variable)
- Results are accessed through the standard output structure

## Benefits of Higher-Order Nodes

### 1. Workflow Reusability
Create workflow templates that can be applied to different computational problems:

```python
# Define a generic optimization pattern
@as_function_node
def optimize(objective_function: Node, initial_guess: float, max_iterations: int = 100) -> float:
    """Optimizes a function using gradient descent"""
    x = initial_guess
    learning_rate = 0.1
    
    for i in range(max_iterations):
        # Compute gradient using finite differences
        f_x = objective_function(x)
        f_x_plus = objective_function(x + 1e-5)
        gradient = (f_x_plus - f_x) / 1e-5
        
        # Update position
        x = x - learning_rate * gradient
        
        # Check for convergence
        if abs(gradient) < 1e-6:
            break
    
    return x  # Output port will be named "x"

# Use with different objective functions
@as_function_node("result")
def quadratic_function(x: float) -> float:
    return x**2 + 2*x + 1

@as_function_node("result")
def sine_function(x: float) -> float:
    return np.sin(x)

# Create optimization workflows
wf_quadratic = Workflow("quadratic_opt")
wf_quadratic.objective = quadratic_function(x=0.0)
wf_quadratic.result = optimize(
    objective_function=wf_quadratic.objective.outputs.result,  # Specify output port
    initial_guess=5.0
)

# Access result
result_value = wf_quadratic.result.outputs.x.value
```

### 2. Dynamic Behavior
Generate workflows based on runtime conditions:

```python
@as_function_node
def select_computation(use_accurate: bool, x: float) -> float:
    """Selects between fast and accurate computation methods"""
    if use_accurate:
        # Create and execute accurate computation node
        accurate_node = accurate_computation(x=x)
        return accurate_node.run()
    else:
        # Create and execute fast computation node
        fast_node = fast_computation(x=x)
        return fast_node.run()

# This node dynamically creates and executes different computation paths
# based on the 'use_accurate' parameter
```

### 3. Abstraction
Hide complex workflow patterns behind simple interfaces:

```python
@as_function_node
def monte_carlo_simulation(
    model_function: Node, 
    num_samples: int = 1000,
    random_seed: int = 42
) -> np.ndarray:
    """Runs Monte Carlo simulation using the provided model"""
    np.random.seed(random_seed)
    results = []
    
    for _ in range(num_samples):
        # Generate random inputs for the model
        inputs = np.random.uniform(-1, 1, size=model_function.input_dimension)
        
        # Execute the model with these inputs
        result = model_function(*inputs)
        results.append(result)
    
    return np.array(results)  # Output port will be named "results"

# Usage becomes simple despite the complex underlying pattern
wf.result = monte_carlo_simulation(
    model_function=wf.physics_model.outputs.result,  # Specify output port
    num_samples=5000
)

# Access result
simulation_results = wf.result.outputs.results.value
```

## Advanced Example: Conditional Execution

Here's an example showing conditional execution paths:

```python
from pyiron_core.pyiron_workflow import Node

@as_function_node
def conditional_execute(condition: bool, true_node: Node, false_node: Node) -> float:
    """Executes true_node if condition is True, otherwise executes false_node"""
    if condition:
        return true_node.run()
    else:
        return false_node.run()

# Define two different computation paths
@as_function_node("result")
def fast_computation(x: float) -> float:
    """Fast but less accurate computation"""
    return x * 2

@as_function_node
def accurate_computation(x: float) -> float:
    """Slow but more accurate computation"""
    import time
    time.sleep(0.1)  # Simulate expensive computation
    return x * x

# Create workflow
wf = Workflow("conditional_example")
wf.x = 2.0
wf.use_accurate = True  # Switch between computation methods

# Set up the computation nodes
wf.fast = fast_computation(x=wf.x)
wf.accurate = accurate_computation(x=wf.x)

# Use conditional execution
wf.result = conditional_execute(
    condition=wf.use_accurate,
    true_node=wf.accurate.outputs.result,  # Must specify output port
    false_node=wf.fast.outputs.result      # Must specify output port
)

# Execute workflow
wf.run()

# Access result
result_value = wf.result.outputs.node.value
```

## Output Access Pattern for Higher-Order Nodes

After executing a workflow with higher-order nodes, you can access results through the structured output system:

```python
# For a higher-order node that returns a single value
result_value = wf.result.outputs.x.value  # Where "x" is the output port name

# For a higher-order node that returns multiple values
# (would follow the same pattern as regular multiple-output nodes)
```

The complete structure is:
- `wf.result` - The higher-order node instance
- `.outputs` - Contains all output ports of the higher-order node
- `.x` - The specific output port (name depends on return variable)
- `.value` - The actual data value produced

**Important:** The output port name for higher-order nodes follows the same rules as regular nodes:
- Based on the return variable name
- Or explicitly specified in the decorator

## Common Mistakes to Avoid

### 1. Missing Output Port Specification

```python
# ❌ Wrong - missing output port specification when connecting nodes
wf.result = loop_until(
    recursive_function=wf.recursive_func,  # Missing .outputs.port_name
    max_steps=40
)

# ✅ Correct - specifying the exact output port
wf.result = loop_until(
    recursive_function=wf.recursive_func.outputs.new_x,
    max_steps=40
)
```

### 2. Incorrect Output Access

```python
# ❌ Wrong - incorrect output port name
result_value = wf.result.outputs.result.value

# ✅ Correct - using the actual output port name
result_value = wf.result.outputs.x.value  # Based on return variable name
```

### 3. Inconsistent Output Port Naming

```python
# ❌ Wrong - inconsistent output port naming
@as_function_node
def process_value(x: float) -> float:
    result = x * 2
    return result  # Output port will be "result"

@as_function_node
def optimize(objective_function: Node, initial_guess: float) -> float:
    x = initial_guess
    # ...
    return x  # Output port will be "x" (inconsistent with above)

# ✅ Better - consistent output port naming
@as_function_node("value")
def process_value(x: float) -> float:
    result = x * 2
    return result  # Output port will be "value"

@as_function_node("value")
def optimize(objective_function: Node, initial_guess: float) -> float:
    x = initial_guess
    # ...
    return x  # Output port will be "value"
```

## Best Practices for Higher-Order Nodes

### 1. Explicit Output Port Naming
Use explicit output port names with the decorator for clarity and consistency:

```python
@as_function_node("value")
def compute_something(x: float) -> float:
    result = x * x
    return result
```

This ensures predictable output port names that are easier to work with in higher-order nodes.

### 2. Document Output Ports
Clearly document the output ports of your nodes:

```python
@as_function_node
def SplitData(array: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Splits an array into even and odd indexed elements
    
    Returns:
    - even_elements: Elements at even indices
    - odd_elements: Elements at odd indices
    """
    even_elements = array[::2]
    odd_elements = array[1::2]
    return even_elements, odd_elements
    # Output ports: "even_elements" and "odd_elements"
```

### 3. Use Meaningful Return Variable Names
Choose return variable names that clearly indicate their purpose:

```python
# ❌ Unclear
return x, delta

# ✅ Clear
final_value = x
convergence_delta = delta
return final_value, convergence_delta
# Output ports: "final_value" and "convergence_delta"
```

### 4. Test with Multiple Scenarios
Test higher-order nodes with various input nodes to ensure robustness:

```python
def test_loop_until():
    # Test with different initial values
    # Test with different convergence criteria
    # Test with maximum iterations reached
    pass
```

## Real-World Applications

### 1. Parameter Optimization
```python
@as_function_node("optimized_parameters")
def optimize_parameters(
    objective_function: Node,
    initial_parameters: np.ndarray,
    learning_rate: float = 0.01,
    max_iterations: int = 100
) -> np.ndarray:
    """Optimizes parameters using gradient descent"""
    params = initial_parameters.copy()
    
    for i in range(max_iterations):
        # Compute gradient by finite differences
        gradients = np.zeros_like(params)
        for j in range(len(params)):
            params_plus = params.copy()
            params_plus[j] += 1e-5
            loss = objective_function(params)
            loss_plus = objective_function(params_plus)
            gradients[j] = (loss_plus - loss) / 1e-5
        
        # Update parameters
        params -= learning_rate * gradients
        
        # Check convergence
        if np.linalg.norm(gradients) < 1e-6:
            break
    
    return params  # Output port will be "optimized_parameters"

# Usage
wf.objective = physics_model(parameters=initial_guess)
wf.optimized = optimize_parameters(
    objective_function=wf.objective.outputs.loss,  # Specify output port
    initial_parameters=initial_guess
)
result = wf.optimized.outputs.optimized_parameters.value
```

### 2. Adaptive Mesh Refinement
```python
@as_function_node("final_solution")
def adaptive_mesh_refinement(
    simulation_node: Node,
    error_estimator_node: Node,
    refine_mesh_node: Node,
    initial_mesh_size: float,
    max_refinements: int = 5,
    tolerance: float = 0.01
) -> float:
    """Refines mesh until solution converges"""
    mesh = initial_mesh_size
    
    for i in range(max_refinements):
        # Run simulation on current mesh
        solution = simulation_node(mesh=mesh)
        
        # Estimate error
        error = error_estimator_node(solution=solution, mesh_size=mesh)
        
        # Check if we've converged
        if error < tolerance:
            return solution
        
        # Refine mesh in high-error regions
        mesh = refine_mesh_node(mesh_size=mesh, error=error)
    
    return solution  # Output port will be "final_solution"

# Usage
wf.simulation = fluid_dynamics_simulation(mesh_size=initial_size)
wf.error_estimator = error_calculator(solution=wf.simulation)
wf.refine_mesh = mesh_refiner(mesh_size=initial_size, error=0.1)

wf.result = adaptive_mesh_refinement(
    simulation_node=wf.simulation.outputs.solution,       # Specify output ports
    error_estimator_node=wf.error_estimator.outputs.error,
    refine_mesh_node=wf.refine_mesh.outputs.refined_mesh,
    initial_mesh_size=initial_size
)
solution = wf.result.outputs.final_solution.value
```

## Conclusion

Higher-order nodes represent one of the most powerful features of Pyiron Workflow, enabling you to create sophisticated computational patterns that would be difficult or impossible with traditional workflow systems. By treating nodes as first-class objects that can be passed around and manipulated, Pyiron Workflow supports a truly functional approach to scientific computing.

When used appropriately, higher-order nodes can:
- Reduce code duplication
- Improve workflow readability
- Enable more sophisticated computational patterns
- Make your scientific workflows more adaptable and reusable

**Critical Implementation Notes:**
- Always specify exact output ports when connecting nodes (`node.outputs.port_name`)
- Be consistent with output port naming conventions
- Access results through the standard output structure (`wf.node.outputs.port.value`)
- Define all nodes at the module level (not inside functions)

As with any powerful feature, higher-order nodes should be used judiciously. Start with simple applications and gradually incorporate more complex patterns as you become comfortable with the paradigm. The ability to create dynamic, self-modifying workflows opens up new possibilities for scientific computing that go beyond traditional static workflow systems.