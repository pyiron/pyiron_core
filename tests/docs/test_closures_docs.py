from typing import Tuple

import numpy as np

from pyiron_core.pyiron_workflow import (
    Node,
    Workflow,
    as_function_node,
)

# =============================================================================
# NODE DEFINITIONS (MUST BE AT MODULE LEVEL, NOT INSIDE TEST FUNCTIONS)
# =============================================================================


# Basic computational nodes
@as_function_node
def recursive_step(x: float) -> Tuple[float, bool]:
    """Computes next value and determines if we should stop"""
    new_x = x * 0.9  # Reduce by 10% each step
    should_stop = new_x < 0.1  # Stop when value drops below 0.1
    return new_x, should_stop


@as_function_node("result")
def quadratic_function(x: float) -> float:
    """Simple quadratic function for optimization"""
    return x**2 + 2 * x + 1


@as_function_node("result")
def sine_function(x: float) -> float:
    """Sine function for optimization"""
    return np.sin(x)


@as_function_node("result")
def fast_computation(x: float) -> float:
    """Fast but less accurate computation"""
    return x * 2


@as_function_node
def accurate_computation(x: float) -> float:
    """Slow but more accurate computation"""
    # Simulate expensive computation
    result = x * x
    return result


@as_function_node
def gradient_descent_step(x: float, learning_rate: float = 0.01) -> float:
    """One step of gradient descent for x^2 function"""
    # Derivative of x^2 is 2x
    gradient = 2 * x
    new_x = x - learning_rate * gradient
    return new_x


@as_function_node("result")
def error_estimator(solution: float, mesh_size: float) -> float:
    """Simple error estimator"""
    # In a real scenario, this would compute actual error
    return solution * mesh_size


@as_function_node("result")
def refine_mesh(mesh_size: float, error: float) -> float:
    """Refine mesh based on error"""
    # In a real scenario, this would refine specific regions
    return mesh_size * 0.5  # Halve the mesh size


@as_function_node("result")
def simulation_on_mesh(mesh_size: float) -> float:
    """Simple simulation on a mesh"""
    # In a real scenario, this would run a full simulation
    return 1.0 / mesh_size  # Example solution


# Higher-order nodes
@as_function_node
def loop_until(recursive_function: Node, max_steps: int = 10) -> float:
    """Executes a recursive function until a condition is met"""
    # Get the initial value from the recursive function's inputs
    x = recursive_function.inputs.x.value

    for _ in range(max_steps):
        # Execute the recursive function with current value
        x, break_condition = recursive_function(x)

        if break_condition:
            break

    return x  # Return the final value


@as_function_node
def optimize(
    objective_function: Node, initial_guess: float, max_iterations: int = 100
) -> float:
    """Optimizes a function using simple gradient descent"""
    x = initial_guess
    learning_rate = 0.1

    for _ in range(max_iterations):
        # Compute gradient using finite differences
        f_x = objective_function(x)
        f_x_plus = objective_function(x + 1e-5)
        gradient = (f_x_plus - f_x) / 1e-5

        # Update position
        x = x - learning_rate * gradient

        # Check for convergence
        if abs(gradient) < 1e-6:
            break

    return x


@as_function_node
def conditional_execute(condition: bool, true_node: Node, false_node: Node) -> float:
    """Executes true_node if condition is True, otherwise executes false_node"""
    if condition:
        node = true_node.run()
    else:
        node = false_node.run()
    return node


@as_function_node
def convergence_checker(
    iterative_function: Node, tolerance: float = 1e-6, max_iterations: int = 100
) -> float:
    """
    Executes an iterative function until convergence.

    Parameters:
    - iterative_function: A node that takes current value and returns (new_value, delta)
    - tolerance: Convergence threshold for delta
    - max_iterations: Maximum number of iterations

    Returns:
    - The converged value
    """
    # Get initial value from inputs
    x = iterative_function.inputs.x.value

    for _ in range(max_iterations):
        # Execute the function
        x, delta = iterative_function(x)

        # Check convergence
        if abs(delta) < tolerance:
            break

    return x


@as_function_node
def optimize_parameters(
    objective_function: Node,
    initial_parameters: float,
    learning_rate: float = 0.01,
    max_iterations: int = 100,
) -> float:
    """Optimizes parameters using gradient descent"""
    params = initial_parameters

    for _ in range(max_iterations):
        # Compute gradient by finite differences
        loss = objective_function(params)
        loss_plus = objective_function(params + 1e-5)
        gradient = (loss_plus - loss) / 1e-5

        # Update parameters
        params = params - learning_rate * gradient

        # Check convergence
        if abs(gradient) < 1e-6:
            break

    return params


@as_function_node
def adaptive_mesh_refinement(
    simulation_node: Node,
    error_estimator_node: Node,
    refine_mesh_node: Node,
    initial_mesh_size: float,
    max_refinements: int = 5,
    tolerance: float = 0.01,
) -> float:
    """Refines mesh until solution converges"""
    mesh = initial_mesh_size

    for _ in range(max_refinements):
        # Run simulation on current mesh
        solution = simulation_node(mesh=mesh)

        # Estimate error
        error = error_estimator_node(solution=solution, mesh_size=mesh)

        # Check if we've converged
        if error < tolerance:
            return solution

        # Refine mesh
        mesh = refine_mesh_node(mesh_size=mesh, error=error)

    return solution  # Return best solution found


# Create a simple iterative function that converges to 0
@as_function_node
def iterative_function(x: float) -> Tuple[float, float]:
    new_x = x * 0.5  # Halve the value each step
    delta = abs(new_x - x)  # Change in value
    return new_x, delta


# Define a simple objective function (x^2)
@as_function_node("result")
def objective_function(x: float) -> float:
    return x * x


# =============================================================================
# TEST FUNCTIONS
# =============================================================================


def test_loop_until():
    """Test the loop_until higher-order node example"""

    # Create workflow
    wf = Workflow("looping_example")

    # Set up the recursive function with initial value
    wf.recursive_func = recursive_step(x=1.0)

    # Use the loop_until higher-order node
    wf.result = loop_until(
        recursive_function=wf.recursive_func.outputs.new_x, max_steps=40
    )

    # Execute workflow
    wf.run()

    # Verify result
    result_value = wf.result.outputs.x.value
    assert result_value < 0.1  # Should have converged below threshold
    assert result_value > 0.0  # Should be positive


def test_optimization_pattern():
    """Test the optimization pattern example"""

    # Create workflow for quadratic function
    wf_quad = Workflow("quadratic_opt")
    wf_quad.objective = quadratic_function(x=0.0)
    wf_quad.result = optimize(objective_function=wf_quad.objective, initial_guess=5.0)

    # Execute workflow
    wf_quad.run()

    # Verify result (minimum of x^2 + 2x + 1 is at x = -1)
    result_value = wf_quad.result.outputs.x.value
    assert abs(result_value + 1.0) < 1e-5  # Should be close to -1

    # Create workflow for sine function
    wf_sine = Workflow("sine_opt")
    wf_sine.objective = sine_function(x=0.0)
    wf_sine.result = optimize(objective_function=wf_sine.objective, initial_guess=2.0)

    # Execute workflow
    wf_sine.run()

    # Verify result (minimum of sin(x) near x=3*pi/2 ≈ 4.71)
    result_value = wf_sine.result.outputs.x.value
    assert abs(result_value - 4.71238898) < 0.1  # Should be close to 3*pi/2


def test_conditional_execution():
    """Test the conditional execution example"""

    # Create workflow
    wf = Workflow("conditional_example")

    # Set up the computation nodes
    wf.fast = fast_computation(x=2)
    wf.accurate = accurate_computation(x=2)

    # Use conditional execution
    wf.result = conditional_execute(
        condition=True,
        true_node=wf.accurate.outputs.result,
        false_node=wf.fast.outputs.result,
    )

    # Execute workflow
    wf.run()

    # Verify accurate computation was used
    result_value = wf.result.outputs.node.value
    assert result_value == 4.0  # 2.0 * 2.0 = 4.0

    # Test with fast computation
    wf2 = Workflow("conditional_example_fast")

    wf2.fast = fast_computation(x=2)
    wf2.accurate = accurate_computation(x=2)

    wf2.result = conditional_execute(
        condition=False, true_node=wf2.accurate, false_node=wf2.fast
    )

    # Execute workflow
    wf2.run()

    # Verify fast computation was used
    result_value = wf2.result.outputs.node.value
    assert result_value == 4.0  # 2.0 * 2 = 4.0


def test_convergence_checker():
    """Test the convergence_checker higher-order node"""

    # Create workflow
    wf = Workflow("convergence_test")
    wf.iterative_func = iterative_function(x=1.0)
    wf.result = convergence_checker(
        iterative_function=wf.iterative_func.outputs.new_x,
        tolerance=1e-6,
        max_iterations=100,
    )

    # Execute workflow
    wf.run()

    # Verify convergence
    result_value = wf.result.outputs.x.value
    assert abs(result_value) < 1e-6  # Should have converged close to 0


def test_parameter_optimization():
    """Test the parameter optimization example"""

    # Create workflow
    wf = Workflow("optimization_test")

    wf.objective = objective_function(x=0.0)
    wf.result = optimize_parameters(
        objective_function=wf.objective, initial_parameters=5.0, learning_rate=0.1
    )

    # Execute workflow
    wf.run()

    # Verify optimization result (minimum at x=0)
    result_value = wf.result.outputs.params.value
    assert abs(result_value) < 1e-5  # Should be close to 0


def test_adaptive_mesh_refinement():
    """Test the adaptive mesh refinement example"""

    # Create workflow
    wf = Workflow("mesh_refinement_test")

    # Set up the nodes
    wf.simulation = simulation_on_mesh(mesh_size=1.0)
    wf.error_estimator = error_estimator(solution=1.0, mesh_size=1.0)
    wf.refine_mesh = refine_mesh(mesh_size=1.0, error=1.0)

    # Use adaptive mesh refinement
    wf.result = adaptive_mesh_refinement(
        simulation_node=wf.simulation,
        error_estimator_node=wf.error_estimator,
        refine_mesh_node=wf.refine_mesh,
        initial_mesh_size=1.0,
        max_refinements=10,
        tolerance=0.01,
    )

    # Execute workflow
    wf.run()

    # Verify refinement occurred (solution should be larger with smaller mesh)
    result_value = wf.result.outputs.solution.value
    assert result_value >= 1.0  # Should be greater than initial solution
