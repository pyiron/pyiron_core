import unittest
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
    result = x * x
    return result


@as_function_node
def gradient_descent_step(x: float, learning_rate: float = 0.01) -> float:
    """One step of gradient descent for x^2 function"""
    gradient = 2 * x
    new_x = x - learning_rate * gradient
    return new_x


@as_function_node("result")
def error_estimator(solution: float, mesh_size: float) -> float:
    """Simple error estimator"""
    return solution * mesh_size


@as_function_node("result")
def refine_mesh(mesh_size: float, error: float) -> float:
    """Refine mesh based on error"""
    return mesh_size * 0.5  # Halve the mesh size


@as_function_node("result")
def simulation_on_mesh(mesh_size: float) -> float:
    """Simple simulation on a mesh"""
    return 1.0 / mesh_size  # Example solution


# Higher-order nodes
@as_function_node
def loop_until(recursive_function: Node, max_steps: int = 10) -> float:
    """Executes a recursive function until a condition is met"""
    x = recursive_function.inputs.x.value

    for _ in range(max_steps):
        x, break_condition = recursive_function(x)
        if break_condition:
            break

    return x


@as_function_node
def optimize(
    objective_function: Node, initial_guess: float, max_iterations: int = 100
) -> float:
    """Optimizes a function using simple gradient descent"""
    x = initial_guess
    learning_rate = 0.1

    for _ in range(max_iterations):
        f_x = objective_function(x)
        f_x_plus = objective_function(x + 1e-5)
        gradient = (f_x_plus - f_x) / 1e-5

        x = x - learning_rate * gradient

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
    x = iterative_function.inputs.x.value

    for _ in range(max_iterations):
        x, delta = iterative_function(x)
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
        loss = objective_function(params)
        loss_plus = objective_function(params + 1e-5)
        gradient = (loss_plus - loss) / 1e-5

        params = params - learning_rate * gradient

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
        solution = simulation_node(mesh=mesh)
        error = error_estimator_node(solution=solution, mesh_size=mesh)

        if error < tolerance:
            return solution

        mesh = refine_mesh_node(mesh_size=mesh, error=error)

    return solution


@as_function_node
def iterative_function(x: float) -> Tuple[float, float]:
    new_x = x * 0.5  # Halve the value each step
    delta = abs(new_x - x)  # Change in value
    return new_x, delta


@as_function_node("result")
def objective_function(x: float) -> float:
    return x * x


# =============================================================================
# TEST CASES
# =============================================================================


class TestClosures(unittest.TestCase):
    """Test cases for higher-order node examples"""

    def test_loop_until(self):
        """Test the loop_until higher-order node example"""
        wf = Workflow("looping_example")
        wf.recursive_func = recursive_step(x=1.0)
        wf.result = loop_until(
            recursive_function=wf.recursive_func.outputs.new_x, max_steps=40
        )
        wf.run()

        result_value = wf.result.outputs.x.value
        self.assertLess(result_value, 0.1)
        self.assertGreater(result_value, 0.0)

    def test_optimization_pattern(self):
        """Test the optimization pattern example"""
        wf_quad = Workflow("quadratic_opt")
        wf_quad.objective = quadratic_function(x=0.0)
        wf_quad.result = optimize(
            objective_function=wf_quad.objective, initial_guess=5.0
        )
        wf_quad.run()

        result_value = wf_quad.result.outputs.x.value
        self.assertAlmostEqual(result_value, -1.0, places=5)

        wf_sine = Workflow("sine_opt")
        wf_sine.objective = sine_function(x=0.0)
        wf_sine.result = optimize(
            objective_function=wf_sine.objective, initial_guess=2.0
        )
        wf_sine.run()

        result_value = wf_sine.result.outputs.x.value
        self.assertAlmostEqual(result_value, 4.71238898, delta=0.1)

    def test_conditional_execution(self):
        """Test the conditional execution example"""
        wf = Workflow("conditional_example")
        wf.fast = fast_computation(x=2)
        wf.accurate = accurate_computation(x=2)
        wf.result = conditional_execute(
            condition=True,
            true_node=wf.accurate.outputs.result,
            false_node=wf.fast.outputs.result,
        )
        wf.run()

        result_value = wf.result.outputs.node.value
        self.assertEqual(result_value, 4.0)

        wf2 = Workflow("conditional_example_fast")
        wf2.fast = fast_computation(x=2)
        wf2.accurate = accurate_computation(x=2)
        wf2.result = conditional_execute(
            condition=False, true_node=wf2.accurate, false_node=wf2.fast
        )
        wf2.run()

        result_value = wf2.result.outputs.node.value
        self.assertEqual(result_value, 4.0)

    def test_convergence_checker(self):
        """Test the convergence_checker higher-order node"""
        wf = Workflow("convergence_test")
        wf.iterative_func = iterative_function(x=1.0)
        wf.result = convergence_checker(
            iterative_function=wf.iterative_func.outputs.new_x,
            tolerance=1e-6,
            max_iterations=100,
        )
        wf.run()

        result_value = wf.result.outputs.x.value
        self.assertLess(abs(result_value), 1e-6)

    def test_parameter_optimization(self):
        """Test the parameter optimization example"""
        wf = Workflow("optimization_test")
        wf.objective = objective_function(x=0.0)
        wf.result = optimize_parameters(
            objective_function=wf.objective, initial_parameters=5.0, learning_rate=0.1
        )
        wf.run()

        result_value = wf.result.outputs.params.value
        self.assertAlmostEqual(result_value, 0.0, places=5)

    def test_adaptive_mesh_refinement(self):
        """Test the adaptive mesh refinement example"""
        wf = Workflow("mesh_refinement_test")
        wf.simulation = simulation_on_mesh(mesh_size=1.0)
        wf.error_estimator = error_estimator(solution=1.0, mesh_size=1.0)
        wf.refine_mesh = refine_mesh(mesh_size=1.0, error=1.0)
        wf.result = adaptive_mesh_refinement(
            simulation_node=wf.simulation,
            error_estimator_node=wf.error_estimator,
            refine_mesh_node=wf.refine_mesh,
            initial_mesh_size=1.0,
            max_refinements=10,
            tolerance=0.01,
        )
        wf.run()

        result_value = wf.result.outputs.solution.value
        self.assertGreaterEqual(result_value, 1.0)


if __name__ == "__main__":
    unittest.main()
