import unittest
from typing import Literal, Tuple

import numpy as np

from pyiron_core.pyiron_workflow import (
    Workflow,
    as_function_node,
    as_inp_dataclass_node,
    as_out_dataclass_node,
)
from pyiron_core.pyiron_workflow.data_fields import DataArray, EmptyArrayField

# =============================================================================
# NODE DEFINITIONS (MUST BE AT MODULE LEVEL, NOT INSIDE TEST FUNCTIONS)
# =============================================================================


# Dataclass nodes
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


@as_inp_dataclass_node
class ArrayParams:
    """Parameters for array creation"""

    size: int = 10


@as_inp_dataclass_node
class SimulationParams:
    """Parameters for simulation"""

    temperature: float = 300.0
    pressure: float = 1.0
    steps: int = 1000
    ensemble: Literal["NVT", "NPT"] = "NVT"


@as_out_dataclass_node
class Results:
    """Generic results container"""

    value: float = 0.0


# Function nodes
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
    return grid


@as_function_node
def ComputeDensity(grid: Grid2D, amplitude: float = 1.0) -> DensityResults:
    """Computes a sample density field on the grid"""
    density = amplitude * np.sin(grid.X) * np.cos(grid.Y)

    results = DensityResults().dataclass()
    results.density = density
    results.grid = grid
    return results


@as_function_node
def CreateArray(params: ArrayParams) -> np.ndarray:
    """Creates an array of specified size"""
    result = np.arange(params.size)
    return result


@as_function_node
def SplitArray(array: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Splits an array into even and odd indexed elements"""
    even_elements = array[::2]
    odd_elements = array[1::2]
    return even_elements, odd_elements


@as_function_node
def ProcessArray(data: np.ndarray) -> np.ndarray:
    """Processes the data array"""
    processed = data * 2
    return processed


@as_function_node
def ProcessValue(value: float) -> Results:
    """Processes a single value"""
    results = Results().dataclass()
    results.value = value
    return results


@as_function_node
def CalculateAmplitude(base: float) -> float:
    """Calculates square root of base"""
    result = np.sqrt(base)
    return result


@as_function_node
def AnalyzeGrid(grid: Grid2D) -> float:
    """Analyzes grid data"""
    result = np.mean(grid.X)
    return result


@as_function_node
def ProcessGrid(grid: Grid2D) -> Grid2D:
    """Processes grid data"""
    result = Grid2D().dataclass()
    result.X = grid.X**2
    result.Y = grid.Y**2
    return result


# =============================================================================
# TEST CASES
# =============================================================================


class TestBasicWorkflow(unittest.TestCase):
    """Test cases for basic workflow structure"""

    def test_basic_workflow_structure(self):
        """Test the basic workflow structure example from documentation"""
        wf = Workflow("test_workflow")

        wf.grid_params = GridParams(x_min=0.0, x_max=2 * np.pi, nx=200)
        wf.grid = CreateGrid(params=wf.grid_params)
        wf.compute_density = ComputeDensity(grid=wf.grid, amplitude=np.sqrt(2.0))
        wf.run()

        self.assertTrue(hasattr(wf, "grid_params"))
        self.assertTrue(hasattr(wf, "grid"))
        self.assertTrue(hasattr(wf, "compute_density"))

        wf.grid.pull()
        grid_result = wf.grid
        self.assertEqual(grid_result.outputs.grid.value.X.shape, (200, 100))

        density_result = wf.compute_density
        self.assertEqual(density_result.outputs.results.value.density.shape, (200, 100))

    def test_workflow_syntax_understanding(self):
        """Test understanding of workflow syntax (labels, connections)"""
        wf = Workflow("syntax_test")

        wf.grid_params = GridParams()
        wf.grid_1 = CreateGrid(params=wf.grid_params)
        wf.grid_2 = CreateGrid(params=wf.grid_params)

        self.assertTrue(hasattr(wf, "grid_1"))
        self.assertTrue(hasattr(wf, "grid_2"))

        wf.grid_1.pull()
        wf.grid_2.pull()

        grid_1_value = wf.grid_1.outputs.grid.value
        grid_2_value = wf.grid_2.outputs.grid.value
        np.testing.assert_array_equal(grid_1_value.X, grid_2_value.X)
        np.testing.assert_array_equal(grid_1_value.Y, grid_2_value.Y)


class TestWorkflowConnections(unittest.TestCase):
    """Test cases for workflow connections"""

    def test_single_vs_multi_output_connections(self):
        """Test single-output vs multi-output node connections"""
        wf = Workflow("output_test")

        wf.array_params = ArrayParams(size=10)
        wf.array = CreateArray(params=wf.array_params)

        wf.split = SplitArray(array=wf.array)
        wf.process_even = ProcessArray(data=wf.split.outputs.even_elements)
        wf.process_odd = ProcessArray(data=wf.split.outputs.odd_elements)

        wf.array.pull()
        array_result = wf.array.outputs.result.value
        self.assertEqual(len(array_result), 10)

        wf.process_even.pull()
        wf.process_odd.pull()

        even_result = wf.process_even.outputs.processed.value
        odd_result = wf.process_odd.outputs.processed.value

        expected_even = np.array([0, 2, 4, 6, 8]) * 2
        np.testing.assert_array_equal(even_result, expected_even)

        expected_odd = np.array([1, 3, 5, 7, 9]) * 2
        np.testing.assert_array_equal(odd_result, expected_odd)

        with self.assertRaises(ValueError):
            _ = wf.split.outputs[0]


class TestInputDataclass(unittest.TestCase):
    """Test cases for input dataclass handling"""

    def test_input_dataclass_defaults(self):
        """Test input dataclass with only non-default parameters"""
        wf = Workflow("input_test")

        wf.params = SimulationParams(temperature=350.0, steps=5000)

        params = wf.params.pull()
        self.assertEqual(params.temperature, 350.0)
        self.assertEqual(params.pressure, 1.0)
        self.assertEqual(params.steps, 5000)
        self.assertEqual(params.ensemble, "NVT")


class TestWorkflowExecution(unittest.TestCase):
    """Test cases for workflow execution methods"""

    def test_workflow_execution_methods(self):
        """Test workflow execution methods (run, pull, on-demand)"""
        wf = Workflow("execution_test")

        wf.grid_params = GridParams(nx=5, ny=5)
        wf.grid = CreateGrid(params=wf.grid_params)
        wf.processed = ProcessGrid(grid=wf.grid)

        wf.grid.pull()
        self.assertTrue(hasattr(wf.grid.outputs, "grid"))
        self.assertTrue(hasattr(wf.grid.outputs.grid, "value"))

        grid_value = wf.grid.outputs.grid.value
        self.assertEqual(grid_value.X.shape, (5, 5))

        wf.processed.pull()
        processed_result = wf.processed
        processed_value = processed_result.outputs.result.value
        self.assertEqual(processed_value.X.shape, (5, 5))

        wf2 = Workflow("execution_test2")
        wf2.grid_params = GridParams(nx=5, ny=5)
        wf2.grid = CreateGrid(params=wf2.grid_params)
        wf2.processed = ProcessGrid(grid=wf2.grid)

        wf2.run()
        self.assertTrue(hasattr(wf2.processed.outputs, "result"))
        self.assertTrue(hasattr(wf2.processed.outputs.result, "value"))

        processed_value2 = wf2.processed.outputs.result.value
        self.assertEqual(processed_value2.X.shape, (5, 5))


class TestDataclassIntegrity(unittest.TestCase):
    """Test cases for dataclass handling"""

    def test_dataclass_integrity(self):
        """Test proper dataclass handling in workflows"""
        wf = Workflow("dataclass_test")

        wf.grid_input = GridParams(nx=5, ny=5)
        wf.grid = CreateGrid(params=wf.grid_input)
        wf.density = ComputeDensity(grid=wf.grid, amplitude=0.5)

        wf.analysis = AnalyzeGrid(grid=wf.grid)

        wf.analysis.pull()
        analysis_result = wf.analysis.outputs.result.value
        self.assertIsInstance(analysis_result, float)

        with self.assertRaises(AttributeError):
            _ = wf.grid.X

        wf.grid.pull()
        grid_value = wf.grid.outputs.grid.value
        self.assertTrue(hasattr(grid_value, "X"))
        self.assertEqual(grid_value.X.shape, (5, 5))


class TestInlineComputations(unittest.TestCase):
    """Test cases for inline computations"""

    def test_inline_computations(self):
        """Test inline computations in workflow assembly"""
        wf = Workflow("computation_test")

        wf.result_sqrt = ProcessValue(value=np.sqrt(2.0))

        wf.result_sqrt.pull()
        result = wf.result_sqrt.outputs.results.value
        self.assertTrue(np.isclose(result.value, np.sqrt(2.0)))

        wf.amplitude = CalculateAmplitude(base=2.0)
        wf.result_amplitude = ProcessValue(value=wf.amplitude)

        wf.result_amplitude.pull()
        result_amp = wf.result_amplitude.outputs.results.value
        self.assertTrue(np.isclose(result_amp.value, np.sqrt(2.0)))


class TestErrorCases(unittest.TestCase):
    """Test cases for documented error conditions"""

    def test_error_cases(self):
        """Test cases that should raise errors as documented"""
        wf = Workflow("error_test")
        wf.split = SplitArray(array=np.arange(10))

        wf.split.pull()
        self.assertTrue(hasattr(wf.split.outputs, "even_elements"))

        with self.assertRaises(ValueError):
            _ = wf.split.outputs[0]

        wf.grid_input = GridParams(nx=5, ny=5)
        wf.grid = CreateGrid(params=wf.grid_input)

        with self.assertRaises(AttributeError):
            _ = wf.grid.X

        wf.grid.pull()
        grid_value = wf.grid.outputs.grid.value
        self.assertTrue(hasattr(grid_value, "X"))
        self.assertEqual(grid_value.X.shape, (5, 5))


if __name__ == "__main__":
    unittest.main()
