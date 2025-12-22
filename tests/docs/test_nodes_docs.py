import numpy as np
from typing import Tuple

from pyiron_core.pyiron_workflow import (
    as_function_node,
    as_out_dataclass_node,
    as_inp_dataclass_node,
)
from pyiron_core.pyiron_workflow.data_fields import DataArray, EmptyArrayField


# =============================================================================
# NODE DEFINITIONS (MUST BE AT MODULE LEVEL, NOT INSIDE TEST FUNCTIONS)
# =============================================================================

# Function nodes
@as_function_node
def SquareNumber(x: float) -> float:
    """Squares a number"""
    result = x * x
    return result  # Output port will be named "result"

@as_function_node("square")
def SquareNumberDecorated(x: float) -> float:
    """Squares a number with explicit output name"""
    return x * x  # Output port will be named "square"

@as_function_node
def ProcessValue(x: float) -> float:
    """Processes a value based on sign"""
    result = x * 2 if x > 0 else x / 2
    return result  # Output port will be named "result"

@as_function_node
def SplitData(array: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Splits an array into even and odd indexed elements"""
    even_elements = array[::2]  # Named variable for return
    odd_elements = array[1::2]  # Named variable for return
    return even_elements, odd_elements  # Output ports will be "even_elements" and "odd_elements"

@as_function_node(labels=["even", "odd"])
def SplitDataWithLabels(array: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Splits an array with explicit output labels"""
    return array[::2], array[1::2]  # Output ports will be "even" and "odd"

# Dataclass nodes
@as_inp_dataclass_node
class CircleParams:
    """Parameters for creating a circle"""
    radius: float = 1.0
    center_x: float = 0.0
    center_y: float = 0.0

@as_out_dataclass_node
class Point2D:
    """Holds coordinates for a point in 2D space"""
    x: float = 0.0
    y: float = 0.0


# =============================================================================
# TEST FUNCTIONS
# =============================================================================

def test_function_node_output_naming():
    """Verify function node output naming examples work correctly"""
    
    # Option 1: Return named variables
    square = SquareNumber(x=2.0)
    out = square.run()
    assert square.outputs.result.value == 4.0
    assert out == 4.0
    
    # Option 2: Specify output name in decorator
    square_decorated = SquareNumberDecorated(x=3.0)
    out = square_decorated.run()
    assert square_decorated.outputs.square.value == 9.0
    assert out == 9.0


def test_single_return_statement():
    """Verify single return statement requirement"""
    
    process_value = ProcessValue(x=4.0)
    out = process_value.run()
    assert process_value.outputs.result.value == 8.0
    assert out == 8.0
    
    process_value = ProcessValue(x=-4.0)
    out = process_value.run()
    assert process_value.outputs.result.value == -2.0
    assert out == -2.0


def test_multiple_output_ports():
    """Verify multiple output ports examples"""
    
    test_array = np.array([1, 2, 3, 4, 5, 6])
    
    # Option 1: Return multiple named variables
    split_data = SplitData(array=test_array)
    out = split_data.run()
    assert np.array_equal(split_data.outputs.even_elements.value, np.array([1, 3, 5]))
    assert np.array_equal(split_data.outputs.odd_elements.value, np.array([2, 4, 6]))
    
    # Option 2: Specify output labels in decorator
    split_data_labels = SplitDataWithLabels(array=test_array)
    out = split_data_labels.run()
    assert np.array_equal(split_data_labels.outputs.even.value, np.array([1, 3, 5]))
    assert np.array_equal(split_data_labels.outputs.odd.value, np.array([2, 4, 6]))


def test_dataclass_definitions():
    """Verify dataclass node definitions work correctly"""
    
    params = CircleParams().run()
    assert params.radius == 1.0
    assert params.center_x == 0.0
    assert params.center_y == 0.0
    
    point = Point2D().dataclass()
    assert point.x == 0.0
    assert point.y == 0.0