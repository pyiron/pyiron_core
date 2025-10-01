"""
data_fields.py

A small utility module providing common type aliases and default factory helpers
for dataclasses using array-like fields and scalar numerics in scientific Python applications.

Contents:
- DataArray: Type alias for list or numpy.ndarray.
- Scalar: Type alias for a scalar numeric (int or float).
- OptionalScalar: Type alias for a scalar numeric or None.
- empty_array: Function returning an empty numpy array.
- EmptyArrayField: Helper function that returns a dataclass field with an empty array default.
"""

from dataclasses import field
from typing import Optional, Union

import numpy as np

#: Type alias for a scalar numeric value (int or float).
Scalar = Union[int, float]

#: Type alias for an optional scalar numeric value.
OptionalScalar = Optional[Scalar]

#: Type alias for 1D sequence data that can be a Python list or a numpy array.
DataArray = Union[list, np.ndarray]

#: Type alias for a scalar numeric value (int or float).
DataNumber = Union[int, float]

#: Type alias for an optional scalar numeric value.
OptionalDataNumber = Optional[DataNumber]


def empty_array() -> np.ndarray:
    """Return a new empty numpy array of dtype float64."""
    return np.array([], dtype=float)


def EmptyArrayField():
    """
    Convenience function for defining dataclass fields
    with an empty numpy array as the default value.
    """
    return field(default_factory=empty_array)
