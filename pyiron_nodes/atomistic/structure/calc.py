from __future__ import annotations

from typing import Optional

from ase import Atoms
from pyiron_workflow import as_function_node
import numpy as np


@as_function_node
def Volume(structure: Optional[Atoms] = None, per_atom: bool = False) -> float:
    # print("Calculating volume of the structure", structure)
    volume = structure.get_volume()
    # if per_atom:
    volume = volume / structure.get_number_of_atoms()
    return volume


@as_function_node
def NumberOfAtoms(structure: Optional[Atoms] = None) -> int:
    number_of_atoms = structure.get_number_of_atoms()
    return number_of_atoms()


@as_function_node
def GetDistances(
    structure: Optional[Atoms] = None,
    num_neighbors: int = 12,
    flatten: bool = True,
    # vector: bool = False,
) -> np.ndarray:
    """
    Get distances between atoms in the structure.

    :param structure: Atoms object representing the structure.
    :param num_neighbors: Number of nearest neighbors to consider.
    :param flatten: If True, return a flat list of distances.
    # :param vector: If True, return distances as vectors.
    :return: List of distances between atoms.
    """
    from structuretoolkit import get_neighbors

    distances = get_neighbors(structure, num_neighbors=num_neighbors).distances
    if flatten:
        distances = distances.flatten()

    return distances


@as_function_node
def LinearInterpolationDescriptor(
    structure: Optional[Atoms] = None,
    # descriptor: str = "distance",
    r_bins: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Interpolate a descriptor linearly over a specified range.
    :param structure: Atoms object representing the structure.
    :param num_points: Number of points in the interpolation.
    :param r_min: Start value for interpolation.
    :param r_max: Stop value for interpolation.
    :return: Interpolated values of the descriptor.
    """
    from pyiron_nodes.math import LinearBin

    if r_bins is None:
        r_bins = np.linspace(2.5, 7, 51)  # Default range if not provided 

    num_points = r_bins.size 
    distances = GetDistances()._func(
        structure=structure, num_neighbors=num_points, flatten=True
    )

    # r_bins = np.linspace(r_min, r_max, num_points)
    counts, _ = LinearBin()._func(data=distances, bin_centers=r_bins)

    return counts


@as_function_node
def GetNeighbors(
    structure: Optional[Atoms] = None,
    num_neighbors: int = 12,
    tolerance: int = 2,
    id_list: Optional[list] = None,
    cutoff_radius: float = 10.0,
    width_buffer: float = 1.2,
    mode: str = "filled",
    norm_order: int = 2,
):
    """
    Get the neighbors of each atom in the structure.

    :param structure: Atoms object representing the structure.
    :param num_neighbors: Number of nearest neighbors to consider.
    :return: List of neighbors for each atom.
    """
    from structuretoolkit import get_neighbors

    neighbors = get_neighbors(
        structure,
        num_neighbors=num_neighbors,
        tolerance=tolerance,
        id_list=id_list,
        cutoff_radius=cutoff_radius,
        width_buffer=width_buffer,
        mode=mode,
        norm_order=norm_order,
    )

    return neighbors
