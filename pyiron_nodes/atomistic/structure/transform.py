from __future__ import annotations

from ase import Atoms
from pyiron_workflow import as_function_node
from typing import Optional, Union


@as_function_node("structure")
def Repeat(structure: Atoms, repeat_scalar: int = 1) -> Atoms:
    """
    Repeat a crystal structure periodically along all lattice vectors.

    Parameters
    ----------
    structure : Atoms
        The ASE ``Atoms`` object to be repeated.
    repeat_scalar : int, optional
        Number of repetitions along each lattice vector (default is ``1`` – no change).

    Returns
    -------
    Atoms
        A new ``Atoms`` object containing the repeated supercell.

    Task hint
    ----------
    Use this node when the workflow requires building a larger supercell
    from a primitive cell (e.g., “create a 2×2×2 bulk Al supercell”).
    """
    # print("Repeat: ", type(repeat_scalar), repeat_scalar)
    return structure.repeat(int(repeat_scalar))


@as_function_node("structure")
def ApplyStrain(structure: Optional[Atoms] = None, strain: Union[float] = 0) -> Atoms:
    """
    Apply a homogeneous strain to a structure.

    Parameters
    ----------
    structure : Atoms, optional
        The input structure. If ``None`` the node will raise an error.
    strain : float, optional
        Strain magnitude (default ``0`` – no deformation). Positive values
        expand the lattice, negative values compress it.

    Returns
    -------
    Atoms
        A copy of the input structure with the strain applied.

    Task hint
    ----------
    Suitable for tasks such as “apply 5 % tensile strain to a bulk cell”
    or “compress a slab by 2 % before relaxation”.
    """
    struct = structure.copy()
    struct.apply_strain(strain)
    return struct


@as_function_node
def CreateVacancy(structure, index: Optional[int] = None) -> Atoms:
    """
    Remove a single atom from a structure, creating a vacancy.

    Parameters
    ----------
    structure : Atoms
        The input structure from which the atom will be removed.
    index : int, optional
        Index of the atom to delete. If ``None`` the node does nothing
        (useful as a placeholder).

    Returns
    -------
    Atoms
        A copy of the original structure with the specified atom removed.

    Task hint
    ----------
    Use when the scientific goal is “introduce a vacancy at site 5”
    or “generate a defect structure for defect formation energy calculations”.
    """
    structure = structure.copy()
    if index is not None:
        del structure[int(index)]

    return structure


@as_function_node("structure")
def RotateAxisAngle(
    structure: Atoms,
    angle: float | int = 0,
    axis: list = [0, 0, 1],
    center=(0, 0, 0),
    rotate_cell: bool = False,
) -> Atoms:
    """
    Rotate a structure around a given axis by a specified angle.

    Parameters
    ----------
    structure : Atoms
        The structure to rotate.
    angle : float or int, optional
        Rotation angle in degrees (default ``0`` – no rotation).
    axis : list of three floats, optional
        Rotation axis vector (default ``[0, 0, 1]`` – the z‑axis).
    center : tuple of three floats, optional
        Point about which the rotation is performed (default origin).
    rotate_cell : bool, optional
        If ``True`` also rotate the simulation cell (default ``False``).

    Returns
    -------
    Atoms
        A new ``Atoms`` object with the rotated coordinates (and optionally cell).

    Task hint
    ----------
    Ideal for “orient a surface normal to the x‑axis”,
    “apply a 45° tilt to a grain boundary”, or any situation where a
    specific crystallographic orientation is required.
    """

    structure_rotated = structure.copy()
    structure_rotated.rotate(a=angle, v=axis, center=center, rotate_cell=rotate_cell)
    return structure_rotated


# @as_function_node
# def ase_to_pyiron(structure: Atoms) -> Atoms:
#     """
#     Convert an ASE Atoms object to a pyiron Atoms object.

#     :param structure: ASE Atoms object.
#     :return: pyiron Atoms object.
#     """
#     from pyiron import
#     return PyironAtoms(structure)
