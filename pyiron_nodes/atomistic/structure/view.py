from pyiron_workflow import as_function_node

from ase import Atoms as _Atoms
import numpy as _np
from typing import Optional


@as_function_node("plot")
def Plot3d(
    structure: _Atoms,
    camera: str = "orthographic",
    particle_size: Optional[int | float] = 1.0,
    select_atoms: Optional[_np.ndarray] = None,
    view_plane: _np.ndarray = _np.array([0, 0, 1]),
    distance_from_camera: Optional[int | float] = 1.0,
):
    """Display atomistic structure (ase.Atoms) using nglview"""
    from structuretoolkit import plot3d

    return structure.plot3d(
        # structure=structure,
        camera=camera,
        particle_size=particle_size,
        select_atoms=select_atoms,
        view_plane=view_plane,
        distance_from_camera=distance_from_camera,
    )
