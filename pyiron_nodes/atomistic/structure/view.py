from pyiron_workflow import as_function_node

from ase import Atoms as _Atoms
import numpy as np
from typing import Optional, Union, Literal


@as_function_node("plot")
def Plot3d(
    structure: _Atoms,
    camera: str = "orthographic",
    particle_size: float = 1.0,
    background: Literal["white", "black"] = "white",
    select_atoms: Optional[np.ndarray | list] = None,
    view_plane: Optional[list] = None,
    distance_from_camera: Optional[float] = 1.0,
):
    """Display atomistic structure (ase.Atoms) using nglview"""
    from structuretoolkit import plot3d

    if view_plane is None:
        view_plane = [1, 1, 1]

    return structure.plot3d(
        camera=camera,
        particle_size=particle_size,
        background=background,
        select_atoms=select_atoms,
        view_plane=view_plane,
        distance_from_camera=distance_from_camera,
    )
