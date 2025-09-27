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
    """
    Display atomistic structure (ase.Atoms) using nglview.

    Task
    ----
    Visualise a static atomic structure, e.g., after building a bulk cell,
    creating a surface slab, or after a geometry optimisation. This node is
    typically used when the user wants to inspect the geometry, defects, or
    surface features directly in a Jupyter notebook.

    Parameters
    ----------
    structure: ase.Atoms
        The atomic structure to visualise.
    camera: str, optional
        Camera mode, either "orthographic" or "perspective".
    particle_size: float, optional
        Size of the rendered atoms.
    background: {"white", "black"}, optional
        Background colour of the view.
    select_atoms: np.ndarray or list, optional
        Indices of atoms to highlight.
    view_plane: list, optional
        Plane normal for the view.
    distance_from_camera: float, optional
        Distance of the camera from the structure.
    """
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


@as_function_node("animate")
def Animate(
    trajectory,
    initial_structure,
    spacefill: bool = True,
    show_cell: bool = True,
    center_of_mass: bool = False,
    particle_size: float = 0.5,
    camera: str = "orthographic",
):
    """
    Animate a series of atomic structures.

    Task
    ----
    Create an animation of a trajectory of structures, for example to
    visualise the time evolution of a molecular dynamics run, monitor defect
    migration, or generate a presentation of structural changes. This node is
    useful when the user needs a dynamic view of multiple frames rather than a
    single static plot.

    Parameters
    ----------
    trajectory : Trajectory‑like object
        An object that provides ``positions`` (e.g. a pyiron ``Trajectory``
        or any object with a ``positions`` attribute).
    initial_structure : Structure‑like object
        The reference structure that defines the atomic species,
        lattice vectors, etc.
    spacefill : bool, default=True
        If ``True`` the atoms are visualised in *space‑fill* style
        (large spheres whose radius is proportional to the atomic
        number).  If ``False`` a ball‑and‑stick representation is used.
    show_cell : bool, default=True
        Show the unit‑cell boundaries of the structure.
    center_of_mass : bool, default=False
        If ``True`` the atomic coordinates are shifted so that the
        centre‑of‑mass of each frame is at the origin.  If ``False``
        the coordinates are taken as‑is.
    particle_size : float, default=0.5
        Scaling factor for the spheres that represent the atoms.
        The actual radius is ``particle_size * atomic_number``.
    camera : {{'orthographic', 'perspective'}}, default='orthographic'
        Camera perspective to be used for the animation.
    """
    from pyiron_atomistics.atomistics.job.atomistic import Trajectory

    # ------------------------------------------------------------------
    # Build a pyiron Trajectory object from the supplied data
    # ------------------------------------------------------------------
    traj = Trajectory(positions=trajectory.positions, structure=initial_structure)

    # ------------------------------------------------------------------
    # Forward the animation options to the underlying pyiron method
    # ------------------------------------------------------------------
    return traj.animate_structures(
        spacefill=spacefill,
        show_cell=show_cell,
        center_of_mass=center_of_mass,
        particle_size=particle_size,
        camera=camera,
    )
