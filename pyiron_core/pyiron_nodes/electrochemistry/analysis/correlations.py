from typing import Sequence
import numpy as np

from ase.atoms import Atoms
from pyiron_core.pyiron_workflow import as_function_node
from pyiron_core.pyiron_nodes.atomistic.calculator.data import OutputCalcMD


"""
rotate_water.py

Utility to rotate a water molecule (O‑H‑H) such that the bisector of the two
O‑H bonds (the vector ``OH1 + OH2``) points along the positive *y*‑axis while
the molecule stays in its original molecular plane.  The rotation is
applied to the whole trajectory (all atoms) after the oxygen atom has been
shifted to the origin – i.e. a moving‑reference‑frame representation.

The function returns a new ``positions`` array with the same shape as the
input ``(n_steps, n_atoms, 3)`` but expressed in the rotated coordinate
system.

Typical usage
-------------
>>> from rotate_water import rotate_water_frame
>>> pos_rot = rotate_water_frame(
...     positions,               # (n_step, n_atom, 3)
...     ind_oxygen=12,
...     ind_hydrogen=[13, 14],
...     a1=[30., 0., 0.],
...     a2=[0., 30., 0.],
...     a3=[0., 0., 50.],
... )
"""


def _wrap_minimum_image(vec: np.ndarray, cell: np.ndarray) -> np.ndarray:
    """
    Apply the minimum‑image convention for an orthogonal cell.

    Parameters
    ----------
    vec : (..., 3) array
        Cartesian vectors to be wrapped.
    cell : (3, 3) array
        Orthogonal lattice vectors as rows (a1, a2, a3).

    Returns
    -------
    wrapped : (..., 3) array
        Vectors shifted by integer multiples of the cell so that each
        component lies in ``[-L_i/2, L_i/2)``.
    """
    L = np.diag(cell)  # cell lengths (Lx, Ly, Lz)
    return (vec + L / 2) % L - L / 2  # shift into centred box


def _rotation_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    """
    Rodrigues rotation matrix for a rotation of ``angle`` (rad) around ``axis``.

    Parameters
    ----------
    axis : (3,) array
        Unit vector defining the rotation axis.
    angle : float
        Rotation angle in radians.

    Returns
    -------
    R : (3, 3) array
        Orthogonal rotation matrix (R.T @ R = I).
    """
    axis = axis / np.linalg.norm(axis)
    ux, uy, uz = axis
    c = np.cos(angle)
    s = np.sin(angle)
    C = 1.0 - c
    # fmt: off
    R = np.array([[c + ux*ux*C,      ux*uy*C - uz*s, ux*uz*C + uy*s],
                  [uy*ux*C + uz*s,   c + uy*uy*C,    uy*uz*C - ux*s],
                  [uz*ux*C - uy*s,   uz*uy*C + ux*s, c + uz*uz*C]])
    # fmt: on
    return R


@as_function_node
def rotate_water_frame(
    md_output: OutputCalcMD
) -> np.ndarray:
    """
    Rotate every MD frame so that the bisector of the two O‑H bonds points
    along the positive *y*‑axis while the molecular plane is preserved.

    Parameters
    ----------
    positions : np.ndarray
        Trajectory array of shape ``(n_steps, n_atoms, 3)``.
    ind_oxygen : int
        Index of the oxygen atom (reference atom).
    ind_hydrogen : Sequence[int]
        Two indices of the hydrogen atoms belonging to the same water
        molecule.
    a1, a2, a3 : Sequence[float]
        Orthogonal lattice vectors of the simulation cell.  They are only
        needed for the periodic‑boundary‑condition wrapping.

    Returns
    -------
    rotated : np.ndarray
        New trajectory array with the same shape as ``positions`` but expressed
        in the rotated coordinate system.
    """
    # ------------------------------------------------------------------
    # Basic checks
    # ------------------------------------------------------------------
    positions = md_output.positions
    # position in species list
    # ind_H = list(md_output.species).index('H')  
    ind_O = list(md_output.species).index('O')
    # all indeces of a given element in a structure/snapshot
    # ind_hydrogen = np.argwhere(md_output.indices[0] == ind_H)
    ind_oxygen = np.argwhere(md_output.indices[0] == ind_O)[0]
    ind_hydrogen = [ind_oxygen + 1, ind_oxygen + 2]
    # ind_hydrogen_1 = ind_oxygen + 1
    # ind_hydrogen_2 = ind_oxygen + 2
    # ind_hydrogen = np.append(ind_hydrogen_1, ind_hydrogen_2)
    if positions.ndim != 3 or positions.shape[2] != 3:
        raise ValueError(
            "positions must have shape (n_steps, n_atoms, 3) with xyz as last axis."
        )
    n_steps, n_atoms, _ = positions.shape
    if len(ind_hydrogen) != 2:
        raise ValueError("Exactly two hydrogen indices must be supplied.")

    # ------------------------------------------------------------------
    # Build orthogonal cell matrix (used for wrapping)
    # ------------------------------------------------------------------
    # cell = np.array([a1, a2, a3], dtype=float)  # (3, 3)
    # if not np.allclose(cell, np.diag(np.diag(cell))):
    #     raise ValueError("The supplied cell vectors must be orthogonal (diagonal).")
    cell = md_output.cells[0]  # assume constant volume

    # ------------------------------------------------------------------
    # Pre‑allocate output array
    # ------------------------------------------------------------------
    rotated = np.empty_like(positions)

    # ------------------------------------------------------------------
    # Loop over time steps – the rotation matrix is *identical* for every
    # frame because it only depends on the instantaneous geometry of the
    # water molecule (which may fluctuate).  Therefore we recompute it at
    # each step.
    # ------------------------------------------------------------------
    for t in range(n_steps):
        # 1) wrap positions into the orthogonal cell
        pos_t = _wrap_minimum_image(positions[t], cell)  # (n_atoms, 3)

        # 2) shift so that the oxygen sits at the origin
        o_pos = pos_t[ind_oxygen]  # (3,)
        rel = pos_t - o_pos  # (n_atoms, 3)

        # 3) vectors O→H1 and O→H2 (still in the original frame)
        v1 = rel[ind_hydrogen[0]]  # (3,)
        v2 = rel[ind_hydrogen[1]]  # (3,)
        print("v1: ", v1)

        # 4) plane normal = v1 × v2  (the molecular plane)
        normal = np.cross(v1, v2)
        if np.linalg.norm(normal) < 1e-12:
            # Degenerate geometry – fall back to identity rotation
            R = np.eye(3)
        else:
            normal = normal / np.linalg.norm(normal)

            # 5) bisector (sum of the two OH vectors) – this already lies in the
            #    molecular plane, but we project it explicitly to avoid numerical
            #    drift out of the plane.
            bisector = v1 + v2
            print('bisector: ', bisector, normal)
            bisector_proj = bisector - np.dot(bisector, normal) * normal
            if np.linalg.norm(bisector_proj) < 1e-12:
                # If the two OH vectors are opposite (unlikely), keep identity.
                R = np.eye(3)
            else:
                bisector_proj = bisector_proj / np.linalg.norm(bisector_proj)

                # 6) Desired direction = +y
                target = np.array([0.0, 1.0, 0.0])

                # 7) Angle between projected bisector and target
                cos_theta = np.clip(np.dot(bisector_proj, target), -1.0, 1.0)
                theta = np.arccos(cos_theta)

                # 8) Rotation axis = normal × target (or normal if theta≈0)
                #    The axis must be perpendicular to the plane, i.e. parallel to
                #    the molecular normal.  Using the cross product ensures the
                #    rotation stays within the plane.
                axis = np.cross(bisector_proj, target)
                if np.linalg.norm(axis) < 1e-12:
                    # Already aligned – no rotation needed
                    R = np.eye(3)
                else:
                    axis = axis / np.linalg.norm(axis)
                    # Because the axis is already parallel to the molecular normal,
                    # the rotation will not tilt the plane.
                    R = _rotation_matrix(axis, theta)

        # 9) Apply rotation to *all* atoms (including O at origin)
        rotated[t] = rel @ R.T  # transpose because we want column vectors rotated

    return rotated
