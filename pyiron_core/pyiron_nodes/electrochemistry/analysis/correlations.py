from typing import Sequence, Optional
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
    md_output: OutputCalcMD,
    eps: float = 0.1,
    xy_max: Optional[float] = 5,
    as_log: bool = True,
    n_bins: int = 100,
):
    """
    Rotate every MD frame so that the bisector of the two O‑H bonds points
    along the positive *y*‑axis while the molecular plane is preserved.
    Return the 2D histogram
    """
    # ------------------------------------------------------------------
    # Basic checks
    # ------------------------------------------------------------------
    positions = md_output.positions
    # position in species list
    # ind_H = list(md_output.species).index('H')
    ind_O = list(md_output.species).index("O")
    # all indeces of a given element in a structure/snapshot
    # ind_hydrogen = np.argwhere(md_output.indices[0] == ind_H)

    xy = []
    ind_oxygens = np.argwhere(md_output.indices[0] == ind_O)[0]
    for i, ind_oxygen in enumerate(ind_oxygens):
        ind_hydrogen = [ind_oxygen - 1, ind_oxygen - 2]
        print("ind: ", ind_oxygen, ind_hydrogen)
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

        mat = rotated.reshape(-1, 3)
        xy_ox = mat[np.abs(mat[:, 2]) < eps]
        if i == 0:
            xy = xy_ox
        else:
            # print("shape: ", np.shape(xy_ox), np.shape(xy))
            xy = np.concatenate((xy, xy_ox), axis=0)
        # np.append(xy, xy_ox)

    # ------------------------------------------------------------------
    # 3️⃣  Determine the histogram range
    # ------------------------------------------------------------------
    if xy_max is None:
        x_min, x_max = xy[:, 0].min(), xy[:, 0].max()
        y_min, y_max = xy[:, 1].min(), xy[:, 1].max()
    else:
        x_min, x_max = -xy_max, xy_max
        y_min, y_max = -xy_max, xy_max

    # Slightly enlarge the range so that points on the edge fall into a bin
    pad = 1e-12
    x_edges = np.linspace(x_min - pad, x_max + pad, n_bins + 1)
    y_edges = np.linspace(y_min - pad, y_max + pad, n_bins + 1)

    # from matplotlib.pylab import plt
    # plt.hist2d(xy[:, 0], xy[:, 1], bins=(nbins, nbins))
    #
    # ------------------------------------------------------------------
    # 4️⃣  Build the 2‑D histogram
    # ------------------------------------------------------------------
    hist, x_edges, y_edges = np.histogram2d(xy[:, 0], xy[:, 1], bins=[x_edges, y_edges])

    print("edges: ", x_edges)
    hist = hist.T
    if as_log:
        hist = np.log(hist + 1)
    return hist
