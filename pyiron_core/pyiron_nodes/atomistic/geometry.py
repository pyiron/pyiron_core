"""Utility functions for simple geometric analysis of atomic structures.

The functions in this module are intentionally lightweight and have no external
dependencies beyond ``numpy`` and ``ase``.  They are useful in workflows where
one needs to extract bond lengths and bond angles for water molecules from an
ASE ``Atoms`` object.

Only O–H bonds and the H‑O‑H angle are calculated – this matches the canonical
definition of a water molecule.  The algorithm is tolerant of structures that
contain multiple water molecules, as long as the O atoms are correctly identified
by their chemical symbol ``"O"`` and the H atoms by ``"H"``.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from ase import Atoms

from pyiron_core.pyiron_workflow import as_function_node


@as_function_node("structure")
def compute_water_bonds_angles(
    structure: Atoms,
    cutoff: float = 1.2,
) -> Dict[str, List]:
    """Compute O–H bond lengths and H‑O‑H angles for water molecules.

    Parameters
    ----------
    structure:
        ASE ``Atoms`` object containing one or more water molecules.
    cutoff:
        Distance (Å) used to decide whether an O–H pair belongs to the same
        molecule.  A typical O–H bond length in water is ~0.96 Å, so a cutoff of
        1.2 Å safely captures bonded pairs while excluding non‑bonded H atoms.

    Returns
    -------
    dict
        ``{"bonds": [(O_idx, H_idx, distance), ...],
        "angles": [(O_idx, H1_idx, H2_idx, angle_deg), ...]}``
        where ``O_idx``/``H_idx`` are the atom indices in the original ASE
        object, ``distance`` is the O–H bond length in Å and ``angle_deg`` the
        H‑O‑H angle in degrees.

    Notes
    -----
    The function does **not** attempt to verify chemical connectivity beyond the
    distance criterion.  If an oxygen atom has fewer or more than two hydrogen
    neighbours within the cutoff, only the available bonds are reported and the
    angle is omitted for that oxygen.
    """

    positions = structure.get_positions()
    symbols = np.array(structure.get_chemical_symbols())

    # Identify oxygen and hydrogen indices
    o_indices = np.where(symbols == "O")[0]
    h_indices = np.where(symbols == "H")[0]

    bonds: List[Tuple[int, int, float]] = []
    angles: List[Tuple[int, int, int, float]] = []

    # Pre‑compute distance matrix for efficiency
    diff = positions[:, None, :] - positions[None, :, :]
    dist_matrix = np.linalg.norm(diff, axis=2)

    for o_idx in o_indices:
        # Find hydrogen atoms within the cutoff distance
        h_neighbors = [
            h_idx for h_idx in h_indices if dist_matrix[o_idx, h_idx] < cutoff
        ]

        # Record bond lengths for every O‑H pair found
        for h_idx in h_neighbors:
            # Append a tuple (O_index, H_index, distance)
            bonds.append((int(o_idx), int(h_idx), float(dist_matrix[o_idx, h_idx])))

        # Compute the H‑O‑H angle only if exactly two hydrogens are bonded
        if len(h_neighbors) == 2:
            vec1 = positions[h_neighbors[0]] - positions[o_idx]
            vec2 = positions[h_neighbors[1]] - positions[o_idx]
            # Calculate the angle via the dot product
            cos_angle = np.dot(vec1, vec2) / (
                np.linalg.norm(vec1) * np.linalg.norm(vec2)
            )
            # Numerical safety – clip to the valid domain of arccos
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle_rad = np.arccos(cos_angle)
            angle_deg = np.degrees(angle_rad)
            # Append a tuple (O_index, H1_index, H2_index, angle_deg)
            angles.append(
                (int(o_idx), int(h_neighbors[0]), int(h_neighbors[1]), float(angle_deg))
            )

    return {"bonds": bonds, "angles": angles}
