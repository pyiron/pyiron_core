import ase.units as units

from ase.atoms import Atoms
from pyiron_core.pyiron_workflow import as_function_node


@as_function_node("water")
def build_water(project: str = "test2", n_mols: int = 10) -> Atoms:
    """
    Construct a bulk water super‑cell with a target number of water molecules.

    The function creates a cubic simulation cell whose volume is chosen so that,
    at a user‑specified density, the cell contains exactly ``n_mols`` water molecules.
    A single water molecule (H₂O) is placed in the primitive cell, then the cell
    is repeated ``n × n × n`` times to reach the desired molecule count.

    Parameters
    ----------
    project : str, optional
        Name (or path) of a *pyiron* project.  If a string is supplied it is
        converted to a :class:`pyiron_atomistics.Project` instance; otherwise an
        already‑instantiated ``Project`` object can be passed.  Default is ``"test2"``.
    n_mols : int, optional
        Desired total number of water molecules in the final structure.
        The function will automatically compute the cubic root to obtain the
        repeat factor ``n``.  Default is ``10``.

    Returns
    -------
    Atoms
        A :class:`pyiron_atomistics.Atoms` object containing ``n_mols`` water
        molecules arranged in a cubic lattice with periodic boundary conditions.
    """

    import numpy as np
    from pyiron_atomistics import Project

    density = 1.0e-24  # g/A^3
    mol_mass_water = 18.015  # g/mol

    # Determining the supercell size size
    mass = mol_mass_water * n_mols / units.mol  # g
    vol_h2o = mass / density  # in A^3
    a = vol_h2o ** (1.0 / 3.0)  # A

    # Constructing the unitcell
    n = int(round(n_mols ** (1.0 / 3.0)))

    dx = 0.7
    r_O = [0, 0, 0]
    r_H1 = [dx, dx, 0]
    r_H2 = [-dx, dx, 0]
    unit_cell = (a / n) * np.eye(3)

    if isinstance(project, str):
        project = Project(project)

    water = project.create_atoms(
        elements=["H", "H", "O"], positions=[r_H1, r_H2, r_O], cell=unit_cell, pbc=True
    )
    water.set_repeat([n, n, n])
    return water


@as_function_node
def add_water_film(
    electrode: Atoms,
    water_width: float = 10.0,
    hydrophobic_gap: float = 3.0,
    density: float = 1.0e-24,
) -> Atoms:
    """
    Append a thin slab of liquid water on top of an existing electrode structure.

    The function builds a water film of user‑defined thickness (`water_width`) by
    placing a single water molecule in a cubic cell sized to match the target
    density, then repeating that cell to fill the lateral dimensions of the
    electrode.  The film is shifted upward by ``hydrophobic_gap`` above the
    highest atom of the electrode, mimicking a non‑wetting interfacial layer.

    Parameters
    ----------
    electrode : Atoms
        The base structure (typically a metallic or semiconductor slab) to which
        the water film will be attached.  Must have a defined periodic cell.
    water_width : float, optional
        Desired thickness of the water slab in Ångström.  Default is ``10.0 Å``.
    hydrophobic_gap : float, optional
        Vertical separation (in Å) between the topmost electrode atom and the
        first water molecule.  This mimics a hydrophobic “gap” often observed
        experimentally.  Default is ``3.0 Å``.
    density : float, optional
        Target mass density of water in g Å⁻³.  The default corresponds to the
        experimental value of ≈ 1 g cm⁻³.  Changing this value will affect the
        number of water molecules placed in the film.

    Returns
    -------
    Atoms
        A new atomic structure that contains the original
        electrode atoms plus the added water film.  The cell of the returned
        object is identical to the electrode’s cell (periodic in all three
        directions).
    """

    import ase.units as units
    import numpy as np
    from ase.build import molecule
    from pyiron_atomistics import ase_to_pyiron

    lx, ly = electrode.cell.diagonal()[:2]
    zmin = np.max(electrode.positions[:, 2])

    # Water
    n_mols = 1
    mol_mass_water = 18.015  # g/mol

    # Determining the supercell size size
    mass = mol_mass_water * n_mols / units.mol  # g
    vol_h2o = mass / density  # in A^3
    a = vol_h2o ** (1.0 / 3.0)  # A
    cell = np.array([lx, ly, water_width])
    cell_repeat = (cell // a).astype(int)

    H2O = molecule("H2O", cell=cell / cell_repeat)
    H2O.set_pbc(True)
    H2O = ase_to_pyiron(H2O).repeat(cell_repeat)
    H2O.positions[:, 2] += zmin + hydrophobic_gap
    H2O.set_cell(electrode.cell)

    electrochemical_cell = electrode + H2O

    return electrochemical_cell


@as_function_node
def add_neon_layer(structure, d_eq: float = 3, hydrophobic_gap: float = 3.0):
    """
    Add a single layer of neon atoms to the input structure above the maximum z value of an atom.

    Parameters:
    structure (ase.Atoms): The input atomic structure.
    d_eq (float): The optimum Ne-Ne distance.
    hydrophobic_gap (float): The gap between the input structure and the neon layer. Default is 1.0.

    Returns:
    ase.Atoms: The modified structure with a single layer of neon atoms.
    """
    import numpy as np

    from pyiron_core.pyiron_nodes.atomistic.structure.build import Surface

    # Get the maximum z value of an atom in the structure
    max_z = np.max(structure.get_positions()[:, 2])

    # Get the cell dimensions in the x and y directions
    a = structure.get_cell()[0][0]
    b = structure.get_cell()[1][1]

    # Calculate the number of Ne atoms in the x and y directions
    nx = int(np.ceil(a / d_eq))
    ny = int(np.ceil(b / (d_eq * 2))) * 2

    # Create a new Atoms object for the neon layer
    neon_layer = Surface(
        "Ne", "fcc111", size=f"{nx} {ny} 1", vacuum=25.0, orthogonal=True
    ).run()
    neon_layer.set_cell(cell=structure.cell, scale_atoms=True)
    neon_layer.positions[:, 2] = max_z + hydrophobic_gap

    # Create a new Atoms object for the modified structure
    modified_structure = structure.copy()

    # Add the neon layer to the modified structure
    modified_structure.extend(neon_layer)

    return modified_structure


@as_function_node
def add_ion_pair(
    structure: Atoms,
    anion: str = "Na",
    cation: str = "Cl",
    number: int = 0,
    seed: int = 1234,
) -> Atoms:
    """
    Insert a specified number of ion pairs (anion/cation) into a structure.

    The function selects ``number`` oxygen atoms at random, replaces the first
    half with the provided ``anion`` species and the second half with the
    ``cation`` species.  After the substitution, two atoms immediately above
    each selected oxygen are removed – this mimics the removal of water
    molecules that would otherwise coordinate the ion.

    Parameters
    ----------
    structure : Any
        An atomic structure (e.g., ``pyiron_atomistics.Atoms``) that contains
        oxygen atoms.  The structure is copied internally; the original is not
        modified.
    anion : str
        Chemical symbol of the anion to place on the first half of the selected
        oxygen sites.
    cation : str
        Chemical symbol of the cation to place on the second half of the
        selected oxygen sites.
    number : int
        Total number of oxygen atoms to be replaced.  Must be an even number;
        otherwise the integer division ``number // 2`` determines the split.
    seed : int, optional
        Random seed for reproducible selection of oxygen atoms.  Default is
        ``1234``.

    Returns
    -------
    Any
        The modified structure with ion pairs inserted and neighbouring atoms
        removed.
    """
    import numpy as np
    from pyiron_atomistics import (
        Project,
    )  # retained for compatibility with existing code
    import random  # retained for compatibility; not used directly

    # Work on a copy to avoid side‑effects on the input structure
    electrolyte = structure.copy()

    # Indices of all oxygen atoms in the structure
    ind_O = electrolyte.select_index("O")

    # Randomly choose ``number`` distinct oxygen indices
    rng = np.random.default_rng(seed)
    picked = np.sort(rng.choice(ind_O, int(number), replace=False)).tolist()

    # Replace the first half with the anion and the second half with the cation
    half = number // 2
    for i in picked[:half]:
        electrolyte[i] = anion
    for i in picked[half:number]:
        electrolyte[i] = cation

    # Remove two atoms above each selected site (e.g., coordinated water)
    for i in reversed(picked):
        del electrolyte[i + 2]
        del electrolyte[i + 1]

    return electrolyte
