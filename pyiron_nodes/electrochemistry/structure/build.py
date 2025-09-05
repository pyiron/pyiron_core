from pyiron_workflow import Workflow, as_function_node
import ase.units as units


@as_function_node("water")
def build_water(project, n_mols: int = 10):

    import numpy as np

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
    water = project.create_atoms(
        elements=["H", "H", "O"], positions=[r_H1, r_H2, r_O], cell=unit_cell, pbc=True
    )
    water.set_repeat([n, n, n])
    return water


@as_function_node
def add_water_film(electrode, water_width: float = 10.0, hydrophobic_gap: float = 3.0, density: float = 1.0e-24):

    from ase.build import molecule
    from pyiron_atomistics import ase_to_pyiron
    import ase.units as units
    import numpy as np
    
    lx, ly = electrode.cell.diagonal()[:2]
    zmin = np.max(electrode.positions[:,2])
    zmax = zmin + water_width
    
    # Water 
    n_mols = 1
    # density = 1.0e-24  # g/A^3
    mol_mass_water = 18.015  # g/mol
    
    # Determining the supercell size size
    mass = mol_mass_water * n_mols / units.mol  # g
    vol_h2o = mass / density  # in A^3
    a = vol_h2o ** (1.0 / 3.0)  # A
    cell = np.array([lx, ly, water_width])
    cell_repeat = (cell // a).astype(int)
    
    H2O = molecule('H2O',cell=cell/cell_repeat)
    H2O.set_pbc(True)
    H2O = ase_to_pyiron(H2O).repeat(cell_repeat)
    H2O.positions[:,2] += zmin + hydrophobic_gap
    H2O.set_cell(electrode.cell)

    electrochemical_cell = electrode + H2O

    return electrochemical_cell


@as_function_node
def add_neon_layer(structure, d_eq: float, hydrophobic_gap: float = 1.0):
    """
    Add a single layer of neon atoms to the input structure above the maximum z value of an atom.

    Parameters:
    structure (ase.Atoms): The input atomic structure.
    d_eq (float): The optimum Ne-Ne distance.
    hydrophobic_gap (float): The gap between the input structure and the neon layer. Default is 1.0.

    Returns:
    ase.Atoms: The modified structure with a single layer of neon atoms.
    """
    from pyiron_nodes.atomistic.structure.build import Bulk, Surface
    import numpy as np
    
    # Get the maximum z value of an atom in the structure
    max_z = np.max(structure.get_positions()[:, 2])

    # Get the cell dimensions in the x and y directions
    a = structure.get_cell()[0][0]
    b = structure.get_cell()[1][1]

    # Calculate the number of Ne atoms in the x and y directions
    nx = int(np.ceil(a / d_eq))
    ny = int(np.ceil(b / (d_eq * 2))) * 2

    # Create a new Atoms object for the neon layer
    neon_layer = Surface('Ne', 'fcc111', size = f'{nx} {ny} 1', vacuum = 25.0, orthogonal=True).run()
    neon_layer.set_cell(cell = structure.cell, scale_atoms = True)
    neon_layer.positions[:,2] = max_z + hydrophobic_gap
    
    # Create a new Atoms object for the modified structure
    modified_structure = structure.copy()

    # Add the neon layer to the modified structure
    modified_structure.extend(neon_layer)

    return modified_structure