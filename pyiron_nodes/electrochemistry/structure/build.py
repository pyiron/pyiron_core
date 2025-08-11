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