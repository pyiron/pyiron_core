from ase.atoms import Atoms

from pyiron_core import as_function_node
from pyiron_core.pyiron_nodes.atomistic.calculator.data import OutputCalcMD


@as_function_node
def structure_to_calc_md(structure: Atoms) -> OutputCalcMD:
    import numpy as np

    calc_md = OutputCalcMD().dataclass()
    calc_md.positions = np.array([structure.positions])
    calc_md.cells = np.array([structure.cell * np.eye(3)])
    calc_md.indices = np.array([structure.indices])
    calc_md.species = [e.Abbreviation for e in structure.species]

    return calc_md


@as_function_node
def get_structure(trajectory: OutputCalcMD, index: int = -1) -> Atoms:
    import numpy as np
    from pyiron.atomistics.structure.atoms import ase_to_pyiron

    species = np.array(trajectory.species)[trajectory.indices[index]]

    structure = Atoms(
        species,
        positions=trajectory.positions[index],
        cell=trajectory.cells[index],
        pbc=[True, True, True],
    )
    structure = ase_to_pyiron(structure)
    return structure
