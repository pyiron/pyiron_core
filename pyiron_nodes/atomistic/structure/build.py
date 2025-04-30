from __future__ import annotations

from pyiron_workflow import as_function_node, as_macro_node
from typing import Optional
from ase.atoms import Atoms

# from pyiron_workflow.workflow import Workflow


@as_function_node("structure")
def Bulk(
    name: str,
    crystalstructure: Optional[str] = None,
    a: Optional[float | int] = None,
    c: Optional[float | int] = None,
    c_over_a: Optional[float] | int = None,
    u: Optional[float | int] = None,
    orthorhombic: bool = False,
    cubic: bool = False,
):
    from pyiron_atomistics import _StructureFactory

    return _StructureFactory().bulk(
        name,
        crystalstructure,
        a,
        c,
        c_over_a,
        u,
        orthorhombic,
        cubic,
    )


@as_function_node("struct")
def HighIndexSurface(
    element: str,
    crystal_structure: str,
    lattice_constant: float,
    terrace_orientation: Optional[list] = None,
    step_orientation: Optional[list] = None,
    kink_orientation: Optional[list] = None,
    step_down_vector: Optional[list] = None,
    length_step: int = 0,
    length_terrace: int = 0,
    length_kink: int = 0,
    layers: int = 6,
    vacuum: int = 10,
) -> Atoms:
    """
    Gives a slab positioned at the bottom with the high index surface computed by high_index_surface_info().
    Args:
        element (str): The parent element eq. "N", "O", "Mg" etc.
        crystal_structure (str): The crystal structure of the lattice
        lattice_constant (float): The lattice constant
        terrace_orientation (list): The miller index of the terrace. default: [1,1,1]
        step_orientation (list): The miller index of the step. default: [1,1,0]
        kink_orientation (list): The miller index of the kink. default: [1,1,1]
        step_down_vector (list): The direction for stepping down from the step to next terrace. default: [1,1,0]
        length_terrace (int): The length of the terrace along the kink direction in atoms. default: 3
        length_step (int): The length of the step along the step direction in atoms. default: 3
        length_kink (int): The length of the kink along the kink direction in atoms. default: 1
        layers (int): Number of layers of the high_index_surface. default: 60
        vacuum (float): Thickness of vacuum on the top of the slab. default:10

    Returns:
        slab: ase.atoms.Atoms instance Required surface
    """
    from structuretoolkit.build.surface import get_high_index_surface_info
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    from pyiron import ase_to_pyiron
    from ase.build import bulk, surface
    from structuretoolkit.common.pymatgen import ase_to_pymatgen, pymatgen_to_ase
    import numpy as np

    basis = bulk(
        name=element, crystalstructure=crystal_structure, a=lattice_constant, cubic=True
    )
    high_index_surface, _, _ = get_high_index_surface_info(
        element=element,
        crystal_structure=crystal_structure,
        lattice_constant=lattice_constant,
        terrace_orientation=terrace_orientation,
        step_orientation=step_orientation,
        kink_orientation=kink_orientation,
        step_down_vector=step_down_vector,
        length_step=length_step,
        length_terrace=length_terrace,
        length_kink=length_kink,
    )
    surf = surface(basis, high_index_surface, layers, vacuum)
    slab = pymatgen_to_ase(
        SpacegroupAnalyzer(ase_to_pymatgen(structure=surf)).get_refined_structure()
    )
    slab.positions[:, 2] = slab.positions[:, 2] - np.min(slab.positions[:, 2])
    slab.set_pbc(True)
    return ase_to_pyiron(slab)


@as_macro_node("structure")
def CubicBulkCell(
    element: str, cell_size: int = 1, vacancy_index: Optional[int] = None
):
    from pyiron_nodes.atomistic.structure.transform import (
        CreateVacancy,
        Repeat,
    )
    from pyiron_workflow import Workflow

    wf = Workflow("macro")

    wf.bulk = Bulk(name=element, cubic=True)
    wf.repeat = Repeat(structure=wf.bulk, repeat_scalar=cell_size)

    wf.vacancy = CreateVacancy(structure=wf.repeat, index=vacancy_index)
    return wf.vacancy
