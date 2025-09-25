from __future__ import annotations

from typing import Literal, Optional

from ase.atoms import Atoms

from pyiron_core.pyiron_workflow import as_function_node, as_macro_node

# from pyiron_core.pyiron_workflow.workflow import Workflow


@as_function_node("structure")
def Bulk(
    name: str,
    crystalstructure: Optional[
        Literal["fcc", "bcc", "hcp", "diamond", "rocksalt"]
    ] = None,
    a: Optional[float] = None,
    c: Optional[float] = None,
    c_over_a: Optional[float] = None,
    u: Optional[float] = None,
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
    import numpy as np
    from ase.build import bulk, surface
    from pyiron import ase_to_pyiron
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    from structuretoolkit.build.surface import get_high_index_surface_info
    from structuretoolkit.common.pymatgen import ase_to_pymatgen, pymatgen_to_ase

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
    from pyiron_core.pyiron_nodes.atomistic.structure.transform import (
        CreateVacancy,
        Repeat,
    )
    from pyiron_core.pyiron_workflow import Workflow

    if (
        vacancy_index is not None
        and type(vacancy_index) is not int
        and "va_i_" not in vacancy_index
    ):
        print("Vacancy Index: ", vacancy_index, type(vacancy_index))
        vacancy_index = int(vacancy_index)
    wf = Workflow("macro")

    wf.bulk = Bulk(name=element, cubic=True)
    wf.repeat = Repeat(structure=wf.bulk, repeat_scalar=cell_size)

    wf.vacancy = CreateVacancy(structure=wf.repeat, index=vacancy_index)
    return wf.vacancy


@as_function_node("structure")
def Surface(
    element: str,
    surface_type: str,
    size: str = "1 1 1",
    vacuum: float = 1.0,
    center: bool = False,
    pbc: bool = True,
    orthogonal: bool = False,
):
    """
    Generate a surface based on the ase.build.surface module.

    Args:
        element (str): Element name
        surface_type (str): The string specifying the surface type generators available through ase (fcc111,
        hcp0001 etc.)
        size (tuple): Size of the surface
        vacuum (float): Length of vacuum layer added to the surface along the z direction
        center (bool): Tells if the surface layers have to be at the center or at one end along the z-direction
        orthogonal (bool): Orthogonal option
        pbc (list/numpy.ndarray): List of booleans specifying the periodic boundary conditions along all three
                                  directions.
        **kwargs: Additional, arguments you would normally pass to the structure generator like 'a', 'b', etc.

    Returns:
        pyiron_atomistics.atomistics.structure.atoms.Atoms instance: Required surface

    """
    import types

    import numpy as np
    from ase.build import (
        add_adsorbate,
        add_vacuum,
        bcc100,
        bcc110,
        bcc111,
        bcc111_root,
        diamond100,
        diamond111,
        fcc100,
        fcc110,
        fcc111,
        fcc111_root,
        fcc211,
        hcp0001,
        hcp0001_root,
        hcp10m10,
        mx2,
        root_surface,
        root_surface_analysis,
    )
    from ase.build import (
        surface as ase_surf,
    )
    from pyiron_atomistics.atomistics.structure.atoms import (
        ase_to_pyiron,
    )

    # https://gitlab.com/ase/ase/blob/master/ase/lattice/surface.py
    if pbc is None:
        pbc = True
    for surface_class in [
        add_adsorbate,
        add_vacuum,
        bcc100,
        bcc110,
        bcc111,
        diamond100,
        diamond111,
        fcc100,
        fcc110,
        fcc111,
        fcc211,
        hcp0001,
        hcp10m10,
        mx2,
        hcp0001_root,
        fcc111_root,
        bcc111_root,
        root_surface,
        root_surface_analysis,
        ase_surf,
    ]:
        if surface_type == surface_class.__name__:
            surface_type = surface_class
            break

    size = [int(x) for x in size.split(" ")]
    if isinstance(surface_type, types.FunctionType):
        if center:
            surface = surface_type(
                symbol=element, size=size, vacuum=vacuum, orthogonal=orthogonal
            )
        else:
            surface = surface_type(symbol=element, size=size, orthogonal=orthogonal)
            z_max = np.max(surface.positions[:, 2])
            surface.cell[2, 2] = z_max + vacuum
        surface.pbc = pbc
    return ase_to_pyiron(surface)
    # else:
    #     raise ValueError(f"Surface type {surface_type} not recognized.")
