from __future__ import annotations

from typing import Literal, Optional

from ase.atoms import Atoms

from pyiron_core.pyiron_workflow import as_function_node, as_macro_node


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
) -> Atoms:
    """
    Create a bulk crystal structure.

    **Scientific purpose**
    Generate a pristine bulk unit cell for a given element and crystal lattice
    (e.g. fcc Al, bcc Fe, hcp Mg). This is the typical starting point for defect
    studies, molecular‑dynamics simulations, or high‑throughput materials screening.

    **Required inputs**
    - ``name``: Chemical symbol of the element (e.g. ``"Al"``).
    - ``crystalstructure``: One of ``"fcc"``, ``"bcc"``, ``"hcp"``, ``"diamond"``,
      ``"rocksalt"``; if omitted the default of the underlying factory is used.
    - ``a``, ``c``, ``c_over_a``, ``u``: Optional lattice parameters or internal
      coordinates.
    - ``orthorhombic`` / ``cubic``: Force the cell shape to be orthorhombic or cubic.

    **Typical use‑cases**
    * Building a bulk material before inserting vacancies, interstitials, or
      surfaces.
    * Preparing input structures for relaxation or MD runs.
    * Generating a library of bulk cells for high‑throughput workflows.

    Returns
    -------
    Node returns an Atoms object from ``pyiron_atomistics._StructureFactory().bulk`` compatible
    with ASE/pyiron.
    """
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
    Generate a high‑index surface slab (terrace/step/kink) for surface‑science studies.

    **Scientific purpose**
    Construct a slab that contains a specific high‑index facet, optionally with
    steps and kinks, which are crucial for catalytic activity, adsorption, and
    defect formation analyses.

    **Required inputs**
    - ``element``: Chemical symbol (e.g. ``"Ni"``).
    - ``crystal_structure``: Lattice type (e.g. ``"fcc"``, ``"hcp"``).
    - ``lattice_constant``: Bulk lattice constant in Å.
    - ``terrace_orientation``, ``step_orientation``, ``kink_orientation``: Miller
      indices defining the terrace, step, and kink planes (default ``[1,1,1]``,
      ``[1,1,0]``, ``[1,1,1]`` respectively).
    - ``step_down_vector``: Direction vector for stepping down from the step to the
      next terrace.
    - ``length_step``, ``length_terrace``, ``length_kink``: Number of atoms along
      each direction (defaults give a minimal slab).
    - ``layers``: Number of atomic layers in the slab.
    - ``vacuum``: Vacuum thickness (Å) added on top of the slab.

    **Typical use‑cases**
    * Preparing catalyst surface models with realistic step/kink features.
    * Studying adsorption energetics on non‑low‑index facets.
    * Generating input structures for DFT or classical MD surface calculations.

    Returns
    -------
    ``ase.atoms.Atoms`` instance representing the high‑index slab (converted to a
    pyiron structure before returning).
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
) -> Atoms:
    """
    Build a cubic bulk supercell and optionally introduce a single vacancy.

    **Scientific purpose**
    Produce a cubic replication of a bulk unit cell (useful for convergence tests,
    defect calculations, or large‑scale MD) and optionally create a vacancy at a
    user‑specified lattice site.

    **Required inputs**
    - ``element``: Chemical symbol of the bulk material (e.g. ``"Si"``).
    - ``cell_size``: Integer scaling factor for the cubic repeat (default ``1``).
    - ``vacancy_index``: Index of the atom to be removed (``None`` means no vacancy).

    **Typical use‑cases**
    * Generating a supercell for finite‑size scaling of defect formation energies.
    * Preparing a large periodic cell for classical MD simulations.
    * Creating a simple vacancy model for DFT relaxation studies.

    Returns
    -------
    The workflow node that contains the final structure (with or without the vacancy).
    """
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
) -> Atoms:
    """
    Generate a low‑index surface slab using ASE's built‑in surface generators.

    **Scientific purpose**
    Quickly create common surface terminations (e.g. fcc111, bcc110) for adsorption,
    catalysis, or surface‑energy calculations.

    **Required inputs**
    - ``element``: Chemical symbol (e.g. ``"Cu"``).
    - ``surface_type``: Name of the ASE surface generator (e.g. ``"fcc111"``,
      ``"bcc110"``, ``"hcp0001"``, etc.).
    - ``size``: Replication of the primitive surface cell expressed as three
      integers separated by spaces (default ``"1 1 1"``).
    - ``vacuum``: Vacuum thickness (Å) added on the top of the slab.
    - ``center``: If ``True``, place the slab in the middle of the cell; otherwise
      the slab sits at the bottom.
    - ``pbc``: Periodic boundary conditions flag (default ``True``).
    - ``orthogonal``: Force orthogonal cell vectors when ``True``.

    **Typical use‑cases**
    * Building a clean surface for DFT adsorption studies.
    * Generating a slab for classical MD surface simulations.
    * Creating a reference surface for surface‑energy or work‑function calculations.

    Returns
    -------
    ``pyiron_atomistics.atomistics.structure.atoms.Atoms`` instance representing the surface.
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
