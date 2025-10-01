from ase import Atoms
from pyiron_workflow import as_function_node
from pyiron_nodes.atomistic.engine.generic import OutputEngine


@as_function_node
def Static(
    structure: Atoms,
    engine=None,
):
    """Calculate static properties of a structure using ASE.

    This node evaluates the potential energy and forces of a given
    ``Atoms`` object with a specified calculator (engine). It is
    typically used for single‑point energy evaluations or as a building
    block in larger workflows where only the static configuration is of
    interest.

    Parameters
    ----------
    structure : ase.Atoms
        The atomic configuration to be evaluated.
    engine : OutputEngine, optional
        An ``OutputEngine`` wrapping an ASE calculator. If ``None`` a
        default EMT calculator is instantiated.

    Returns
    -------
    OutputCalcStaticList
        Dataclass containing ``energies_pot`` (array of potential
        energies) and ``forces`` (array of force vectors).
    """
    import numpy as np
    from pyiron_nodes.atomistic.calculator.data import OutputCalcStaticList

    if engine is None:
        from ase.calculators.emt import EMT
        from pyiron_nodes.atomistic.engine.generic import OutputEngine

        engine = OutputEngine(calculator=EMT())

    structure.calc = engine.calculator

    out = OutputCalcStaticList().dataclass()
    out.energies_pot = np.array(
        [float(structure.get_potential_energy())]
    )  # TODO: originally of type np.float32 -> why??
    out.forces = np.array([structure.get_forces()])

    return out


@as_function_node
def StaticEnergy(
    structure: Atoms,
    engine: OutputEngine,
):
    """Return the potential energy of a structure.

    A lightweight wrapper around ASE's ``get_potential_energy`` that
    expects an ``OutputEngine`` providing a calculator. Useful when only
    the scalar energy value is required, for example in optimization or
    screening workflows.

    Parameters
    ----------
    structure : ase.Atoms
        Atomic configuration.
    engine : OutputEngine
        Engine containing the ASE calculator to be used.

    Returns
    -------
    float
        The potential energy of the structure.
    """
    structure.calc = engine.calculator
    energy = structure.get_potential_energy()

    return energy


@as_function_node("out")
def Minimize(
    structure: Atoms = None, engine=None, fmax: float = 0.005, log_file: str = "tmp.log"
):
    """Relax a structure to a local minimum using BFGS optimization.

    The node performs geometry optimization with ASE's BFGS optimizer,
    recording energies, forces, and intermediate structures. It is
    commonly employed to obtain relaxed configurations before further
    property calculations.

    Parameters
    ----------
    structure : ase.Atoms, optional
        Initial atomic configuration. Must be provided.
    engine : OutputEngine, optional
        Engine wrapping an ASE calculator. If ``None`` a default EMT
        calculator is used.
    fmax : float, default 0.005
        Convergence criterion for the maximum force component.
    log_file : str, default "tmp.log"
        Path to the optimizer log file. Use ``"-"`` to write to stdout.

    Returns
    -------
    OutputCalcStaticList
        Dataclass containing energies, forces, structures, convergence
        flag, and number of optimization steps.
    """
    from ase.optimize import BFGS
    from ase.io.trajectory import Trajectory
    from pyiron_nodes.atomistic.calculator.data import OutputCalcStaticList

    if engine is None:
        from ase.calculators.emt import EMT
        from pyiron_nodes.atomistic.engine.generic import OutputEngine

        engine = OutputEngine(calculator=EMT())

    out = OutputCalcStaticList().dataclass()
    out.energies_pot = []
    out.forces = []
    out.structures = []

    out.structures.append(structure)
    initial_structure = structure.copy()
    initial_structure.calc = engine.calculator
    out.energies_pot.append(float(initial_structure.get_potential_energy()))
    out.forces.append(initial_structure.get_forces())

    if log_file is None:  # write to standard io
        log_file = "-"

    dyn = BFGS(initial_structure, logfile=log_file, trajectory="minimize.traj")
    out_dyn = dyn.run(fmax=fmax)

    traj = Trajectory("minimize.traj")
    atoms_relaxed = traj[-1]
    atoms_relaxed.calc = engine.calculator

    out.forces.append(atoms_relaxed.get_forces())
    out.energies_pot.append(float(atoms_relaxed.get_potential_energy()))
    atoms_relaxed.calc = None  # ase calculator is not pickable!!
    out.structures.append(atoms_relaxed)

    out.is_converged = dyn.converged()
    out.iter_steps = dyn.nsteps

    return out
