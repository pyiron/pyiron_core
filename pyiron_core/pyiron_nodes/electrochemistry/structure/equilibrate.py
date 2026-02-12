from typing import Literal

import pandas as pd
from matplotlib.figure import Figure

from pyiron_core.pyiron_nodes.atomistic.calculator.data import InputCalcMD, OutputCalcMD
from pyiron_core.pyiron_workflow import as_function_node

ANION = "F"


@as_function_node
def Equilibrate(
    solvated_electrode,
    water_potential,
    parameters=None,
    server=None,
    store: bool = True,
) -> OutputCalcMD:
    """
    Run a LAMMPS molecular‑dynamics equilibration of a water‑electrolyte systems.

    The routine creates a temporary *pyiron* project called ``equilibrate``,
    builds a LAMMPS job named ``water_equilibration`` and runs it with the
    supplied structure and water potential.  Selective dynamics are applied so
    that only the water atoms (O and H) are allowed to move while the electrode
    atoms remain fixed.  After the job finishes, the most relevant MD output
    (positions, velocities, energies, forces, …) is collected into an
    :class:`OutputCalcMD` dataclass that can be passed downstream to analysis
    or further simulation nodes.

    Parameters
    ----------
    solvated_electrode : pyiron_atomistics.Atoms
        The electrode structure that already contains a water slab.  The object
        must support the ``add_tag`` and ``selective_dynamics`` attributes used
        by pyiron to freeze the electrode atoms.
    water_potential : pandas.DataFrame
        A DataFrame describing the LAMMPS potential for water (and optionally
        metal / neon species).  It is passed unchanged to the LAMMPS job via
        ``j.potential``.
    parameters : InputCalcMD, optional
        A dataclass (or any object that can be converted to a ``dict``) that
        contains the MD control parameters (time step, temperature, ensemble,
        number of steps, etc.).  If ``None`` the function creates a default
        ``InputCalcMD`` instance by calling ``InputCalcMD().run()``.
    store : bool, optional
        If ``True`` the generated ``OutputCalcMD`` instance is stored in the
        pyiron job’s output folder (default).  The current implementation does
        not make use of this flag, but it is kept for API compatibility with
        other nodes.

    Returns
    -------
    OutputCalcMD
        A dataclass containing the full MD trajectory and thermodynamic data
        produced by LAMMPS.
    """
    parameters = InputCalcMD().run() if parameters is None else parameters

    from dataclasses import asdict

    from pyiron_atomistics import Project

    # Create a job for LAMMPS equilibration
    pr = Project("equilibrate")
    j = pr.create.job.Lammps("water_equilibration", delete_existing_job=True)

    solvated_electrode.add_tag(selective_dynamics=[False, False, False])
    solvated_electrode.selective_dynamics[
        solvated_electrode.select_index(["O", "H", "Na", ANION])
    ] = [
        True,
        True,
        True,
    ]

    j.structure = solvated_electrode
    j.potential = water_potential
    j.calc_md(**asdict(parameters))

    if server is not None:
        print(f"Server = {server.queue}, {server}, {server.cores}")
        j.server.queue = server.queue
        j.server.cores = server.cores

    j.run(delete_existing_job=True)

    if server is not None:
        pr.wait_for_job(j, interval_in_s=5, max_iterations=1000)

    job_out = j["output/generic"]

    from pyiron_core.pyiron_nodes.atomistic.calculator.data import OutputCalcMD

    out = OutputCalcMD().dataclass()
    out.cells = job_out["cells"]
    out.energies_pot = job_out["energy_pot"]
    out.energies_tot = job_out["energy_tot"]
    out.forces = job_out["forces"]
    out.indices = job_out["indices"]
    out.natoms = job_out["natoms"]
    out.positions = job_out["positions"]
    out.pressures = job_out["pressures"]
    out.steps = job_out["steps"]
    out.temperatures = job_out["temperature"]
    out.unwrapped_positions = job_out["unwrapped_positions"]
    out.velocities = job_out["velocities"]
    out.volumes = job_out["volume"]
    out.species = [e.Abbreviation for e in solvated_electrode.species]

    return out


@as_function_node
def WaterPotential(
    metal: str = "Al",
    metal_charge: float = 0.0,
    neon_charge: float = 0.0,
    epsilon: float = 0.102,
    sigma: float = 3.188,
):
    import pandas

    water_potential = pandas.DataFrame(
        {
            "Name": ["H2O_tip3p"],
            "Filename": [[]],
            "Model": ["TIP3P"],
            "Species": [["H", "O", metal, "Ne"]],
            "Config": [
                [
                    "# @potential_species H_O  ### species in potential\n",
                    "# W.L. Jorgensen",
                    "The Journal of Chemical Physics 79",
                    "926 (1983); https://doi.org/10.1063/1.445869 \n",
                    "#\n",
                    "\n",
                    "units      real\n",
                    "dimension  3\n",
                    "atom_style full\n",
                    "\n",
                    "# create groups ###\n",
                    "group O type 2\n",
                    "group H type 1\n",
                    f"group {metal} type 3\n",
                    "group Ne type 4\n",
                    "\n",
                    "## set charges - beside manually ###\n",
                    "set group O charge -0.830\n",
                    "set group H charge 0.415\n",
                    f"set group {metal} charge {metal_charge}\n",
                    f"set group Ne charge {neon_charge}\n",
                    "\n",
                    "### TIP3P Potential Parameters ###\n",
                    "pair_style lj/cut/coul/long 10.0\n",
                    "pair_coeff * * 0.000 0.000 \n",
                    "pair_coeff 2 2 0.102 3.188 \n",
                    "pair_coeff 2 3 {:.4} {:.4} \n".format(epsilon, sigma),
                    "pair_coeff 2 4 {:.4} {:.4} \n".format(epsilon, sigma),
                    "bond_style  harmonic\n",
                    "bond_coeff  1 450 0.9572\n",
                    "angle_style harmonic\n",
                    "angle_coeff 1 55 104.52\n",
                    "kspace_style pppm 1.0e-5   # final npt relaxation\n",
                    "\n",
                ]
            ],
        }
    )

    return water_potential


@as_function_node
def IonPotential(
    metal: str = "Al",
    metal_charge: float = 0.0,
    cation: str = "Na",
    cation_charge: float = 1.0,
    epsilon_cation: float = 0.102,
    sigma_cation: float = 1.188,
    anion: str = "F",
    anion_charge: float = -1.0,
    epsilon_anion: float = 0.102,
    sigma_anion: float = 1.188,
    sigma_ion_pair: float = 3.188,
    epsilon_ion_pair: float = 0.102,
    neon_charge: float = 0.0,
    epsilon: float = 0.102,
    sigma: float = 3.188,
) -> pd.DataFrame:
    """
    Generate a LAMMPS input configuration for a TIP3P water model containing
    a metal slab, a neon layer, and optional ion pair (cation/anion).

    The function builds a :class:`pandas.DataFrame` with a single row that
    stores the name of the potential, the associated filename (empty list by
    default), the model type, the list of species involved, and a list of
    configuration strings that can be written directly to a LAMMPS input file.

    Parameters
    ----------
    metal : str, default ``"Al"``
        Symbol of the metal element forming the slab.
    metal_charge : float, default ``0.0``
        Partial charge assigned to the metal atoms.
    cation : str, default ``"Na"``
        Symbol of the cation species.
    cation_charge : float, default ``1.0``
        Charge of the cation.
    epsilon_cation : float, default ``0.102``
        Lennard‑Jones epsilon parameter for the cation (eV).
    sigma_cation : float, default ``1.188``
        Lennard‑Jones sigma parameter for the cation (Å).
    anion : str, default ``"F"``
        Symbol of the anion species.
    anion_charge : float, default ``-1.0``
        Charge of the anion.
    epsilon_anion : float, default ``0.102``
        Lennard‑Jones epsilon parameter for the anion (eV).
    sigma_anion : float, default ``1.188``
        Lennard‑Jones sigma parameter for the anion (Å).
    sigma_ion_pair : float, default ``3.188``
        Lennard‑Jones sigma for the cation‑anion cross term.
    epsilon_ion_pair : float, default ``0.102``
        Lennard‑Jones epsilon for the cation‑anion cross term.
    neon_charge : float, default ``0.0``
        Charge of the neon atoms (typically neutral).
    epsilon : float, default ``0.102``
        Default epsilon for interactions involving water oxygen.
    sigma : float, default ``3.188``
        Default sigma for interactions involving water oxygen.

    Returns
    -------
    pandas.DataFrame
        DataFrame with a single entry describing the TIP3P water potential
        together with the full LAMMPS input script (as a list of strings) that
        defines groups, charges, pair styles, bond/angle styles, and kspace
        settings.
    """
    # Build the configuration strings; they are stored as a list of lines.
    config_lines = [
        "# @potential_species H_O  ### species in potential\n",
        "# W.L. Jorgensen",
        "The Journal of Chemical Physics 79",
        "926 (1983); https://doi.org/10.1063/1.445869 \n",
        "#\n",
        "\n",
        "units      real\n",
        "dimension  3\n",
        "atom_style full\n",
        "\n",
        "# create groups ###\n",
        "group O type 2\n",
        "group H type 1\n",
        f"group {metal} type 3\n",
        "group Ne type 4\n",
        f"group {cation} type 5\n",
        f"group {anion} type 6\n",
        "\n",
        "## set charges - beside manually ###\n",
        "set group O charge -0.830\n",
        "set group H charge 0.415\n",
        f"set group {metal} charge {metal_charge}\n",
        f"set group Ne charge {neon_charge}\n",
        f"set group {cation} charge {cation_charge}\n",
        f"set group {anion} charge {anion_charge}\n",
        "\n",
        "### TIP3P Potential Parameters ###\n",
        "pair_style lj/cut/coul/long 12.0\n",
        "pair_modify mix geometric\n",
        "pair_coeff 1 1 0.000 0.000 \n",
        "pair_coeff 2 2 0.102 3.188 \n",
        "pair_coeff 3 3 0.350 2.62  \n",
        "pair_coeff 4 4 0.010 3.000 \n",
        "pair_coeff 5 5 0.102 1.188 \n",
        "pair_coeff 6 6 0.102 1.188 \n",
        f"pair_coeff 2 3 {epsilon:.4f} {sigma:.4f} \n",
        f"pair_coeff 2 4 {epsilon:.4f} {sigma:.4f} \n",
        f"pair_coeff 2 5 {epsilon_cation:.4f} {sigma_cation:.4f} \n",
        f"pair_coeff 3 5 {epsilon_cation:.4f} {sigma_cation:.4f} \n",
        f"pair_coeff 3 6 {epsilon_anion:.4f} {sigma_anion:.4f} \n",
        f"pair_coeff 4 5 {epsilon_cation:.4f} {sigma_cation:.4f} \n",
        f"pair_coeff 4 6 {epsilon_anion:.4f} {sigma_anion:.4f} \n",
        f"pair_coeff 5 6 {epsilon_ion_pair:.4f} {sigma_ion_pair:.4f} \n",
        "pair_coeff 1 3 0.010 1.60 \n",
        "pair_coeff 1 4 0.010 1.60 \n",
        "pair_coeff 1 5 0.010 1.60 \n",
        "pair_coeff 1 6 0.020 1.60 \n",
        "################################################\n",
        "bond_style  harmonic\n",
        "bond_coeff  1 450 0.9572\n",
        "angle_style harmonic\n",
        "angle_coeff 1 55 104.52\n",
        "kspace_style pppm 1.0e-5   # final npt relaxation\n",
        "\n",
    ]

    water_potential = pd.DataFrame(
        {
            "Name": ["H2O_tip3p"],
            "Filename": [[]],
            "Model": ["TIP3P"],
            "Species": [["H", "O", metal, "Ne", cation, anion]],
            "Config": [config_lines],
        }
    )

    return water_potential


@as_function_node("Ion_density")
def element_density(trajectory, initial_structure, initial_step: int = 0):
    """
    Animate a series of atomic structures.

    Parameters
    ----------
    trajectory : Trajectory‑like object
        An object that provides ``positions`` (e.g. a pyiron
        ``Trajectory`` or any object with a ``positions`` attribute).
    initial_structure : Structure‑like object
        The reference structure that defines the atomic species,
        lattice vectors, etc.
    Returns

    """
    import numpy as np

    electrolyte = initial_structure.copy()

    ind_O = electrolyte.select_index("O")
    ind_H = electrolyte.select_index("H")
    ind_Ne = electrolyte.select_index("Ne")
    ind_Al = electrolyte.select_index("Al")
    ind_Na = electrolyte.select_index("Na")
    ind_F = electrolyte.select_index("F")

    slab_bot = np.max(electrolyte.positions[ind_Al, 2])
    slab_top = np.max(electrolyte.positions[ind_Ne, 2])

    positions = trajectory.positions[initial_step:]

    Data = {}

    for element, ind_el in zip(
        ["Na", "F", "O", "H"], [ind_Na, ind_F, ind_O, ind_H], strict=False
    ):
        Data[element] = []

        z_el = np.array([snapshot[ind_el, 2] for snapshot in positions])
        z_d = z_el - slab_bot
        deltares = 0.2
        binedges = np.arange(0, (slab_top - slab_bot), deltares)  #
        hist, bin_edges = np.histogram(z_d, bins=binedges)

        Data[element] = bin_edges[:-1], hist / np.shape(positions)[0]
    return Data


def Ion_density(trajectory, initial_structure):
    """
    Animate a series of atomic structures.

    Parameters
    ----------
    trajectory : Trajectory‑like object
        An object that provides ``positions`` (e.g. a pyiron
        ``Trajectory`` or any object with a ``positions`` attribute).
    initial_structure : Structure‑like object
        The reference structure that defines the atomic species,
        lattice vectors, etc.
    Returns

    """
    import numpy as np

    electrolyte = initial_structure.copy()

    ind_O = electrolyte.select_index("O")
    ind_H = electrolyte.select_index("H")
    ind_Ne = electrolyte.select_index("Ne")
    ind_Al = electrolyte.select_index("Al")
    ind_Na = electrolyte.select_index("Na")
    ind_F = electrolyte.select_index("F")

    slab_bot = np.max(electrolyte.positions[ind_Al, 2])
    slab_top = np.max(electrolyte.positions[ind_Ne, 2])

    positions = trajectory.positions

    Data = {}

    for element, ind_el in zip(
        ["Na", "F", "O", "H"], [ind_Na, ind_F, ind_O, ind_H], strict=False
    ):
        Data[element] = []

        z_el = np.array([snapshot[ind_el, 2] for snapshot in positions])
        z_d = z_el - slab_bot
        deltares = 0.2
        binedges = np.arange(0, (slab_top - slab_bot), deltares)  #
        hist, bin_edges = np.histogram(z_d, bins=binedges)

        Data[element] = bin_edges[:-1], hist / np.shape(positions)[0]
    return Data


@as_function_node("Charge_distribution_Ion")
def Charge_distribution_Ion(Data: dict, initial_structure):

    import numpy as np
    from matplotlib import pyplot as plt

    Sum = np.array(Data["Na"][1]) - np.array(Data["F"][1])
    z = np.array(Data["Na"][0])

    fig, ax = plt.subplots(1, 1, figsize=(3.25, 2))
    ax.plot(z, Sum)
    ax.set_title("Charge Distribution of Ion")

    ax.set_xlabel(r"z - coordinate ($\mathrm{\AA}$)")

    ax.set_ylabel(r"$\rho_\mathrm{e} $ (e/bohr$^3$)")  # \times 10^4

    return fig


@as_function_node("EPD")
def EPD(Data: dict, initial_structure):
    import numpy as np
    from ase.units import Bohr
    from matplotlib import pyplot as plt

    fig, axs = plt.subplots(4, 1, figsize=(3.25, 2 * 4))
    fig.subplots_adjust(hspace=0.15)  # , wspace=0.5)
    fig.subplots_adjust(hspace=0.3 / (2 / 3.25), wspace=0.3)

    y = (
        np.array(Data["Na"][1])
        - np.array(Data["F"][1])
        - 0.83 * np.array(Data["O"][1])
        + 0.415 * np.array(Data["H"][1])
    )
    z = np.array(Data["Na"][0])

    def get_volume(deltares):
        V = (
            initial_structure.cell[0, 0]
            * initial_structure.cell[1, 1]
            * deltares
            * (1 / Bohr) ** 3
        )
        return V

    electron = -y / get_volume(np.gradient(z)[0])

    epsilon_0 = 8.854187817e-12  # Permittivity of free space (in F/m)
    e = 1.602176634e-19  # Elementary charge (in C)

    def e_c_E_potential(x, rho_e, color, axs):
        angstrom_to_meter = 10 ** (-10)
        axs[0].plot(x, rho_e, color=color)

        rho_c = -rho_e * e * (1 / Bohr * 1 / angstrom_to_meter) ** 3
        axs[1].plot(x, rho_c, color=color)

        E = 1 / epsilon_0 * np.cumsum(rho_c * np.gradient(x * angstrom_to_meter))
        axs[2].plot(x, E, color=color)

        V = np.cumsum(E * np.gradient(x * angstrom_to_meter))
        axs[3].plot(x, V, color=color)

        for ax in axs.flatten():
            ax.set(xlim=[x[0], x[-1]])
        ax.set_xlabel(r"z - coordinate ($\mathrm{\AA}$)")

        axs[3].set_ylabel(r"$\phi^{(i)}$ (V)")
        axs[2].set_ylabel(r"$E$ (V/m)")
        axs[1].set_ylabel(r"$\rho_\mathrm{c}$ (C/m$^3$)")
        axs[0].set_ylabel(r"$\rho_\mathrm{e} $ (e/bohr$^3$)")  # \times 10^4

    e_c_E_potential(z, electron, "black", axs)

    return fig


@as_function_node("Suplot_ion")
def Subplot_ion_d(Data: dict, xlabel="x", ylabel="y"):
    import numpy as np
    from matplotlib import pyplot as plt

    species = [s for s in ["Na", "F"] if s in Data] or list(Data.keys())
    fig, axes = plt.subplots(1, len(species), figsize=(5 * len(species), 3))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    for ax, sp in zip(axes, species, strict=False):
        x, y = Data[sp]
        print(sp, x, y)
        ax.plot(x, y)
        ax.set_title(sp)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
    return fig


@as_function_node("Suplot_ion")
def PlotIonDensity(Data: dict, xlabel="x", ylabel="y"):
    from matplotlib import pyplot as plt

    species = [s for s in ["Na", "F"] if s in Data] or list(Data.keys())
    fig, axes = plt.subplots(1, 1, figsize=(5, 3))
    for sp in species:
        x, y = Data[sp]
        print(sp, x, y)
        axes.plot(x, y)
    axes.set_title(sp)
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.grid(True, alpha=0.3)
    return fig


@as_function_node("Suplot_element")
def Subplot_element(Data: dict, element, xlabel="x", ylabel="y"):
    from matplotlib import pyplot as plt

    x, y = Data[element]
    fig, ax = plt.subplots(1, 1, figsize=(3.25, 2))
    ax.plot(x, y)
    ax.set_title(f"Density of {element}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return fig


@as_function_node
def PlotTrajectory(
    md_data: OutputCalcMD,
    species: str,
    index: Literal["x", "y", "z"] = "z",
) -> Figure:
    """
    Plot the trajectory of a single atomic species along a chosen Cartesian
    direction.

    Parameters
    ----------
    md_data : OutputCalcMD
        Object that contains the MD results.  It must provide the attributes
        ``species`` (list of element symbols), ``indices`` (atom IDs) and
        ``unwrapped_positions`` (numpy array of shape (n_steps, n_atoms, 3)).
    species : str
        Chemical symbol of the atoms to be plotted (e.g. ``"Pt"``).
    index : {"x", "y", "z"}, optional
        Cartesian component to plot.  Default is ``"z"``.

    Returns
    -------
    matplotlib.figure.Figure
        The figure containing the trajectory line.
    """
    # ------------------------------------------------------------------
    # Imports – kept inside the node so that the node can be imported without
    # pulling heavy optional dependencies.
    # ------------------------------------------------------------------

    import numpy as np
    from matplotlib import pyplot as plt

    # ------------------------------------------------------------------
    # Parse the ``species`` argument – allow a space separated list.
    # ------------------------------------------------------------------
    wanted_species = set(species.split())  # e.g. {"Al","O","H"}

    # ------------------------------------------------------------------
    # Convert the axis label to an integer (0 → x, 1 → y, 2 → z)
    # ------------------------------------------------------------------
    axis_index = {"x": 0, "y": 1, "z": 2}[index]

    # ------------------------------------------------------------------
    # Find *all* atom indices that belong to the requested species.
    # ------------------------------------------------------------------
    # md_data.species is assumed to be an iterable of strings with length = n_atoms
    fig, ax = plt.subplots(1, 1, figsize=(3.25, 2))
    species_array = np.asarray(md_data.species)  # shape (n_atoms,)
    colors = ["r", "g", "b", "y", "k"]
    for i_el, el in enumerate(wanted_species):
        mask = species_array == el  # bool mask, shape (n_atoms,)
        if len(np.where(mask)[0]) == 0:
            continue

        # Convert the boolean mask to integer atom IDs (or simply use the mask later)
        species_id = np.where(mask)[0][0]  # 1‑D array of atom indices
        selected_atom_ids = np.argwhere(md_data.indices[0] == species_id).flatten()
        print("selected_atom_ids: ", el, species_id, selected_atom_ids)

        # ------------------------------------------------------------------
        # Slice the position array: (time, atom, xyz) → (time, selected_atom, axis)
        # ------------------------------------------------------------------
        traj = md_data.unwrapped_positions[
            :, selected_atom_ids, axis_index
        ]  # shape (n_steps, n_selected)

        # ------------------------------------------------------------------
        # Plot – each selected atom will be plotted as a separate line.
        # ------------------------------------------------------------------
        # ``ax.plot`` with a 2‑D array draws one line per column and returns a
        # list of Line2D objects.  We set the label only on the first object.
        # ------------------------------------------------------------------
        lines = ax.plot(traj, color=colors[i_el % len(colors)])
        # assign the legend label to the first line only
        lines[0].set_label(el)

    # ------------------------------------------------------------------
    # Finalise the figure
    # ------------------------------------------------------------------
    ax.set_xlabel("Time step")
    ax.set_ylabel(f"Position ({index}) / Å")
    ax.set_title(f"{species} trajectory along {index}")

    if len(wanted_species) <= 10:  # avoid a huge legend
        ax.legend(fontsize="xx-small", ncol=2, loc="upper right")

    ax.grid(True, linestyle=":")

    return fig
