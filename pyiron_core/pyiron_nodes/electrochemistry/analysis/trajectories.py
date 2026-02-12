from typing import Literal

from matplotlib.figure import Figure

from pyiron_core.pyiron_nodes.atomistic.calculator.data import OutputCalcMD
from pyiron_core.pyiron_workflow import as_function_node


@as_function_node("element_density")
def element_density(
    trajectory,
    initial_structure,
    initial_step: int = 0,
    working_electrode: str = "Al",
    reference_electrode: str = "Ne",
    solvent: Literal["water"] = "water",
    solvated_ions: str = "Na,F",
):
    """
    Compute 1D number-density profiles along the z direction for selected species.

    Parameters
    ----------
    trajectory
        Object with a ``positions`` array of shape (n_steps, n_atoms, 3).
    initial_structure
        Reference structure used to identify atom indices by element.
    initial_step
        First MD step to include in the analysis.
    working_electrode, reference_electrode
        Element symbols used to define the z-range (bottom/top) of the histogram.
    solvent
        Solvent type. Currently only ``"water"`` is supported.
    solvated_ions
        Comma-separated list of element symbols to include (e.g. ``"Na,F"``).

    Returns
    -------
    dict
        Mapping element symbol -> (z_bins, density), where ``z_bins`` are the bin
        left-edges (Å) relative to the working electrode and ``density`` is the
        per-bin count averaged over frames.
    """
    import numpy as np

    electrolyte = initial_structure.copy()

    if solvent == "water":
        solvent_species = ["O", "H"]
    else:
        raise ValueError(f"Unknown solvent: {solvent}")
    solvated_species = [s.strip() for s in solvated_ions.split(",") if s.strip()]

    ind_work = electrolyte.select_index(working_electrode)
    ind_ref = electrolyte.select_index(reference_electrode)
    slab_bot = np.max(electrolyte.positions[ind_work, 2])
    slab_top = np.max(electrolyte.positions[ind_ref, 2])

    positions = trajectory.positions[initial_step:]

    data = {}

    for element in solvated_species + solvent_species:
        ind_el = electrolyte.select_index(element)
        z_el = np.array([snapshot[ind_el, 2] for snapshot in positions])
        z_d = z_el - slab_bot
        deltares = 0.2
        binedges = np.arange(0, (slab_top - slab_bot), deltares)  #
        hist, bin_edges = np.histogram(z_d, bins=binedges)
        data[element] = bin_edges[:-1], hist / np.shape(positions)[0]
    return data


@as_function_node("charge_distribution_ion")
def charge_distribution_ion(data: dict):

    import numpy as np
    from matplotlib import pyplot as plt

    charge_sum = np.array(data["Na"][1]) - np.array(data["F"][1])
    z = np.array(data["Na"][0])

    fig, ax = plt.subplots(1, 1, figsize=(3.25, 2))
    ax.plot(z, charge_sum)
    ax.set_title("Charge Distribution of Ion")

    ax.set_xlabel(r"z - coordinate ($\mathrm{\AA}$)")

    ax.set_ylabel(r"$\rho_\mathrm{e} $ (e/bohr$^3$)")  # \times 10^4

    return fig


@as_function_node("epd")
def epd(data: dict, initial_structure):
    import numpy as np
    from ase.units import Bohr
    from matplotlib import pyplot as plt

    fig, axs = plt.subplots(4, 1, figsize=(3.25, 2 * 4))
    fig.subplots_adjust(hspace=0.15)  # , wspace=0.5)
    fig.subplots_adjust(hspace=0.3 / (2 / 3.25), wspace=0.3)

    y = (
        np.array(data["Na"][1])
        - np.array(data["F"][1])
        - 0.83 * np.array(data["O"][1])
        + 0.415 * np.array(data["H"][1])
    )
    z = np.array(data["Na"][0])

    def get_volume(deltares):
        v = (
            initial_structure.cell[0, 0]
            * initial_structure.cell[1, 1]
            * deltares
            * (1 / Bohr) ** 3
        )
        return v

    electron = -y / get_volume(np.gradient(z)[0])

    epsilon_0 = 8.854187817e-12  # Permittivity of free space (in F/m)
    e = 1.602176634e-19  # Elementary charge (in C)

    def e_c_E_potential(x, rho_e, color, axs):
        angstrom_to_meter = 10 ** (-10)
        axs[0].plot(x, rho_e, color=color)

        rho_c = -rho_e * e * (1 / Bohr * 1 / angstrom_to_meter) ** 3
        axs[1].plot(x, rho_c, color=color)

        e_field = 1 / epsilon_0 * np.cumsum(rho_c * np.gradient(x * angstrom_to_meter))
        axs[2].plot(x, e_field, color=color)

        v = np.cumsum(e_field * np.gradient(x * angstrom_to_meter))
        axs[3].plot(x, v, color=color)

        for ax in axs.flatten():
            ax.set(xlim=[x[0], x[-1]])
        ax.set_xlabel(r"z - coordinate ($\mathrm{\AA}$)")

        axs[3].set_ylabel(r"$\phi^{(i)}$ (V)")
        axs[2].set_ylabel(r"$E$ (V/m)")
        axs[1].set_ylabel(r"$\rho_\mathrm{c}$ (C/m$^3$)")
        axs[0].set_ylabel(r"$\rho_\mathrm{e} $ (e/bohr$^3$)")  # \times 10^4

    e_c_E_potential(z, electron, "black", axs)

    return fig


@as_function_node("subplot_ion")
def subplot_ion_d(data: dict, xlabel="x", ylabel="y"):
    import numpy as np
    from matplotlib import pyplot as plt

    species = [s for s in ["Na", "F"] if s in data] or list(data.keys())
    fig, axes = plt.subplots(1, len(species), figsize=(5 * len(species), 3))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    for ax, sp in zip(axes, species, strict=False):
        x, y = data[sp]
        print(sp, x, y)
        ax.plot(x, y)
        ax.set_title(sp)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
    return fig


@as_function_node("plot_ion_density")
def plot_ion_density(data: dict, xlabel="x", ylabel="y"):
    from matplotlib import pyplot as plt

    species = [s for s in ["Na", "F"] if s in data] or list(data.keys())
    fig, axes = plt.subplots(1, 1, figsize=(5, 3))
    for sp in species:
        x, y = data[sp]
        print(sp, x, y)
        axes.plot(x, y)
    axes.set_title(sp)
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.grid(True, alpha=0.3)
    return fig


@as_function_node("subplot_element")
def subplot_element(data: dict, element, xlabel="x", ylabel="y"):
    from matplotlib import pyplot as plt

    x, y = data[element]
    fig, ax = plt.subplots(1, 1, figsize=(3.25, 2))
    ax.plot(x, y)
    ax.set_title(f"Density of {element}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return fig


@as_function_node
def plot_trajectory(
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


@as_function_node("water_density_analysis")
def water_dens_calc(initial_structure, trajectory, frm_st=0, binsnum=500):

    import ase.units as units
    import numpy as np

    o_ind = initial_structure.select_index("O")

    # Extract the positions of the oxygen atoms along the z-axis
    z_pos = trajectory.positions[frm_st:, o_ind, 2].flatten()

    # Define the simulation box dimensions (for example)
    cell = initial_structure.cell
    Lx, Ly, Lz = (
        np.linalg.norm(cell[0]),
        np.linalg.norm(cell[1]),
        np.linalg.norm(cell[2]),
    )
    area_xy = Lx * Ly

    # Calculate histogram with density=False to get the number of atoms in each bin
    hist, bin_edges = np.histogram(z_pos, binsnum, range=[0, Lz], density=False)
    z_axis = np.linspace(0, Lz, binsnum)
    bin_width = bin_edges[1] - bin_edges[0]

    # Spatial density along z (number of atoms per unit volume along z)
    mol_mass_water = 18.015  # g/mol
    spatial_density_z = (
        mol_mass_water
        * hist
        / units.mol
        / (bin_width * area_xy)
        * 1.0e24
        / len(trajectory.positions[frm_st:])
    )
    return z_axis, spatial_density_z
