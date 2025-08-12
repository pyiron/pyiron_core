from pyiron_workflow import as_function_node
from pyiron_nodes.atomistic.calculator.data import InputCalcMD


@as_function_node
def Equilibrate(
    solvated_electrode,
    water_potential,
    parameters: InputCalcMD = InputCalcMD().run(),
    store: bool = True,
):
    """
    Function to equilibrate a solvated electrode structure using LAMMPS.

    Parameters:
        solvated_electrode: The electrode structure with solvent.
        pr: The pyiron project instance.
        water_potential: DataFrame containing potential parameters for water.

    Returns:
        None
    """
    from pyiron_atomistics import Project
    from dataclasses import asdict
    from pyiron_nodes.atomistic.calculator.data import OutputCalcMD

    # Create a job for LAMMPS equilibration
    pr = Project("equilibrate")
    j = pr.create.job.Lammps("water_equilibration", delete_existing_job=True)

    solvated_electrode.add_tag(selective_dynamics=[False, False, False])
    solvated_electrode.selective_dynamics[
        solvated_electrode.select_index(["O", "H"])
    ] = [
        True,
        True,
        True,
    ]

    j.structure = solvated_electrode
    j.potential = water_potential
    j.calc_md(**asdict(parameters))

    j.run(delete_existing_job=True)
    job_out = j["output/generic"]

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

    return out


@as_function_node
def WaterPotential(metal: str = "Al", epsilon=0.102, sigma=3.188):
    import pandas

    water_potential = pandas.DataFrame(
        {
            "Name": ["H2O_tip3p"],
            "Filename": [[]],
            "Model": ["TIP3P"],
            "Species": [["H", "O", metal]],
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
                    "\n",
                    "## set charges - beside manually ###\n",
                    "set group O charge -0.830\n",
                    "set group H charge 0.415\n",
                    f"set group {metal} charge 0.2\n",
                    "\n",
                    "### TIP3P Potential Parameters ###\n",
                    "pair_style lj/cut/coul/long 10.0\n",
                    "pair_coeff * * 0.000 0.000 \n",
                    "pair_coeff 2 2 0.102 3.188 \n",
                    "pair_coeff 2 3 {:.4} {:.4} \n".format(epsilon, sigma),
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


@as_function_node("animate")
def Animate(job):
    return job.animate_structures()
