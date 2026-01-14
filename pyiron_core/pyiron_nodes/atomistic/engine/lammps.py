from __future__ import annotations

from dataclasses import asdict
from typing import List, Optional, Tuple

from ase import Atoms
from pandas import DataFrame
from pyiron_atomistics.lammps.control import LammpsControl

from pyiron_core.pyiron_nodes.atomistic.calculator.data import (
    InputCalcMD,
    InputCalcStatic,
)
from pyiron_core.pyiron_nodes.dev_tools import FileObject
from pyiron_core.pyiron_workflow import (
    as_function_node,
    as_inp_dataclass_node,
    as_macro_node,
    as_out_dataclass_node,
)

as_inp_dataclass_node()


@as_inp_dataclass_node
class InputCalcMinimizeLammps:
    ionic_energy_tolerance: float = 0.0
    ionic_force_tolerance: float = 0.0001
    max_iter: float = 100000
    pressure: float = None
    n_print: float = 100
    style: float = "cg"
    rotation_matrix = None


@as_function_node("calculator")
def Calc(parameters):
    from pyiron_atomistics.lammps.control import LammpsControl

    calculator = LammpsControl()

    if isinstance(parameters, InputCalcMD().dataclass):
        calculator.calc_md(**asdict(parameters))
        calculator.mode = "md"
    elif isinstance(parameters, InputCalcMinimizeLammps):
        calculator.calc_minimize(**parameters)
        calculator.mode = "minimize"
    elif isinstance(parameters, InputCalcStatic):
        calculator.calc_static(**parameters)
        calculator.mode = "static"
    else:
        raise TypeError(f"Unexpected parameters type {parameters}")

    return calculator


@as_function_node("calculator")
def CalcStatic(calculator_input: Optional[InputCalcStatic] = None):
    if calculator_input is None:
        calculator_input = InputCalcStatic().run()

    calculator_kwargs = asdict(calculator_input)
    calculator = LammpsControl()
    calculator.calc_static(**calculator_kwargs)
    calculator.mode = "static"
    calculator["dump___1"] = calculator["dump___1"].replace("su", "u")

    return calculator


@as_function_node("calculator")
def CalcMinimize(calculator_input: Optional[InputCalcMinimizeLammps] = None):
    if calculator_input is None:
        calculator_input = InputCalcMinimizeLammps().run()

    calculator_kwargs = asdict(calculator_input)
    calculator = LammpsControl()
    calculator.calc_minimize(**calculator_kwargs)
    calculator.mode = "static"
    calculator["dump___1"] = calculator["dump___1"].replace("su", "u")

    return calculator


@as_function_node("calculator")
def CalcMD(calculator_input: Optional[InputCalcMD] = None):
    from dataclasses import asdict

    if calculator_input is None:
        calculator_input = InputCalcMD().run()

    calculator_kwargs = asdict(calculator_input)
    calculator = LammpsControl()
    calculator.calc_md(**calculator_kwargs)
    calculator.mode = "md"
    # use absolute coordinates rather than relative ones
    calculator["dump___1"] = calculator["dump___1"].replace("su", "u")

    return calculator


def scale_number_in_string(s: str, scale: float, index: int) -> str:
    """
    Scale the number at a given index in the string by `scale`.

    Parameters
    ----------
    s : str
        Input string containing numbers.
    scale : float
        Scaling factor to multiply the selected number.
    index : int
        Index of the number to scale (0-based, can be negative).

    Returns
    -------
    str
        String with the selected number scaled.

    Example
    -------
    >>> scale_number_in_string("all nvt temp 300.0 300.0 0.1", 1000, -1)
    'all nvt temp 300.0 300.0 100.000000'
    """
    import re

    # Find all numbers as they appear in the string
    numbers = re.findall(r"[-+]?\d*\.\d+|\d+", s)
    if not numbers:
        return s  # no change if no numbers

    # Handle negative indices
    if index < 0:
        index = len(numbers) + index
    if index < 0 or index >= len(numbers):
        raise IndexError("Index out of range for numbers in string")

    # Scale the chosen number
    old_num_str = numbers[index]
    old_num_val = float(old_num_str)
    new_num_val = old_num_val * scale

    # Replace only the chosen occurrence
    count = 0

    def replace_nth(match):
        nonlocal count
        if count == index:
            count += 1
            return f"{new_num_val:.6f}"
        count += 1
        return match.group(0)

    s_new = re.sub(r"[-+]?\d*\.\d+|\d+", replace_nth, s)
    return s_new


@as_function_node("path")
def InitLammps(
    structure: Atoms,
    potential: str | DataFrame,
    calculator,
    working_directory: str,
    create_dir: bool = True,
):
    import os

    from pyiron_atomistics.lammps.potential import LammpsPotential, LammpsPotentialFile

    if create_dir:
        os.makedirs(working_directory, exist_ok=True)
    else:
        assert os.path.isdir(
            working_directory
        ), f"working directory {working_directory} is missing, create it!"

    pot = LammpsPotential()
    if isinstance(potential, str):
        pot.df = LammpsPotentialFile().find_by_name(potential)
    elif isinstance(potential, DataFrame):
        pot.df = potential

    units = None
    if "units" in pot.keys():
        from pyiron_atomistics import pyiron_to_ase

        print("Potential contains structure block")
        units = pot["units"]
        calculator["units"] = pot["units"]
        calculator["atom_style"] = pot["atom_style"]
        calculator["dimension"] = pot["dimension"]
        calculator["timestep"] *= 1000
        calculator["fix___ensemble"] = scale_number_in_string(
            calculator["fix___ensemble"], 1000, -1
        )
        print("timestep: ", calculator["timestep"])
        print("fix___ensemble", calculator["fix___ensemble"])

        pot.remove_structure_block()

        # presently used only for TIP3P water
        bonds, angles = compute_water_bonds_indices(structure)
        structure = pyiron_to_ase(structure)
        bonds_array = make_bonds_array(bonds, n_atoms=len(structure))
        print("bonds_array: ", bonds_array)
        structure.set_array("bonds", bonds_array)

        angles_array = make_angles_array(angles, n_atoms=len(structure))
        structure.set_array("angles", angles_array)

    pot.write_file(file_name="potential.inp", cwd=working_directory)
    pot.copy_pot_files(working_directory)

    with open(os.path.join(working_directory, "structure.inp"), "w") as f:
        if not units:
            structure.write(f, format="lammps-data", specorder=pot.get_element_lst())
        else:
            charges = extract_charges_from_lammps_potential(pot.df.Config[0])
            write_lammps_data_full(
                f,
                structure,
                specorder=pot.get_element_lst(),
                bonds=True,
                atom_style="full",
                units="real",
                masses=True,
                charges=charges,
            )

    calculator.write_file(file_name="control.inp", cwd=working_directory)

    return os.path.abspath(working_directory)


def compute_water_bonds_indices(
    structure: Atoms,
    cutoff: float = 1.2,
):
    """Return integer bond and angle connectivity for water molecules.

    Uses :func:`structuretoolkit.get_neighbors` to obtain distances respecting
    periodic boundary conditions.

    Parameters
    ----------
    structure:
        ASE ``Atoms`` object containing water molecules.
    cutoff:
        Maximum O–H distance (Å) to be considered a bond.

    Returns
    -------
    tuple
        ``(bonds, angles)`` where ``bonds`` is a list of ``(O_idx, H_idx)`` and
        ``angles`` is a list of ``(O_idx, H1_idx, H2_idx)``.  Indices are zero‑based.
    """

    from structuretoolkit import get_neighbors

    # Chemical symbols as a simple list; we locate O and H indices using
    o_idxs = structure.select_index("O")
    h_idxs = structure.select_index("H")

    bonds: List[Tuple[int, int]] = []
    angles: List[Tuple[int, int, int]] = []

    # Loop over each oxygen atom and find hydrogen neighbors within ``cutoff``.
    neigh_obj = get_neighbors(structure, num_neighbors=5, cutoff_radius=cutoff)
    for o in o_idxs:
        neighbor_ids = neigh_obj.indices[o]
        neighbor_dists = neigh_obj.distances[o]

        h_neighbors = [
            int(idx)
            for idx, d in zip(neighbor_ids, neighbor_dists, strict=False)
            if (d < cutoff) and (idx in h_idxs)
        ]

        for h in h_neighbors:
            bonds.append((int(o), int(h)))

        if len(h_neighbors) == 2:
            angles.append((int(o), int(h_neighbors[0]), int(h_neighbors[1])))
        print("bonds: ", o, h_neighbors)

    print("bonds: ", bonds)
    return bonds, angles


@as_function_node("log")
def ParseLogFile(log_file):
    from pymatgen.io.lammps.outputs import parse_lammps_log

    log = parse_lammps_log(log_file.path)
    if len(log) == 0:
        print(f"check {log_file.path}")
        raise ValueError("lammps_log_parser: failed")

    return log


@as_function_node("dump")
def ParseDumpFile(dump_file):
    from pymatgen.io.lammps.outputs import parse_lammps_dumps

    dump = list(parse_lammps_dumps(dump_file.path))
    return dump


@as_out_dataclass_node
class ShellOutput:
    stdout: str = ""
    stderr: str = ""
    return_code: int = 0


@as_function_node("output", "dump", "log")
def Shell(
    working_directory: str,
    command: str = "lmp",
    environment: Optional[dict] = None,
    arguments: Optional[list] = None,
):
    arguments = ["-in", "control.inp"] if arguments is None else arguments
    import os
    import subprocess

    if environment is None:
        environment = {}
    if arguments is None:
        arguments = []

    environ = dict(os.environ)
    environ.update({k: str(v) for k, v in environment.items()})
    proc = subprocess.run(
        [command, *map(str, arguments)],
        capture_output=True,
        cwd=working_directory,
        encoding="utf8",
        env=environ,
    )

    output = ShellOutput()
    output.stdout = proc.stdout
    output.stderr = proc.stderr
    output.return_code = proc.returncode
    dump = FileObject("dump.out", working_directory)
    log = FileObject("log.lammps", working_directory)

    return output, dump, log


@as_out_dataclass_node
class GenericOutput:
    energy_pot = []
    energy_kin = []
    forces = []


@as_function_node
def Collect(
    out_dump,
    out_log,
    calc_mode,
):
    import numpy as np

    from pyiron_core.pyiron_nodes.atomistic.calculator.data import (
        OutputCalcMD,
        OutputCalcMinimize,
        OutputCalcStatic,
    )

    log = out_log[0]

    if isinstance(calc_mode, str) and calc_mode in ["static", "minimize", "md"]:
        pass
    elif isinstance(calc_mode, (InputCalcMinimizeLammps, InputCalcMD, InputCalcStatic)):
        calc_mode = calc_mode.__class__.__name__.replace("InputCalc", "").lower()
    elif isinstance(calc_mode, LammpsControl):
        calc_mode = calc_mode.mode
    else:
        raise ValueError(f"Unexpected calc_mode {calc_mode}")

    if calc_mode == "static":
        generic = OutputCalcStatic().dataclass()
        generic.energy_pot = log["PotEng"].values[0]
        generic.force = np.array([o.data[["fx", "fy", "fz"]] for o in out_dump])[0]

    elif calc_mode == "minimize":
        generic = OutputCalcMinimize().dataclass()

    elif calc_mode == "md":
        generic = OutputCalcMD().dataclass()
        generic.energies_tot = log["TotEng"].values
        generic.energies_pot = log["PotEng"].values
        generic.steps = log["Step"].values
        generic.temperatures = log["Temp"].values
        generic.volumes = log["Volume"].values
        # read Pxx, Pxy, Pxz etc. and construct pressures array of 3x3 matrices
        Pxx = log["Pxx"].values
        Pyy = log["Pyy"].values
        Pzz = log["Pzz"].values
        Pxy = log["Pxy"].values
        Pxz = log["Pxz"].values
        Pyz = log["Pyz"].values
        pressures = np.array([Pxx, Pxy, Pxz, Pxy, Pyy, Pyz, Pxz, Pyz, Pzz])
        generic.pressures = pressures.reshape(-1, 3, 3)

        generic.forces = np.array([o.data[["fx", "fy", "fz"]] for o in out_dump])
        generic.positions = np.array([o.data[["xu", "yu", "zu"]] for o in out_dump])

    return generic


# TIP3P water box generation
@as_function_node("path")
def Tip3pData(
    working_directory: str,
    n_mols: int = 3,
    box_size: float = 15.0,
    OH_distance: float = 0.9572,
    HOH_angle: float = 104.52,
):
    """Create TIP3P water box LAMMPS data file and input script.
    Writes ``tip3p.data`` and ``in.tip3p`` into ``working_directory`` and returns
    the data file as a ``FileObject``.
    """
    import os

    import numpy as np

    os.makedirs(working_directory, exist_ok=True)

    # Force field parameters (CHARMM TIP3P)
    atom_masses = {1: 15.9994, 2: 1.008}
    atom_charges = {1: -0.834, 2: 0.417}
    bond_type_params = {1: (450.0, OH_distance)}
    angle_type_params = {1: (55.0, HOH_angle)}
    lj_params = {1: (0.1521, 3.1507), 2: (0.0, 0.0)}

    atoms = []
    bonds = []
    angles = []

    atom_id = 1
    bond_id = 1
    angle_id = 1
    mol_id = 1

    spacing = box_size / n_mols
    angle_rad = np.deg2rad(HOH_angle)
    for ix in range(n_mols):
        for iy in range(n_mols):
            for iz in range(n_mols):
                Ox = ix * spacing + spacing / 2
                Oy = iy * spacing + spacing / 2
                Oz = iz * spacing + spacing / 2

                H1x = Ox + OH_distance
                H1y = Oy
                H1z = Oz

                H2x = Ox + OH_distance * np.cos(angle_rad)
                H2y = Oy + OH_distance * np.sin(angle_rad)
                H2z = Oz

                atoms.append((atom_id, mol_id, 1, atom_charges[1], Ox, Oy, Oz))
                atom_id += 1
                atoms.append((atom_id, mol_id, 2, atom_charges[2], H1x, H1y, H1z))
                atom_id += 1
                atoms.append((atom_id, mol_id, 2, atom_charges[2], H2x, H2y, H2z))
                atom_id += 1

                bonds.append((bond_id, 1, atom_id - 3, atom_id - 2))
                bond_id += 1
                bonds.append((bond_id, 1, atom_id - 3, atom_id - 1))
                bond_id += 1

                angles.append((angle_id, 1, atom_id - 2, atom_id - 3, atom_id - 1))
                angle_id += 1

                mol_id += 1

    # write data file
    data_path = os.path.join(working_directory, "tip3p.data")
    with open(data_path, "w") as f:
        f.write("TIP3P water box\n\n")
        f.write(f"{len(atoms)} atoms\n")
        f.write(f"{len(bonds)} bonds\n")
        f.write(f"{len(angles)} angles\n")
        f.write("0 dihedrals\n0 impropers\n\n")
        f.write(f"{len(atom_masses)} atom types\n")
        f.write(f"{len(bond_type_params)} bond types\n")
        f.write(f"{len(angle_type_params)} angle types\n\n")
        f.write(f"0.0 {box_size} xlo xhi\n")
        f.write(f"0.0 {box_size} ylo yhi\n")
        f.write(f"0.0 {box_size} zlo zhi\n\n")
        f.write("Masses\n\n")
        for t, m in atom_masses.items():
            f.write(f"{t} {m}\n")
        f.write("\nBond Coeffs\n\n")
        for t, (k, r0) in bond_type_params.items():
            f.write(f"{t} {k} {r0}\n")
        f.write("\nAngle Coeffs\n\n")
        for t, (k, ang) in angle_type_params.items():
            f.write(f"{t} {k} {ang}\n")
        f.write("\nAtoms\n\n")
        for atom in atoms:
            id_, mol, type_, q, x, y, z = atom
            f.write(f"{id_} {mol} {type_} {q} {x} {y} {z}\n")
        f.write("\nBonds\n\n")
        for bond in bonds:
            id_, type_, a1, a2 = bond
            f.write(f"{id_} {type_} {a1} {a2}\n")
        f.write("\nAngles\n\n")
        for angle in angles:
            id_, type_, a1, a2, a3 = angle
            f.write(f"{id_} {type_} {a1} {a2} {a3}\n")

    # write input script
    inp_path = os.path.join(working_directory, "control.inp")
    with open(inp_path, "w") as f:
        f.write("units real\n")
        f.write("atom_style full\n")
        f.write("boundary p p p\n\n")
        f.write("bond_style harmonic\n")
        f.write("angle_style harmonic\n")
        f.write("pair_style lj/cut/coul/long 10.0\n")
        f.write("pair_modify mix arithmetic\n")
        f.write("kspace_style pppm 1e-4\n\n")
        f.write("read_data tip3p.data\n\n")
        for t, (eps, sig) in lj_params.items():
            for t2, (eps2, sig2) in lj_params.items():
                f.write(f"pair_coeff {t} {t2} {eps} {sig}\n")
        f.write("neighbor 2.0 bin\n")
        f.write("neigh_modify delay 0 every 1 check yes\n\n")
        f.write("group water type 1 2\n")
        f.write("fix shake_w water shake 1e-6 100 0 b 1 a 1\n")
        f.write("timestep 1.0\n")
        # Append additional LAMMPS control statements required for the workflow
        f.write("\n")
        f.write("fix ensemble all nvt temp 3000.0 3000.0 0.1\n")
        f.write("variable dumptime  equal 100\n")
        f.write("variable thermotime  equal 100\n")
        f.write("velocity all create 6000.0 10298 dist gaussian\n")
        f.write(
            "dump 1 all custom ${dumptime} dump.out id type xu yu zu fx fy fz vx vy vz\n"
        )
        f.write(
            'dump_modify 1 sort id format line "%d %d %20.15g %20.15g %20.15g %20.15g %20.15g %20.15g %20.15g %20.15g %20.15g"\n'
        )
        f.write("thermo_style custom step temp pe etotal pxx pxy pxz pyy pyz pzz vol\n")
        f.write("thermo_modify format float %20.15g\n")
        f.write("thermo ${thermotime}\n")
        f.write("run 1000\n")

    return os.path.abspath(working_directory)


@as_function_node("potential")
def Potential(structure, name: Optional[str] = None, index: int = 0):
    from pyiron_atomistics.lammps.potential import list_potentials as lp

    potentials = lp(structure)
    if name is None:
        pot = potentials[index]
    elif isinstance(name, DataFrame):
        pot = name
    else:
        if name in potentials:
            pot = name
        else:
            raise ValueError("Unknown potential")
    return pot


@as_function_node("potentials")
def ListPotentials(structure):
    from pyiron_atomistics.lammps.potential import list_potentials as lp

    potentials = lp(structure)
    return potentials


def get_calculators():
    calc_dict = {}
    calc_dict["md"] = CalcMD
    calc_dict["minimize"] = CalcMinimize
    calc_dict["static"] = CalcStatic

    return calc_dict


@as_function_node("generic")
def GetEnergyPot(generic, i_start: int = 0, i_end: int = -1):
    return generic.energies_pot[i_start:i_end]


@as_macro_node("generic")
def Code(
    structure: Atoms,
    calculator,
    potential: Optional[str] = None,
    working_dir: str = "test2",
):

    from pyiron_core.pyiron_workflow import Workflow

    wf = Workflow("LammpsMacro")

    wf.InitLammps = InitLammps(
        structure=structure,
        potential=potential,
        calculator=calculator,
        working_directory=working_dir,
    )

    wf.Shell = Shell(working_directory=wf.InitLammps)

    wf.ParseLogFile = ParseLogFile(log_file=wf.Shell.outputs.log)
    wf.ParseDumpFile = ParseDumpFile(dump_file=wf.Shell.outputs.dump)
    wf.Collect = Collect(
        out_dump=wf.ParseDumpFile.outputs.dump,
        out_log=wf.ParseLogFile.outputs.log,
        calc_mode="md",
    )

    return wf.Collect


@as_function_node
def DummyNode(structure1: Atoms, structure2: Atoms):
    return structure1


@as_macro_node(labels=["generic", "path"])
def Code1(
    structure: Atoms,
    calculator,
    potential: Optional[str] = None,
    working_dir: str = "test2",
):

    from pyiron_core.pyiron_workflow import Workflow

    wf = Workflow("LammpsMacro")

    wf.Potential = Potential(structure=structure, name=potential)

    wf.ListPotentials = ListPotentials(structure=structure)

    wf.InitLammps = InitLammps(
        structure=structure,
        potential=wf.Potential,
        calculator=calculator,
        working_directory=working_dir,
    )

    wf.Shell = Shell(working_directory=wf.InitLammps)

    wf.ParseLogFile = ParseLogFile(log_file=wf.Shell.outputs.log)
    wf.ParseDumpFile = ParseDumpFile(dump_file=wf.Shell.outputs.dump)
    wf.Collect = Collect(
        out_dump=wf.ParseDumpFile.outputs.dump,
        out_log=wf.ParseLogFile.outputs.log,
        calc_mode="md",
    )

    return wf.Collect, wf.InitLammps.outputs.path


@as_macro_node("generic")
def Lammps(
    structure: Atoms,
    calculator,
    potential: Optional[str] = None,
    working_dir: str = "test2",
    store: bool = False,
):

    from pyiron_core.pyiron_workflow import Workflow

    wf = Workflow("LammpsMacro")

    wf.Potential = Potential(structure=structure, name=potential)

    wf.ListPotentials = ListPotentials(structure=structure)

    wf.InitLammps = InitLammps(
        structure=structure,
        potential=wf.Potential,
        calculator=calculator,
        working_directory=working_dir,
    )

    wf.Shell = Shell(working_directory=wf.InitLammps)

    wf.ParseLogFile = ParseLogFile(log_file=wf.Shell.outputs.log)
    wf.ParseDumpFile = ParseDumpFile(dump_file=wf.Shell.outputs.dump)
    wf.Collect = Collect(
        out_dump=wf.ParseDumpFile.outputs.dump,
        out_log=wf.ParseLogFile.outputs.log,
        calc_mode="md",
    )

    return wf.Collect


@as_function_node
def ParseOutput(working_directory, structure, potential, units="metal"):

    from pyiron_lammps import parse_lammps_output_files

    parse_dict = parse_lammps_output_files(
        working_directory,
        structure,
        potential_elements=potential.Species,
        units=units,
        dump_h5_file_name="dump.h5",
        dump_out_file_name="dump.out",
        log_lammps_file_name="log.lammps",
    )

    return parse_dict


# helper functions (move later to separate library)


def compute_water_bonds_and_angles(
    structure: Atoms,
    cutoff: float = 1.2,
):
    """Return integer bond and angle connectivity for water molecules.

    Uses :func:`structuretoolkit.get_neighbors` to obtain distances respecting
    periodic boundary conditions.

    Parameters
    ----------
    structure:
        ASE ``Atoms`` object containing water molecules.
    cutoff:
        Maximum O–H distance (Å) to be considered a bond.

    Returns
    -------
    tuple
        ``(bonds, angles)`` where ``bonds`` is a list of ``(O_idx, H_idx)`` and
        ``angles`` is a list of ``(O_idx, H1_idx, H2_idx)``.  Indices are zero‑based.
    """

    from structuretoolkit import get_neighbors

    # Chemical symbols as a simple list; we locate O and H indices using
    o_idxs = structure.select_index("O")
    h_idxs = structure.select_index("H")

    bonds: List[Tuple[int, int]] = []
    angles: List[Tuple[int, int, int]] = []

    # Loop over each oxygen atom and find hydrogen neighbors within ``cutoff``.
    neigh_obj = get_neighbors(structure, num_neighbors=5, cutoff_radius=cutoff)
    for o in o_idxs:
        neighbor_ids = neigh_obj.indices[o]
        neighbor_dists = neigh_obj.distances[o]

        h_neighbors = [
            int(idx)
            for idx, d in zip(neighbor_ids, neighbor_dists, strict=False)
            if d < cutoff and idx in h_idxs
        ]

        for h in h_neighbors:
            bonds.append((int(o), int(h)))

        if len(h_neighbors) == 2:
            angles.append((int(o), int(h_neighbors[0]), int(h_neighbors[1])))

    return bonds, angles


def make_bonds_array(atom_pairs, bond_type=1, n_atoms=None):
    """Convert list of 0-based atom index pairs to ASE bonds array strings."""
    import numpy as np

    if n_atoms is None:
        n_atoms = max(max(p) for p in atom_pairs) + 1
    bonds_per_atom = [[] for _ in range(n_atoms)]
    for a1, a2 in atom_pairs:
        bonds_per_atom[a1].append(f"{a2}({bond_type})")
    # Format strings, '_' if no bonds
    return np.array([",".join(b) if b else "_" for b in bonds_per_atom])


def make_angles_array(angle_triplets, angle_type=1, n_atoms=None):
    """Convert list of (a1, central, a3) triplets into ASE angles array format."""
    import numpy as np

    if n_atoms is None:
        n_atoms = max(max(trip) for trip in angle_triplets) + 1
    angles_per_atom = [[] for _ in range(n_atoms)]
    for a1, central, a3 in angle_triplets:
        angles_per_atom[central].append(f"{a1}-{a3}({angle_type})")
    return np.array([",".join(lst) if lst else "_" for lst in angles_per_atom])


def extract_charges_from_lammps_potential(lines):
    """
    Extract charges for atom types from a LAMMPS potential file lines.

    Parameters
    ----------
    lines : list of str
        Lines from the potential file.

    Returns
    -------
    list of float
        Charges in order of atom type id (index 0 = type 1).
        Missing charges are set to None.
    """
    import re

    type_to_group = {}  # map type_id -> group_name
    charges_dict = {}  # map group_name -> charge_value

    # Regex patterns
    group_pattern = re.compile(r"group\s+(\S+)\s+type\s+(\d+)", re.IGNORECASE)
    charge_pattern = re.compile(
        r"set\s+group\s+(\S+)\s+charge\s+(-?\d*\.?\d+)", re.IGNORECASE
    )

    for line in lines:
        line = line.strip()
        # match group lines
        gmatch = group_pattern.match(line)
        if gmatch:
            group_name = gmatch.group(1)
            type_id = int(gmatch.group(2))
            type_to_group[type_id] = group_name
        # match charge lines
        cmatch = charge_pattern.match(line)
        if cmatch:
            group_name = cmatch.group(1)
            charge_val = float(cmatch.group(2))
            charges_dict[group_name] = charge_val

    # Build list of charges ordered by type id
    max_type = max(type_to_group.keys())
    charges_list = [None] * max_type
    for type_id, group_name in type_to_group.items():
        charges_list[type_id - 1] = charges_dict.get(group_name, None)

    return charges_list


def write_lammps_data_full(
    fd,
    atoms,
    *,
    specorder=None,
    reduce_cell=False,
    force_skew=False,
    prismobj=None,
    write_image_flags=False,
    masses=False,
    velocities=False,
    units="metal",
    bonds=True,
    atom_style="atomic",
    charges=None,
):
    """Extended ASE LAMMPS writer with bonds, angles, dihedrals, impropers support.
    charges : list or array of floats (optional)
        If provided, overrides atom charges from `atoms.get_initial_charges()`.
        Must be ordered by type-ID (type 1 first, then type 2, etc.).
    """
    import numpy as np
    from ase.calculators.lammps.coordinatetransform import Prism
    from ase.io.lammpsdata import _write_masses, convert

    print("charges: ", charges)
    if isinstance(atoms, list):
        if len(atoms) > 1:
            raise ValueError("Can only write one configuration to a LAMMPS data file!")
        atoms = atoms[0]

    fd.write("(written by ASE - extended)\n\n")

    symbols = atoms.get_chemical_symbols()
    n_atoms = len(symbols)
    fd.write(f"{n_atoms} atoms\n")

    if specorder is None:
        species = sorted(set(symbols))
    else:
        species = specorder
    n_atom_types = len(species)
    fd.write(f"{n_atom_types} atom types\n\n")

    # -------- Bonds counting -----------
    bonds_in = []
    if bonds and (atom_style == "full") and (atoms.arrays.get("bonds") is not None):
        n_bonds = 0
        n_bond_types = 1
        for i, bondsi in enumerate(atoms.arrays["bonds"]):
            if bondsi != "_":
                for bond in bondsi.split(","):
                    dummy1, dummy2 = bond.split("(")
                    bond_type = int(dummy2.split(")")[0])
                    at1 = i + 1
                    at2 = int(dummy1) + 1
                    bonds_in.append((bond_type, at1, at2))
                    n_bonds += 1
                    if bond_type > n_bond_types:
                        n_bond_types = bond_type
        fd.write(f"{n_bonds} bonds\n")
        fd.write(f"{n_bond_types} bond types\n\n")
    else:
        n_bonds = 0
        n_bond_types = 0

    # -------- Angles counting ----------
    angles_in = []
    if atoms.arrays.get("angles") is not None:
        n_angles = 0
        n_angle_types = 1
        for i, angsi in enumerate(atoms.arrays["angles"]):
            if angsi != "_":
                for angle in angsi.split(","):
                    triple, ang_type_str = angle.split("(")
                    ang_type = int(ang_type_str.split(")")[0])
                    a1_str, a3_str = triple.split("-")
                    center = i + 1
                    a1 = int(a1_str) + 1
                    a3 = int(a3_str) + 1
                    angles_in.append((ang_type, a1, center, a3))
                    n_angles += 1
                    if ang_type > n_angle_types:
                        n_angle_types = ang_type
        fd.write(f"{n_angles} angles\n")
        fd.write(f"{n_angle_types} angle types\n\n")
    else:
        n_angles = 0
        n_angle_types = 0

    # -------- Dihedrals counting -------
    dihedrals_in = []
    if atoms.arrays.get("dihedrals") is not None:
        n_dihedrals = 0
        n_dihedral_types = 1
        for i, dihe in enumerate(atoms.arrays["dihedrals"]):
            if dihe != "_":
                for dih in dihe.split(","):
                    quad, dih_type_str = dih.split("(")
                    dih_type = int(dih_type_str.split(")")[0])
                    a1_str, a2_str, a4_str = quad.split("-")
                    a1 = int(a1_str) + 1
                    a2 = i + 1
                    a4 = int(a4_str) + 1
                    dihedrals_in.append((dih_type, a1, a2, int(a2_str) + 1, a4))
                    n_dihedrals += 1
                    if dih_type > n_dihedral_types:
                        n_dihedral_types = dih_type
        fd.write(f"{n_dihedrals} dihedrals\n")
        fd.write(f"{n_dihedral_types} dihedral types\n\n")
    else:
        n_dihedrals = 0
        n_dihedral_types = 0

    # -------- Impropers counting -------
    impropers_in = []
    if atoms.arrays.get("impropers") is not None:
        n_impropers = 0
        n_improper_types = 1
        for i, impr in enumerate(atoms.arrays["impropers"]):
            if impr != "_":
                for imp in impr.split(","):
                    quad, imp_type_str = imp.split("(")
                    imp_type = int(imp_type_str.split(")")[0])
                    a1_str, a2_str, a4_str = quad.split("-")
                    a1 = int(a1_str) + 1
                    a2 = i + 1
                    a4 = int(a4_str) + 1
                    impropers_in.append((imp_type, a1, a2, int(a2_str) + 1, a4))
                    n_impropers += 1
                    if imp_type > n_improper_types:
                        n_improper_types = imp_type
        fd.write(f"{n_impropers} impropers\n")
        fd.write(f"{n_improper_types} improper types\n\n")
    else:
        n_impropers = 0
        n_improper_types = 0

    # -------- Cell dimensions ----------
    if prismobj is None:
        prismobj = Prism(atoms.get_cell(), reduce_cell=reduce_cell)
    xhi, yhi, zhi, xy, xz, yz = convert(
        prismobj.get_lammps_prism(), "distance", "ASE", units
    )
    fd.write(f"0.0 {xhi:23.17g}  xlo xhi\n")
    fd.write(f"0.0 {yhi:23.17g}  ylo yhi\n")
    fd.write(f"0.0 {zhi:23.17g}  zlo zhi\n")
    if force_skew or prismobj.is_skewed():
        fd.write(f"{xy:23.17g} {xz:23.17g} {yz:23.17g}  xy xz yz\n")
    fd.write("\n")

    if masses:
        _write_masses(fd, atoms, species, units)

    # -------- Atoms section ------------
    fd.write(f"Atoms # {atom_style}\n\n")
    pos = prismobj.vector_to_lammps(atoms.get_positions(), wrap=write_image_flags)
    if atom_style == "atomic":
        pos = convert(pos, "distance", "ASE", units)
        for i, r in enumerate(pos):
            s = species.index(symbols[i]) + 1
            fd.write(f"{i+1:>6} {s:>3} {r[0]:23.17g} {r[1]:23.17g} {r[2]:23.17g}\n")
    elif atom_style == "full":
        # If charges list is provided, override atomic charges
        if charges is not None:
            charges_array = np.array(
                [charges[species.index(symbol)] for symbol in symbols]
            )
        else:
            charges_array = atoms.get_initial_charges()
        charges_array = convert(charges_array, "charge", "ASE", units)

        molecules = np.ones(len(atoms), dtype=int)  # default molecule IDs
        pos = convert(pos, "distance", "ASE", units)
        for i, (m, q, r) in enumerate(zip(molecules, charges_array, pos, strict=False)):
            s = species.index(symbols[i]) + 1
            fd.write(
                f"{i+1:>6} {m:>3} {s:>3} {q:>9.6f} {r[0]:23.17g} {r[1]:23.17g} {r[2]:23.17g}\n"
            )

    # -------- Bonds section ------------
    if n_bonds > 0:
        fd.write("\nBonds\n\n")
        for idx, (bt, a1, a2) in enumerate(bonds_in, start=1):
            fd.write(f"{idx:>3} {bt:>3} {a1:>3} {a2:>3}\n")

    # -------- Angles section -----------
    if n_angles > 0:
        fd.write("\nAngles\n\n")
        for idx, (atype, cen, a1, a3) in enumerate(angles_in, start=1):
            fd.write(f"{idx:>3} {atype:>3} {a1:>3} {cen:>3} {a3:>3}\n")

    # -------- Dihedrals section --------
    if n_dihedrals > 0:
        fd.write("\nDihedrals\n\n")
        for idx, (dtype, a1, a2, a3, a4) in enumerate(dihedrals_in, start=1):
            fd.write(f"{idx:>3} {dtype:>3} {a1:>3} {a2:>3} {a3:>3} {a4:>3}\n")

    # -------- Impropers section --------
    if n_impropers > 0:
        fd.write("\nImpropers\n\n")
        for idx, (itype, a1, a2, a3, a4) in enumerate(impropers_in, start=1):
            fd.write(f"{idx:>3} {itype:>3} {a1:>3} {a2:>3} {a3:>3} {a4:>3}\n")

    fd.flush()
