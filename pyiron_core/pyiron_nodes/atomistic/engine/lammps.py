from __future__ import annotations

from dataclasses import asdict
from typing import Optional

# if TYPE_CHECKING:
from ase import Atoms
from pandas import DataFrame
from pyiron_atomistics.lammps.control import LammpsControl

from pyiron_core.pyiron_nodes.atomistic.calculator.data import (
    # InputCalcMinimize,
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

    # print('calculator_input: ', calculator_input.run())
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

    # print('calculator_input: ', calculator_input.run())
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

    # print('calculator_input: ', calculator_input.run())
    calculator_kwargs = asdict(calculator_input)
    # print('kwargs: ', calculator_kwargs)
    # calculator_kwargs = parse_input_kwargs(calculator_input, InputCalcMD)
    calculator = LammpsControl()
    calculator.calc_md(**calculator_kwargs)
    calculator.mode = "md"
    # use absolute coordinates rather than relative ones
    calculator["dump___1"] = calculator["dump___1"].replace("su", "u")

    return calculator


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

    # print('structure: ', structure)

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
    print("Potential: ", pot.df)
    pot.write_file(file_name="potential.inp", cwd=working_directory)
    pot.copy_pot_files(working_directory)

    with open(os.path.join(working_directory, "structure.inp"), "w") as f:
        structure.write(f, format="lammps-data", specorder=pot.get_element_lst())

    calculator.write_file(file_name="control.inp", cwd=working_directory)

    return os.path.abspath(working_directory)


@as_function_node("log")
def ParseLogFile(log_file):
    from pymatgen.io.lammps.outputs import parse_lammps_log

    # print(f"parse log file:  {type(log_file)})")
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


# @wf_data_class()
@as_out_dataclass_node
class ShellOutput:
    stdout: str = ""
    stderr: str = ""
    return_code: int = 0
    dump: FileObject = FileObject()  # TODO: should be done in a specific lammps object
    log: FileObject = FileObject()


@as_function_node("output", "dump", "log")
def Shell(
    working_directory: str,
    command: str = "lmp",
    environment: Optional[dict] = None,
    arguments: Optional[list] = None,
):
    arguments = ["-in", "control.inp"] if arguments is None else arguments
    # -> (ShellOutput, FileObject, FileObject):  TODO: fails -> why
    import os
    import subprocess

    if environment is None:
        environment = {}
    if arguments is None:
        arguments = []

    environ = dict(os.environ)
    environ.update({k: str(v) for k, v in environment.items()})
    # print ([str(command), *map(str, arguments)], working_directory, environment)
    # print("start shell")
    proc = subprocess.run(
        [command, *map(str, arguments)],
        capture_output=True,
        cwd=working_directory,
        encoding="utf8",
        env=environ,
    )
    # print("end shell")

    output = ShellOutput()
    output.stdout = proc.stdout
    output.stderr = proc.stderr
    output.return_code = proc.returncode
    dump = FileObject("dump.out", working_directory)
    log = FileObject("log.lammps", working_directory)

    return output, dump, log


# @wf_data_class()
@as_out_dataclass_node
class GenericOutput:
    energy_pot = []
    energy_kin = []
    forces = []


@as_function_node
def Collect(
    out_dump,
    out_log,
    calc_mode,  # : str | LammpsControl | InputCalcMinimizeLammps | InputCalcMD | InputCalcStatic,
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
        # print("output Collect: ", generic, isinstance(generic, OutputCalcStatic))
        # if isinstance(generic, OutputCalcStatic):
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

        # generic.energies_kin = log["TotEng"].values - generic.energies_pot
        generic.forces = np.array([o.data[["fx", "fy", "fz"]] for o in out_dump])
        generic.positions = np.array([o.data[["xu", "yu", "zu"]] for o in out_dump])

    return generic


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
    # print("energies_pot: ", generic.energies_pot)
    return generic.energies_pot[i_start:i_end]


@as_macro_node("generic")
def Code(
    structure: Atoms,
    calculator,  # =InputCalcStatic(),  # TODO: Don't use mutable defaults
    potential: Optional[str] = None,
    working_dir: str = "test2",
):

    from pyiron_core.pyiron_workflow import Workflow

    # print("structure: ", structure)

    wf = Workflow("LammpsMacro")

    # wf.Potential = Potential(structure=structure, name=potential)

    # wf.ListPotentials = ListPotentials(structure=structure)

    # wf.calc = CalcMD(calculator)

    wf.InitLammps = InitLammps(
        structure=structure,
        potential=potential,  # wf.Potential,
        # calculator=wf.calc,
        calculator=calculator,
        working_directory=working_dir,
    )

    wf.Shell = Shell(
        # command=ExecutablePathResolver(module="lammps", code="lammps").path(),
        working_directory=wf.InitLammps,
    )

    wf.ParseLogFile = ParseLogFile(log_file=wf.Shell.outputs.log)
    wf.ParseDumpFile = ParseDumpFile(dump_file=wf.Shell.outputs.dump)
    wf.Collect = Collect(
        out_dump=wf.ParseDumpFile.outputs.dump,
        out_log=wf.ParseLogFile.outputs.log,
        calc_mode="md",  # wf.calc,
    )

    return wf.Collect


@as_function_node
def DummyNode(structure1: Atoms, structure2: Atoms):
    return structure1


@as_macro_node(labels=["generic", "path"])
def Code1(
    structure: Atoms,
    calculator,  # =InputCalcStatic(),  # TODO: Don't use mutable defaults
    potential: Optional[str] = None,
    working_dir: str = "test2",
):

    from pyiron_core.pyiron_workflow import Workflow

    wf = Workflow("LammpsMacro")

    wf.Potential = Potential(structure=structure, name=potential)
    # wf.DummyNode = DummyNode(structure1=structure, structure2=structure)

    wf.ListPotentials = ListPotentials(structure=structure)

    # wf.calc = CalcMD(calculator)

    wf.InitLammps = InitLammps(
        structure=structure,
        potential=wf.Potential,
        # calculator=wf.calc,
        calculator=calculator,
        working_directory=working_dir,
    )

    wf.Shell = Shell(
        # command=ExecutablePathResolver(module="lammps", code="lammps").path(),
        working_directory=wf.InitLammps,
    )

    wf.ParseLogFile = ParseLogFile(log_file=wf.Shell.outputs.log)
    wf.ParseDumpFile = ParseDumpFile(dump_file=wf.Shell.outputs.dump)
    wf.Collect = Collect(
        out_dump=wf.ParseDumpFile.outputs.dump,
        out_log=wf.ParseLogFile.outputs.log,
        calc_mode="md",  # wf.calc,
    )

    return wf.Collect, wf.InitLammps.outputs.path


@as_macro_node("generic")
def Lammps(
    structure: Atoms,
    calculator,  # =InputCalcStatic(),  # TODO: Don't use mutable defaults
    potential: Optional[str] = None,
    working_dir: str = "test2",
    store: bool = False,
):

    from pyiron_core.pyiron_workflow import Workflow

    wf = Workflow("LammpsMacro")

    wf.Potential = Potential(structure=structure, name=potential)
    # wf.DummyNode = DummyNode(structure1=structure, structure2=structure)

    wf.ListPotentials = ListPotentials(structure=structure)

    # wf.calc = CalcMD(calculator)

    wf.InitLammps = InitLammps(
        structure=structure,
        potential=wf.Potential,
        # calculator=wf.calc,
        calculator=calculator,
        working_directory=working_dir,
    )

    wf.Shell = Shell(
        # command=ExecutablePathResolver(module="lammps", code="lammps").path(),
        working_directory=wf.InitLammps,
    )

    wf.ParseLogFile = ParseLogFile(log_file=wf.Shell.outputs.log)
    wf.ParseDumpFile = ParseDumpFile(dump_file=wf.Shell.outputs.dump)
    wf.Collect = Collect(
        out_dump=wf.ParseDumpFile.outputs.dump,
        out_log=wf.ParseLogFile.outputs.log,
        calc_mode="md",  # calculator,
    )

    return wf.Collect
