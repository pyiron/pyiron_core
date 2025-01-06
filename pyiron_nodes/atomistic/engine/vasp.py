from __future__ import annotations

import os
import warnings
from pathlib import Path
import shutil
import subprocess
from typing import Optional
from dataclasses import dataclass, field

import pandas as pd

from pymatgen.core import Structure
from pymatgen.io.vasp.inputs import Incar, Kpoints
from pymatgen.io.vasp.outputs import Vasprun

from ase import Atoms

from pyiron_workflow import Workflow
from pyiron_nodes.dev_tools import VarType, FileObject
from pyiron_atomistics.vasp.output import parse_vasp_output as pvo

from pyiron_snippets.logger import logger

# from pyiron_snippets.resources import ResourceResolver

from pyiron_nodes.atomistic.engine.lammps import Shell


class Storage:
    def _convert_to_dict(instance):
        # Get the attributes of the instance
        attributes = vars(instance)

        # Convert attributes to a dictionary
        result_dict = {
            key: value for key, value in attributes.items() if "_" not in key[0]
        }

        return result_dict


class ShellOutput(Storage):
    stdout: str
    stderr: str
    return_code: int
    dump: FileObject  # TODO: should be done in a specific lammps object
    log: FileObject


def read_potcar_config(config_file: Path) -> dict:
    """
    Reads the POTCAR configuration from a file and resolves the paths dynamically based on config content.

    Args:
        config_file (Path): Path to the configuration file.

    Returns:
        dict: A dictionary containing the resolved paths and the default POTCAR path.

    Raises:
        ValueError:
            - If no valid `default_POTCAR_set` is found in the config file.
            - If no valid `default_functional` (e.g., PBE or LDA) is found in the config.
        FileNotFoundError: If the configuration file does not exist.
        Exception: For any other unexpected issues encountered while reading the file.
    """

    config_data = {}

    # Read the configuration file
    try:
        with open(config_file, "r") as f:
            for line in f:
                # Skip comments and empty lines
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                # Split the line into key and value
                if "=" in line:
                    key, value = line.split("=", 1)
                    key = key.strip()
                    value = value.strip()

                    # Store the configuration
                    config_data[key] = value

        # Resolve the pyiron_vasp_resources path
        pyiron_vasp_resources = config_data.get("pyiron_vasp_resources", "")
        default_POTCAR_set = config_data.get("default_POTCAR_set")
        default_functional = config_data.get("default_functional")

        # Dynamically identify all POTCAR sets based on keys in the config file
        potcar_sets = []
        for key in config_data:
            if key.startswith("vasp_POTCAR_path_"):
                potcar_sets.append(key.split("vasp_POTCAR_path_")[1])

        # Check if a valid default_POTCAR_set is provided
        if not default_POTCAR_set or default_POTCAR_set not in potcar_sets:
            raise ValueError(
                f"Unknown or missing default_POTCAR_set: {default_POTCAR_set}. Valid options: {potcar_sets}"
            )

        # Check if a valid default_functional is provided
        if not default_functional or default_functional not in ["PBE", "LDA"]:
            raise ValueError(
                f"Unknown or missing default_functional: {default_functional}. Valid options: PBE, LDA"
            )

        # Dynamically generate the paths for all potpaw sets based on the config content
        for potcar_set in potcar_sets:
            key = f"vasp_POTCAR_path_{potcar_set}"
            # Preserve the underscores in the path
            config_data[key] = os.path.join(
                pyiron_vasp_resources, potcar_set.replace("potpaw", "potpaw_")
            )

        # Set the default POTCAR path
        config_data["default_POTCAR_path"] = config_data[
            f"vasp_POTCAR_path_{default_POTCAR_set}"
        ]

        return config_data

    except FileNotFoundError as e:
        raise FileNotFoundError(f"Configuration file not found: {config_file}") from e
    except ValueError as e:
        raise e  # Let specific ValueErrors propagate
    except Exception as e:
        raise Exception(f"Error reading configuration file: {e}") from e


# Look in user's home dir
config_file = os.path.join(Path.home(), ".pyiron_vasp_config")
potcar_config = read_potcar_config(config_file)
default_POTCAR_library_path = potcar_config["default_POTCAR_path"]
default_POTCAR_generation_path = os.path.join(
    potcar_config["default_POTCAR_path"], potcar_config["default_functional"]
)
default_functional = potcar_config["default_functional"]
POTCAR_default_specification_data = str(
    Path(__file__).parent.joinpath(
        "vasp_resources", f"vasp_pseudopotential_{default_functional}_data.csv"
    )
)


@dataclass
class VaspInput:
    """
    Class to represent the input settings for a VASP calculation, including structure, INCAR parameters, and pseudopotential paths.

    Attributes:
        structure (Structure): The atomic structure of the system to be used in the VASP calculation.
        incar (Incar): The INCAR object containing the VASP input parameters.
        pseudopot_lib_path (str): The path to the library of VASP pseudopotentials (POTCAR files).
        Defaults to `default_POTCAR_generation_path` generated by the .pyiron_vasp_config.
        potcar_paths (Optional[list[str]]): A list of paths to user-specified POTCAR files associated with the calculation.
            When specified, this _OVERRIDES_ default POTCAR generation.
            This list must contain all the necessary POTCAR files for the elements in the structure.
            For example, if the structure contains H O H atoms, the list should contain paths like:
            - ".../vasp/potpaw_64/H/POTCAR"
            - ".../vasp/custom_charged_potential/O/POTCAR"
            - ".../vasp/potpaw_64/H/POTCAR"
            No checks are performed to ensure the POTCAR files match the species in the structure, so it is the user's responsibility
            to ensure consistency between the elements in the structure and the provided POTCAR files when using this functionality.

        kpoints (Optional[Kpoints]): The KPOINTS object defining the k-point mesh for the calculation. If not provided, default settings will be used.

    Notes:
        - The `potcar_paths` attribute is exposed to allow users to make one-off changes to the pseudopotentials. This is
          useful when custom/non-default potentials are required for specific elements.
        - When potcar_paths is used, the class does not validate that the provided POTCAR files match the elements in the structure,
          so incorrect configurations may lead to invalid calculations.
    """

    structure: Structure
    incar: Incar
    pseudopot_lib_path: str = field(default=default_POTCAR_library_path)
    pseudopot_functional: str = "PBE"
    potcar_paths: Optional[list[str]] = None
    kpoints: Optional[Kpoints] = None


@Workflow.wrap.as_function_node("line_found")
def isLineInFile(filepath: str, line: str, exact_match: bool = True) -> bool:
    line_found = False  # Initialize the result as False
    try:
        with open(filepath, "r") as file:
            for file_line in file:
                if exact_match and line == file_line.strip():
                    line_found = True
                    break  # Exit loop if the line is found
                elif not exact_match and line in file_line:
                    line_found = True
                    break  # Exit loop if a partial match is found
    except FileNotFoundError:
        logger.info(f"File '{filepath}' not found.")
    return line_found


def write_POSCAR(workdir: str, structure: Structure, filename: str = "POSCAR") -> str:
    poscar_path = os.path.join(workdir, filename)
    structure.to(fmt="poscar", filename=poscar_path)
    return poscar_path


def write_INCAR(workdir: str, incar: Incar, filename: str = "INCAR") -> str:
    incar_path = os.path.join(workdir, filename)
    incar.write_file(incar_path)
    return incar_path


def write_POTCAR(workdir: str, vasp_input: VaspInput, filename: str = "POTCAR") -> str:
    class PotcarNotGeneratedError(Exception):
        pass

    if vasp_input.potcar_paths is None:
        potcar_paths = get_default_POTCAR_paths(
            vasp_input.structure, vasp_input.pseudopot_lib_path
        )
    else:
        potcar_paths = vasp_input.potcar_paths

    potcar_path = os.path.join(workdir, filename)

    with open(potcar_path, "wb") as wfd:
        for f in potcar_paths:
            with open(f, "rb") as fd:
                shutil.copyfileobj(fd, wfd)

    return potcar_path


def write_KPOINTS(
    workdir: str, kpoints: Optional[Kpoints] = None, filename: str = "KPOINTS"
) -> str:
    kpoint_path = os.path.join(workdir, filename)
    if kpoints is not None:
        kpoints.write_file(kpoint_path)
    else:
        with open(kpoint_path, "w") as f:
            f.write(
                "Automatic mesh\n0\nGamma\n1 1 1\n0 0 0\n"
            )  # Example KPOINTS content
    return kpoint_path


@Workflow.wrap.as_function_node("workdir")
def create_WorkingDirectory(workdir: str, quiet: bool = False) -> str:
    # Check if workdir exists
    if not os.path.exists(workdir):
        os.makedirs(workdir)
        logger.info(f"made directory '{workdir}'")
    else:
        warnings.warn(
            f"Directory '{workdir}' already exists. Existing files may be overwritten."
        )
    return workdir


@Workflow.wrap.as_function_node("workdir")
def write_VaspInputSet(workdir: str, vasp_input: VaspInput) -> str:
    _ = write_POSCAR(workdir=workdir, structure=vasp_input.structure)
    _ = write_INCAR(workdir=workdir, incar=vasp_input.incar)
    _ = write_POTCAR(workdir=workdir, vasp_input=vasp_input)
    if vasp_input.kpoints is not None:
        _ = write_KPOINTS(workdir=workdir, kpoints=vasp_input.kpoints)

    return workdir


@Workflow.wrap.as_function_node("output")
def run_job(
    command: str,
    workdir: str | None = None,
    environment: Optional[dict[str, str]] = None,
    arguments: Optional[list[str]] = None,
) -> ShellOutput:
    if environment is None:
        environment = {}
    if arguments is None:
        arguments = []
    logger.info(f"run_job is in {os.getcwd()}")
    environ = dict(os.environ)
    environ.update({k: str(v) for k, v in environment.items()})
    proc = subprocess.run(
        [command, *map(str, arguments)],
        capture_output=True,
        cwd=workdir,
        encoding="utf8",
        env=environ,
        shell=True,
    )
    output = ShellOutput()
    output.stdout = proc.stdout
    output.stderr = proc.stderr
    output.return_code = proc.returncode
    return output


@Workflow.wrap.as_function_node("output_dict")
def parse_VaspOutput(workdir: str) -> dict:
    logger.info(f"workdir of parse: {workdir}")
    return pvo(workdir)


@Workflow.wrap.as_function_node("convergence")
def check_convergence(
    workdir: str,
    filename_vasprun: str = "vasprun.xml",
    filename_vasplog: str = "vasp.log",
    backup_vasplog: str = "error.out",
) -> bool:
    converged = False
    line_converged = (
        "reached required accuracy - stopping structural energy minimisation"
    )

    try:
        vr = Vasprun(filename=os.path.join(workdir, filename_vasprun))
        converged = vr.converged
    except:
        try:
            converged = isLineInFile.node_function(
                filepath=os.path.join(workdir, filename_vasplog),
                line=line_converged,
                exact_match=False,
            )
        except:
            try:
                converged = isLineInFile.node_function(
                    filepath=os.path.join(workdir, backup_vasplog),
                    line=line_converged,
                    exact_match=False,
                )
            except:
                pass

    return converged


@Workflow.wrap.as_macro_node("vasp_output", "convergence_status")
def vasp_job(
    self,
    workdir: str,
    vasp_input: VaspInput,
    command: str = "module load vasp; module load intel/19.1.0 impi/2019.6; unset I_MPI_HYDRA_BOOTSTRAP; unset I_MPI_PMI_LIBRARY; mpiexec -n 1 vasp_std",
):
    self.working_dir = create_WorkingDirectory(workdir=workdir)
    self.vaspwriter = write_VaspInputSet(workdir=workdir, vasp_input=vasp_input)
    self.job = run_job(command=command, workdir=workdir)
    self.vasp_output = parse_VaspOutput(workdir=workdir)
    self.convergence_status = check_convergence(workdir=workdir)

    (
        self.working_dir
        >> self.vaspwriter
        >> self.job
        >> self.vasp_output
        >> self.convergence_status
    )
    self.starting_nodes = [self.working_dir]
    return self.vasp_output, self.convergence_status


def stack_element_string(structure) -> tuple[list[str], list[int]]:
    # site_element_list = [atom.symbol for atom in atoms]
    site_element_list = [site.species_string for site in structure]
    past_element = site_element_list[0]
    element_list = [past_element]
    element_count = []
    count = 0

    for element in site_element_list:
        if element == past_element:
            count += 1
        else:
            element_count.append(count)
            element_list.append(element)
            count = 1
            past_element = element

    element_count.append(count)
    return element_list, element_count


def get_default_POTCAR_paths(
    structure: Structure,
    pseudopot_lib_path: str,
    potcar_df: pd.DataFrame = pd.read_csv(POTCAR_default_specification_data),
) -> list[str]:
    ele_list, _ = stack_element_string(structure)
    potcar_paths = []
    for element in ele_list:
        ele_default_potcar_path = potcar_df[
            (potcar_df["symbol"] == element) & (potcar_df["default"] == True)
        ].potential_name.values[0]
        potcar_paths.append(
            os.path.join(pseudopot_lib_path, ele_default_potcar_path, "POTCAR")
        )

    return potcar_paths
