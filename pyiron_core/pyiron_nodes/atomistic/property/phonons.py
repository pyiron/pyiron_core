# from __future__ import annotations

from dataclasses import asdict
from typing import Optional
import numpy as np

from ase import Atoms
from phonopy.api_phonopy import Phonopy
from pyiron_core.pyiron_workflow import (
    as_inp_dataclass_node,
    as_out_dataclass_node,
    as_function_node,
    as_macro_node,
)
from structuretoolkit.common import atoms_to_phonopy, phonopy_to_atoms

from pyiron_core.pyiron_nodes.atomistic.calculator.ase import Static
from pyiron_core.pyiron_nodes.atomistic.engine.generic import OutputEngine


@as_function_node("phonopy")
def PhonopyObject(structure):
    return Phonopy(unitcell=atoms_to_phonopy(structure))


@as_inp_dataclass_node
class PhonopyParameters:
    distance: float = 0.01
    is_plusminus: str | bool = "auto"
    is_diagonal: bool = True
    is_trigonal: bool = False
    number_of_snapshots: Optional[int] = None
    random_seed: Optional[int] = None
    temperature: Optional[float] = None
    cutoff_frequency: Optional[float] = None
    max_distance: Optional[float] = None


@as_function_node
def GenerateSupercells(phonopy: Phonopy, parameters: PhonopyParameters) -> list[Atoms]:
    from dataclasses import asdict

    parameters = PhonopyParameters() if parameters is None else parameters
    phonopy.generate_displacements(**asdict(parameters))

    supercells = [phonopy_to_atoms(s) for s in phonopy.supercells_with_displacements]
    return supercells


@as_macro_node("phonopy", "thermal_properties")
def phonopy(
    structure: Atoms,
    engine: OutputEngine,
    phonopy_parameters: PhonopyParameters = None,
    GetThermalProperties__mesh="10",
):

    from pyiron_core.pyiron_workflow import Workflow
    from pyiron_core.pyiron_nodes.atomistic.calculator.ase import Static
    from pyiron_core.pyiron_nodes.atomistic.property.phonons import PhonopyObject
    from pyiron_core.pyiron_nodes.atomistic.property.phonons import GenerateSupercells
    from pyiron_core.pyiron_nodes.controls import iterate
    from pyiron_core.pyiron_nodes.controls import GetAttribute
    from pyiron_core.pyiron_nodes.controls import SetAttribute
    from pyiron_core.pyiron_nodes.atomistic.property.phonons import GetDynamicalMatrix
    from pyiron_core.pyiron_nodes.atomistic.property.phonons import GetThermalProperties

    if phonopy_parameters is None:
        phonopy_parameters = PhonopyParameters()

    wf = Workflow(phonopy)

    wf.PhonopyObject = PhonopyObject(structure=structure)
    wf.GenerateSupercells = GenerateSupercells(
        phonopy=wf.PhonopyObject, parameters=phonopy_parameters
    )
    wf.Static = Static(engine=engine, structure=structure)
    wf.iterate = iterate(
        values=wf.GenerateSupercells, node=wf.Static, input_label="structure"
    )
    wf.GetForces = GetAttribute(obj=wf.iterate, attr="forces")
    wf.SetForces = SetAttribute(val=wf.GetForces, obj=wf.PhonopyObject, attr="forces")
    wf.GetDynamicalMatrix = GetDynamicalMatrix(phonopy=wf.SetForces)
    wf.GetThermalProperties = GetThermalProperties(
        phonopy=wf.GetDynamicalMatrix.outputs.phonopy, mesh=GetThermalProperties__mesh
    )

    return (
        wf.PhonopyObject,
        wf.GetThermalProperties,
    )


@as_function_node
def GetFreeEnergy(
    structure: Atoms,
    engine: OutputEngine,
    temperature: float = 300,
    mesh: int = 10,
    parameters: Optional[PhonopyParameters] = None,
):
    """
    Calculate the free energy of a structure in the Harmonic approximation
    at a given temperature using phonopy.
    """
    from dataclasses import asdict

    print(f"Calculating free energy at {temperature} K")

    phonopy = Phonopy(unitcell=atoms_to_phonopy(structure))

    parameters = PhonopyParameters().run() if parameters is None else parameters
    phonopy.generate_displacements(**asdict(parameters))

    supercells = [phonopy_to_atoms(s) for s in phonopy.supercells_with_displacements]
    forces = []
    for sc in supercells:
        sc.calc = engine.calculator
        forces.append(sc.get_forces())
    phonopy.forces = forces
    phonopy.produce_force_constants()
    phonopy.dynamical_matrix.run(q=[0, 0, 0])

    phonopy.run_mesh([mesh, mesh, mesh])
    phonopy.run_thermal_properties(
        t_min=temperature,
        t_max=temperature,
        t_step=1,
    )

    print(
        f"Free energy calculated for temperature: {temperature} K: ",
        phonopy.get_thermal_properties_dict(),
    )
    free_energy = phonopy.get_thermal_properties_dict()["free_energy"][-1]
    print(f"Free energy: {free_energy} eV")

    return free_energy


@as_function_node("forces")
def ExtractFinalForces(df):
    return [getattr(e, "force")[-1] for e in df["out"].tolist()]


@as_function_node
def GetDynamicalMatrix(phonopy, q=None):
    import numpy as np

    q = [0, 0, 0] if q is None else q
    if phonopy.dynamical_matrix is None:
        phonopy.produce_force_constants()
        phonopy.dynamical_matrix.run(q=q)
    dynamical_matrix = np.real_if_close(phonopy.dynamical_matrix.dynamical_matrix)
    # print (dynamical_matrix)
    return dynamical_matrix, phonopy


@as_function_node
def GetEigenvalues(matrix):
    import numpy as np

    ew = np.linalg.eigvalsh(matrix)
    return ew


@as_macro_node
def CheckConsistency(self, phonopy: Phonopy, tolerance: float = 1e-10):
    self.dyn_matrix = GetDynamicalMatrix(phonopy).run()
    self.ew = GetEigenvalues(self.dyn_matrix)
    self.has_imaginary_modes = HasImaginaryModes(self.ew, tolerance)
    return self.has_imaginary_modes


@as_function_node
def GetTotalDos(
    phonopy,
    mesh: int = None,
    store: bool = False,
):
    from pandas import DataFrame

    mesh = 3 * [10] if mesh is None else mesh

    phonopy.produce_force_constants()
    phonopy.run_mesh(mesh=mesh)
    phonopy.run_total_dos()
    total_dos = DataFrame(phonopy.get_total_dos_dict())
    return total_dos


@as_function_node
def HasImaginaryModes(eigenvalues, tolerance: float = 1e-10) -> bool:
    ew_lt_zero = eigenvalues[eigenvalues < -tolerance]
    if len(ew_lt_zero) > 0:
        print(f"WARNING: {len(ew_lt_zero)} imaginary modes exist")
        has_imaginary_modes = True
    else:
        has_imaginary_modes = False
    return has_imaginary_modes


@as_out_dataclass_node
class ThermalProperties:
    from pyiron_core.pyiron_workflow.data_fields import DataArray, EmptyArrayField

    temperatures: DataArray = EmptyArrayField()
    free_energy: DataArray = EmptyArrayField()
    entropy: DataArray = EmptyArrayField()
    heat_capacity: DataArray = EmptyArrayField()


@as_function_node
def GetThermalProperties(
    phonopy: Phonopy,
    t_min: float = 0,
    t_max: float = 1000,
    t_step: int = 10,
    mesh=20,
    store: bool = False,
):
    """Get thermal properties from phonopy."""
    from pint import UnitRegistry

    ureg = UnitRegistry()
    _, phonopy_new = GetDynamicalMatrix(phonopy).run()
    phonopy_new.run_mesh([mesh, mesh, mesh])
    phonopy_new.run_thermal_properties(
        t_min=t_min,
        t_max=t_max,
        t_step=t_step,
    )

    tp_dict = phonopy_new.get_thermal_properties_dict()
    # print(f"Thermal properties calculated for temperatures: {tp_dict['temperatures']}")
    # Convert the dictionary to a ThermalProperties dataclass
    thermal_properties = ThermalProperties().dataclass(**tp_dict)
    thermal_properties.free_energy *= (
        (1 * ureg.kilojoule / ureg.avogadro_number).to("eV").magnitude
    )

    return thermal_properties


@as_function_node
def GetAnalyticalFreeEnergy(
    nu, dos, temperatures, n_atoms: int = None, quantum: bool = True
):
    from scipy.integrate import simpson
    from scipy.constants import physical_constants

    KB = physical_constants["Boltzmann constant in eV/K"][0]
    H = physical_constants["Planck constant in eV/Hz"][0]

    sel = nu > 0.0
    nu_sel = nu[sel] * 1e12
    dos_sel = dos[sel]
    dos_sel /= simpson(y=dos_sel, x=nu_sel)
    u, fe = [], []
    for temp in temperatures:
        if not quantum:
            u.append(3 * n_atoms * KB * temp)
            fe.append(
                3
                * n_atoms
                * simpson(
                    y=KB * temp * dos_sel * np.log(H * nu_sel / (KB * temp)), x=nu_sel
                )
            )
        else:
            u.append(
                3
                * n_atoms
                * simpson(
                    y=H
                    * dos_sel
                    * nu_sel
                    * (0.5 + (1 / (np.exp(H * nu_sel / (KB * temp)) - 1))),
                    x=nu_sel,
                )
            )
            fe.append(
                3
                * n_atoms
                * simpson(
                    y=(
                        (H * dos_sel * nu_sel * 0.5)
                        + (
                            KB
                            * temp
                            * dos_sel
                            * np.log(1 - np.exp(-H * nu_sel / (KB * temp)))
                        )
                    ),
                    x=nu_sel,
                )
            )
    return u, fe
