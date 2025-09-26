# from __future__ import annotations
from dataclasses import field
from typing import Optional

import atomistics.workflows.elastic.symmetry as sym
import numpy as np

from pyiron_core.pyiron_workflow import (
    Node,
    as_function_node,
    as_inp_dataclass_node,
    as_macro_node,
    # for_node,
    # standard_nodes as standard,
    as_out_dataclass_node,
)

# from pyiron_core.pyiron_nodes.dev_tools import wf_data_class
# from pyiron_core.pyiron_nodes.development.settings import Storage
# from pyiron_core.pyiron_workflow import as_dataclass_node
# from pyiron_core.pyiron_nodes.development.node_dataclass import as_output_node


@as_out_dataclass_node
class OutputElasticSymmetryAnalysis:
    Lag_strain_list: list = field(default_factory=lambda: [])
    epss: np.ndarray = field(default_factory=lambda: np.zeros(0))
    SGN: int = 0
    v0: float = 0.0
    LC: int = 1


@as_inp_dataclass_node
class InputElasticTensor:
    num_of_point: int = 5
    eps_range: float = 0.005
    sqrt_eta: bool = True
    fit_order: int = 2


@as_out_dataclass_node
class DataStructureContainer:
    structure: list = field(default_factory=lambda: [])
    job_name: list = field(default_factory=lambda: [])
    energy: list = field(default_factory=lambda: [])
    forces: list = field(default_factory=lambda: [])
    stress: list = field(default_factory=lambda: [])


@as_out_dataclass_node
class OutputElasticAnalysis:
    # from pyiron_core.pyiron_nodes.development.hash_based_storage import str_to_dict
    strain_energy: list = field(default_factory=lambda: [])
    C: np.ndarray = field(default_factory=lambda: np.zeros(0))
    A2: list = field(default_factory=lambda: [])
    C_eigval: np.ndarray = field(default_factory=lambda: np.zeros(0))
    C_eigvec: np.ndarray = field(default_factory=lambda: np.zeros(0))

    BV: int | float = 0
    GV: int | float = 0
    EV: int | float = 0
    nuV: int | float = 0
    S: int | float = 0
    BR: int | float = 0
    GR: int | float = 0
    ER: int | float = 0
    nuR: int | float = 0
    BH: int | float = 0
    GH: int | float = 0
    EH: int | float = 0
    nuH: int | float = 0
    AVR: int | float = 0
    energy_0: float = 0

    # _serialize: callable = str_to_dict  # provide optional function for serialization
    _skip_default_values = False


@as_function_node  # ("structure_container")
def AddEnergies(
    structure_container: DataStructureContainer,
    engine: Node,
) -> DataStructureContainer:
    for structure in structure_container.structure:
        engine.inputs.structure = structure
        out = engine.run()
        structure_container.energy.append(out.energies_pot[-1])

    return structure_container


@as_function_node("forces")
def ExtractFinalEnergy(df):
    # Looks an awful lot like phonons.ExtractFinalForce -- room for abstraction here
    return [e.energy[-1] for e in df["out"].tolist()]


@as_function_node
def SymmetryAnalysis(
    structure, parameters: Optional[InputElasticTensor] = None
) -> OutputElasticSymmetryAnalysis:
    parameters = InputElasticTensor() if parameters is None else parameters
    out = OutputElasticSymmetryAnalysis().dataclass()  # structure)

    out.SGN = sym.find_symmetry_group_number(structure)
    out.v0 = structure.get_volume()
    out.LC = sym.get_symmetry_family_from_SGN(out.SGN)
    out.Lag_strain_list = sym.get_LAG_Strain_List(out.LC)

    out.epss = np.linspace(
        -parameters.eps_range, parameters.eps_range, parameters.num_of_point
    )
    return out


@as_function_node("structures")
def GenerateStructures(
    structure,
    analysis: OutputElasticSymmetryAnalysis,
    parameters: Optional[InputElasticTensor] = None,
):
    structure_dict = {}
    structures = []

    zero_strain_job_name = "s_e_0"
    if 0.0 in analysis.epss:
        structure_dict[zero_strain_job_name] = structure.copy()
        structures.append(structure.copy())

    for lag_strain in analysis.Lag_strain_list:
        Ls_list = sym.Ls_Dic[lag_strain]
        for eps in analysis.epss:
            if eps == 0.0:
                continue

            Ls = np.zeros(6)
            for ii in range(6):
                Ls[ii] = Ls_list[ii]
            Lv = eps * Ls

            eta_matrix = np.zeros((3, 3))

            eta_matrix[0, 0] = Lv[0]
            eta_matrix[0, 1] = Lv[5] / 2.0
            eta_matrix[0, 2] = Lv[4] / 2.0

            eta_matrix[1, 0] = Lv[5] / 2.0
            eta_matrix[1, 1] = Lv[1]
            eta_matrix[1, 2] = Lv[3] / 2.0

            eta_matrix[2, 0] = Lv[4] / 2.0
            eta_matrix[2, 1] = Lv[3] / 2.0
            eta_matrix[2, 2] = Lv[2]

            norm = 1.0
            eps_matrix = eta_matrix
            if np.linalg.norm(eta_matrix) > 0.7:
                raise Exception(f"Too large deformation {eps}")

            if parameters.sqrt_eta:
                while norm > 1.0e-10:
                    x = eta_matrix - np.dot(eps_matrix, eps_matrix) / 2.0
                    norm = np.linalg.norm(x - eps_matrix)
                    eps_matrix = x

            # --- Calculating the M_new matrix ---
            i_matrix = np.eye(3)
            def_matrix = i_matrix + eps_matrix
            scell = np.dot(structure.get_cell(), def_matrix)
            struct = structure.copy()
            struct.set_cell(scell, scale_atoms=True)

            jobname = subjob_name(lag_strain, eps)

            structures.append(struct.copy())
            structure_dict[jobname] = struct
    job_names = list(structure_dict.keys())
    return structures, job_names


@as_macro_node("elastic_constants")
def ComputeElasticConstantsMacro(
    structure,
    engine,
    calculator: Node,
    input_elastic_tensor=None,
):
    """
    Get the elastic constants of a structure using an ASE calculator.
    """
    input_elastic_tensor = (
        InputElasticTensor() if input_elastic_tensor is None else input_elastic_tensor
    )
    from pyiron_core.pyiron_nodes.controls import Print, iterate
    from pyiron_core.pyiron_workflow import Workflow

    wf = Workflow("elastic_constants")

    wf.calculator = calculator
    # if calculator == "StaticEnergy":
    #     wf.calculator = StaticEnergy(structure=structure, engine=engine)
    # elif calculator == "GetFreeEnergy":
    #     wf.calculator = GetFreeEnergy(structure=structure, engine=engine)
    # else:
    #     test1, test2 = (calculator == "StaticEnergy"), (calculator == "GetFreeEnergy")
    #     raise ValueError(f"Unknown calculator: '{calculator}' {test1} {test2}")

    wf.print = Print(f"calculator: {calculator}")
    # wf.calculator = StaticEnergy(structure=structure, engine=engine)
    # wf.calculator = GetFreeEnergy(structure=structure, engine=engine)
    # print(f"Using calculator: {wf.calculator}")
    # print(f"Input calculator: {calculator}")
    wf.symmetry = SymmetryAnalysis(structure=structure, parameters=input_elastic_tensor)
    wf.structures = GenerateStructures(
        structure=structure, analysis=wf.symmetry, parameters=input_elastic_tensor
    )
    wf.energies = iterate(
        node=wf.calculator,
        values=wf.structures.outputs.structures,
        input_label="structure",
    )

    wf.elastic_constants = AnalyseStructures(
        energies=wf.energies,
        job_names=wf.structures.outputs.job_names,
        analysis=wf.symmetry,
        parameters=input_elastic_tensor,
    )
    return wf.elastic_constants


@as_function_node
def ComputeElasticConstants(
    structure,
    engine,
    calculator: str = "StaticEnergy",  # "StaticEnergy", "GetFreeEnergy"
    input_elastic_tensor: InputElasticTensor = None,
):
    from pyiron_core.pyiron_nodes.atomistic.calculator.ase import StaticEnergy
    from pyiron_core.pyiron_nodes.atomistic.property.phonons import GetFreeEnergy
    from pyiron_core.pyiron_nodes.controls import iterate
    from pyiron_core.pyiron_workflow import Workflow

    wf = Workflow("elastic_constants")
    if input_elastic_tensor is None:
        input_elastic_tensor = InputElasticTensor().run()

    if calculator == "StaticEnergy":
        wf.calculator = StaticEnergy(structure=structure, engine=engine)
    elif calculator == "GetFreeEnergy":
        wf.calculator = GetFreeEnergy(structure=structure, engine=engine)
    else:
        test1, test2 = (calculator == "StaticEnergy"), (calculator == "GetFreeEnergy")
        raise ValueError(f"Unknown calculator: '{calculator}' {test1} {test2}")

    wf.symmetry = SymmetryAnalysis(structure=structure, parameters=input_elastic_tensor)
    wf.structures = GenerateStructures(
        structure=structure, analysis=wf.symmetry, parameters=input_elastic_tensor
    )
    wf.energies = iterate(
        node=wf.calculator,
        values=wf.structures.outputs.structures,
        input_label="structure",
    )

    wf.elastic_constants = AnalyseStructures(
        energies=wf.energies,
        job_names=wf.structures.outputs.job_names,
        analysis=wf.symmetry,
        parameters=input_elastic_tensor,
    )

    elastic_constants = wf.elastic_constants.pull()
    # print("Energies:", wf.energies.outputs.out_lst)
    return elastic_constants


@as_function_node("structures")
def AnalyseStructures(
    # data_df: DataStructureContainer,
    energies,
    job_names,
    analysis: OutputElasticSymmetryAnalysis,
    parameters: Optional[InputElasticTensor] = None,
) -> OutputElasticAnalysis:
    zero_strain_job_name = "s_e_0"

    epss = analysis.epss
    Lag_strain_list = analysis.Lag_strain_list

    out = OutputElasticAnalysis().dataclass()
    energy_dict = dict(zip(job_names, energies, strict=True))

    if 0.0 in epss:
        out.energy_0 = energy_dict[zero_strain_job_name]

    strain_energy = []
    for lag_strain in Lag_strain_list:
        strain_energy.append([])
        for eps in epss:
            if not eps == 0.0:
                job_name = subjob_name(lag_strain, eps)
                ene = energy_dict[job_name]
            else:
                ene = out.energy_0
            strain_energy[-1].append((eps, ene))
    out.strain_energy = strain_energy
    out = fit_elastic_matrix(out, parameters.fit_order, v0=analysis.v0, LC=analysis.LC)
    return out


def calculate_modulus(out: OutputElasticAnalysis):
    C = out.C

    BV = (C[0, 0] + C[1, 1] + C[2, 2] + 2 * (C[0, 1] + C[0, 2] + C[1, 2])) / 9
    GV = (
        (C[0, 0] + C[1, 1] + C[2, 2])
        - (C[0, 1] + C[0, 2] + C[1, 2])
        + 3 * (C[3, 3] + C[4, 4] + C[5, 5])
    ) / 15
    EV = (9 * BV * GV) / (3 * BV + GV)
    nuV = (1.5 * BV - GV) / (3 * BV + GV)
    out.BV = BV
    out.GV = GV
    out.EV = EV
    out.nuV = nuV

    try:
        S = np.linalg.inv(C)

        BR = 1 / (S[0, 0] + S[1, 1] + S[2, 2] + 2 * (S[0, 1] + S[0, 2] + S[1, 2]))
        GR = 15 / (
            4 * (S[0, 0] + S[1, 1] + S[2, 2])
            - 4 * (S[0, 1] + S[0, 2] + S[1, 2])
            + 3 * (S[3, 3] + S[4, 4] + S[5, 5])
        )
        ER = (9 * BR * GR) / (3 * BR + GR)
        nuR = (1.5 * BR - GR) / (3 * BR + GR)

        BH = 0.50 * (BV + BR)
        GH = 0.50 * (GV + GR)
        EH = (9.0 * BH * GH) / (3.0 * BH + GH)
        nuH = (1.5 * BH - GH) / (3.0 * BH + GH)

        AVR = 100.0 * (GV - GR) / (GV + GR)
        out.S = S

        out.BR = BR
        out.GR = GR
        out.ER = ER
        out.nuR = nuR

        out.BH = BH
        out.GH = GH
        out.EH = EH
        out.nuH = nuH

        out.AVR = AVR
    except np.linalg.LinAlgError as e:
        print("LinAlgError:", e)

    eigval, eigvec = np.linalg.eig(C)
    out.C_eigval = eigval
    out.C_eigvec = eigvec

    return out


def fit_elastic_matrix(out: OutputElasticAnalysis, fit_order, v0, LC):
    import scipy

    A2 = []
    fit_order = int(fit_order)
    for s_e in out.strain_energy:
        ss = np.transpose(s_e)
        coeffs = np.polyfit(ss[0], ss[1] / v0, fit_order)
        A2.append(coeffs[fit_order - 2])

    A2 = np.array(A2)
    C = sym.get_C_from_A2(A2, LC)

    for i in range(5):
        for j in range(i + 1, 6):
            C[j, i] = C[i, j]

    CONV = (
        1e21 / scipy.constants.physical_constants["joule-electron volt relationship"][0]
    )  # From eV/Ang^3 to GPa

    C *= CONV
    out.C = C
    out.A2 = A2
    calculate_modulus(out)

    return out


def subjob_name(i, eps):
    return f"s_{i}_e{eps:.5f}".replace(".", "_").replace("-", "m")
