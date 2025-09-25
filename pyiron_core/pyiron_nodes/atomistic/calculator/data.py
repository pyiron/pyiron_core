# from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# from pyiron_core.pyiron_nodes.dev_tools import wf_data_class, wfMetaData
# from pyiron_core.pyiron_nodes.development.node_dataclass import as_output_node
from pyiron_core.pyiron_workflow import as_inp_dataclass_node, as_out_dataclass_node
from pyiron_core.pyiron_workflow.data_fields import DataArray, EmptyArrayField


# only conceptual, not truly implemented
def wfMetaData(log_level=0, doc=""):
    return {"log_level": log_level, "doc": doc}


@as_out_dataclass_node
class OutputSEFS:
    import numpy as np

    structures: Optional[np.ndarray] = field(
        default=None, metadata=wfMetaData(log_level=10)
    )
    energies: Optional[float] = field(default=None, metadata=wfMetaData(log_level=0))
    forces: Optional[np.ndarray] = field(default=None, metadata=wfMetaData(log_level=0))
    stresses: Optional[np.ndarray] = field(
        default=None, metadata=wfMetaData(log_level=10)
    )


@as_out_dataclass_node
class OutputCalcStatic:
    import numpy as np
    from ase import Atoms

    energy: Optional[float] = field(default=None, metadata=wfMetaData(log_level=0))
    force: Optional[np.ndarray] = field(default=None, metadata=wfMetaData(log_level=0))
    stress: Optional[np.ndarray] = field(
        default=None, metadata=wfMetaData(log_level=10)
    )
    structure: Optional[Atoms] = field(default=None, metadata=wfMetaData(log_level=10))

    atomic_energies: Optional[float] = field(
        default=None,
        metadata=wfMetaData(
            log_level=0, doc="per atom energy, only if supported by calculator"
        ),
    )


@as_out_dataclass_node
class OutputCalcStaticList:
    # from ase import Atoms
    import numpy as np

    energies_pot: Optional[np.ndarray] = field(
        default=None, metadata=wfMetaData(log_level=0)
    )
    forces: Optional[np.ndarray] = field(default=None, metadata=wfMetaData(log_level=0))
    stresses: Optional[np.ndarray] = field(
        default=None, metadata=wfMetaData(log_level=10)
    )
    structures: Optional[np.ndarray] = field(
        default=None, metadata=wfMetaData(log_level=10)
    )
    is_converged: bool = False
    iter_steps: int = 0


@as_out_dataclass_node
class OutputCalcMinimize:
    # energies: Optional[np.ndarray] = field(default=None, metadata=wfMetaData(log_level=0))
    initial: Optional[OutputCalcStatic] = field(
        default_factory=lambda: OutputCalcStatic().dataclass(),
        metadata=wfMetaData(log_level=0),
    )
    final: Optional[OutputCalcStatic] = field(
        default_factory=lambda: OutputCalcStatic().dataclass(),
        metadata=wfMetaData(log_level=0),
    )
    is_converged: bool = False
    iter_steps: int = 0


@as_out_dataclass_node
class OutputCalcMD:
    cells: DataArray = EmptyArrayField()
    energies_tot: DataArray = EmptyArrayField()
    energies_pot: DataArray = EmptyArrayField()
    forces: DataArray = EmptyArrayField()
    indices: DataArray = EmptyArrayField()
    natoms: DataArray = EmptyArrayField()
    positions: DataArray = EmptyArrayField()
    pressures: DataArray = EmptyArrayField()
    steps: DataArray = EmptyArrayField()
    temperatures: DataArray = EmptyArrayField()
    unwrapped_positions: DataArray = EmptyArrayField()
    velocities: DataArray = EmptyArrayField()
    volumes: DataArray = EmptyArrayField()


@as_inp_dataclass_node
class InputCalcMD:
    temperature: Optional[int | float] = 300
    n_ionic_steps: int = 10_000
    n_print: int = 100
    pressure: Optional[int | float] = None
    time_step: Optional[int | float] = 1.0
    temperature_damping_timescale: Optional[int | float] = 100.0
    pressure_damping_timescale: Optional[int | float] = 1000.0
    seed: Optional[int] = None
    tloop: Optional[float] = None
    initial_temperature: Optional[float] = None
    langevin: bool = False
    delta_temp: Optional[float] = None
    delta_press: Optional[float] = None


@as_inp_dataclass_node()
class InputCalcMinimize:
    """
        Sets parameters required for minimization.

    Parameters
    e_tol (float) – If the magnitude of difference between energies of two consecutive steps is lower than or equal to e_tol, the minimisation terminates. (Default is 0.0 eV.)

    f_tol (float) – If the magnitude of the global force vector at a step is lower than or equal to f_tol, the minimisation terminates. (Default is 1e-4 eV/angstrom.)

    max_iter (int) – Maximum number of minimisation steps to carry out. If the minimisation converges before max_iter steps, terminate at the converged step. If the minimisation does not converge up to max_iter steps, terminate at the max_iter step. (Default is 100000.)

    pressure (None/float/numpy.ndarray/list) – Target pressure. If set to None, an NVE or an NVT calculation is performed. A list of up to length 6 can be given to specify xx, yy, zz, xy, xz, and yz components of the pressure tensor, respectively. These values can mix floats and None to allow only certain degrees of cell freedom to change. (Default is None, run isochorically.)

    n_print (int) – Write (dump or print) to the output file every n steps (Default: 100)

    style ('cg'/'sd'/other values from Lammps docs) – The style of the numeric minimization, either conjugate gradient, steepest descent, or other keys permissible from the Lammps docs on ‘min_style’. (Default is ‘cg’ – conjugate gradient.)

    rotation_matrix (numpy.ndarray) – The rotation matrix from the pyiron to Lammps coordinate frame.
    """

    e_tol: float = 0.0
    f_tol: float = 1e-4
    max_iter: int = 1_000_000
    pressure: float = None
    n_print: int = 100
    style: str = "cg"


@as_inp_dataclass_node()
class InputCalcStatic:
    # keys_to_store: Optional[list] = field(default_factory=list)
    pass  # LammpsControl.calc_static takes exactly zero arguments, and currently we
    # have the input objects matching their respective LammpsControl counterparts
