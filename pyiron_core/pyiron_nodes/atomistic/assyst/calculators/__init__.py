from abc import ABC, abstractmethod
from dataclasses import dataclass

from ase import Atoms
from ase.calculators.calculator import Calculator

from pyiron_core.pyiron_workflow import Workflow


class AseCalculatorConfig(ABC):
    @abstractmethod
    def get_calculator(self) -> Calculator:
        pass


@dataclass
class PawDftInput:
    encut: int | float | None = 320.0
    kspacing: float | None = 0.5

    scf_energy_convergence: float = 1e-2


@Workflow.wrap.as_dataclass_node
@dataclass
class GpawInput(AseCalculatorConfig, PawDftInput):
    def get_calculator(self, use_symmetry=True):
        import gpaw
        return gpaw.GPAW(
            xc="PBE",
            kpts=(1, 1, 1),
            h=0.25,
            nbands=-2,
            mode=gpaw.PW(self.encut, dedecut="estimate"),
            # FIXME deliberately high values for testing
            convergence={
                "energy": self.scf_energy_convergence,
                "density": 1,
                "eigenstates": 1e-3,
            },
            symmetry={"point_group": use_symmetry},
            txt=None,
        )


@Workflow.wrap.as_dataclass_node
@dataclass
class GenericOptimizerSettings:
    max_steps: int = 10
    force_tolerance: float = 1e-2


from enum import Enum

import numpy as np
from ase.constraints import FixAtoms, FixSymmetry
from ase.filters import FrechetCellFilter, StrainFilter


class RelaxMode(Enum):
    VOLUME = "volume"
    # CELL = "cell"
    # POSITION = "position"
    FULL = "full"

    def apply_filter_and_constraints(self, structure):
        match self:
            case RelaxMode.VOLUME:
                structure.set_constraint(FixAtoms(np.ones(len(structure), dtype=bool)))
                return FrechetCellFilter(structure, hydrostatic_strain=True)
            # case RelaxMode.CELL:
            #     return FrechetCellFilter(structure, constant_volume=True)
            case RelaxMode.FULL:
                return FrechetCellFilter(structure)
            case _:
                raise ValueError("Lazy Marvin")


@Workflow.wrap.as_function_node
def Relax(
    mode: str | RelaxMode,
    calculator: AseCalculatorConfig,
    opt: GenericOptimizerSettings.dataclass,
    structure: Atoms,
) -> Atoms:
    from ase.calculators.singlepoint import SinglePointCalculator
    from ase.optimize import LBFGS

    mode = RelaxMode(mode)

    structure = structure.copy()

    # FIXME: meh
    match mode:
        case RelaxMode.VOLUME:
            structure.calc = calculator.get_calculator(use_symmetry=True)
        case RelaxMode.FULL:
            structure.calc = calculator.get_calculator(use_symmetry=False)
        case _:
            assert False

    filtered_structure = mode.apply_filter_and_constraints(structure)
    lbfgs = LBFGS(filtered_structure)
    lbfgs.run(fmax=opt.force_tolerance, steps=opt.max_steps)
    calc = structure.calc
    structure.calc = SinglePointCalculator(
        structure,
        **{
            "energy": calc.get_potential_energy(),
            "forces": calc.get_forces(),
            "stress": calc.get_stress(),
        },
    )
    # play catch with nodes
    relaxed_structure = structure
    return relaxed_structure


from matgl import load_model, models
from matgl.ext.ase import M3GNetCalculator

GPA2EVA3 = 0.006_241_509_074


@Workflow.wrap.as_dataclass_node
@dataclass
class M3gnetConfig(AseCalculatorConfig):
    model: str = "M3GNet-MP-2021.2.8-PES"

    def get_calculator(self, use_symmetry=True):
        return M3GNetCalculator(
            load_model(self.model), compute_stress=True, stress_weight=GPA2EVA3
        )
