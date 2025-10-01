from pyiron_core.pyiron_nodes.atomistic.calculator.data import (
    OutputCalcMinimize,
    OutputCalcStaticList,
)
from pyiron_core.pyiron_workflow import as_function_node


@as_function_node("energy_last")
def GetEnergyLast(
    calculator=None,
    store: bool = False,
    _db=None,
) -> float:
    if isinstance(calculator, OutputCalcMinimize().dataclass):
        energy_last = calculator.final.energy
    elif isinstance(calculator, OutputCalcStaticList().dataclass):
        energy_last = calculator.energies_pot[-1]
    return energy_last
