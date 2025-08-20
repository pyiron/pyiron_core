# from __future__ import annotations

from typing import Optional

from pyiron_workflow import as_function_node

from pyiron_nodes.atomistic.calculator.data import (
    OutputCalcMinimize,
    OutputCalcStaticList,
)


@as_function_node("energy_last")
def GetEnergyLast(
    # calculator: Optional[OutputCalcMinimize | OutputCalcStatic] = None,
    calculator=None,
    store: bool = False,
    _db=None,
) -> float:
    # print("GetEnergyLast", type(calculator))
    if isinstance(calculator, OutputCalcMinimize().dataclass):
        energy_last = calculator.final.energy
    elif isinstance(calculator, OutputCalcStaticList().dataclass):
        # print("staticlist")
        energy_last = calculator.energies_pot[-1]
    return energy_last
    # print ('energy_last:', calculator.energy[-1], type(calculator.energy[-1]))
    # return calculator.energy[-1]
