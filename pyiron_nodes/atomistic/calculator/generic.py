from pyiron_workflow import as_function_node, Node

from pyiron_nodes.atomistic.calculator.data import (
    InputCalcMinimize,
    InputCalcMD,
    InputCalcStatic,
    OutputSEFS,
)

from typing import Optional


@as_function_node("generic")
def Static(structure=None, engine=None):  # , keys_to_store=None):
    output = engine(
        structure=structure,
        calculator=InputCalcStatic(),  # keys_to_store=keys_to_store)
    )
    return output.generic


@as_function_node
def ApplyEngine(
    sefs_container: OutputSEFS = None,
    engine: Node=None,
    energies: bool = True,
    forces: bool = False,
    stresses: bool = False,
    store: bool = True,
) -> OutputSEFS:
    for structure in sefs_container.structures:
        engine.inputs.structure = structure
        out = engine.run()
        if energies:
            if sefs_container.energies is None:
                sefs_container.energies = []
            sefs_container.energies.append(out.energies_pot[-1])
        if forces:
            if sefs_container.forces is None:
                sefs_container.forces = []
            sefs_container.forces.append(out.forces)
        if stresses:
            if sefs_container.stresses is None:
                sefs_container.stresses = []
            sefs_container.stresses.append(out.stresses)
        sefs_container.energies.append(out.energies_pot[-1])

    return sefs_container
