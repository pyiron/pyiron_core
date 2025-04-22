from __future__ import annotations

from pyiron_workflow import as_function_node


@as_function_node("engine")
def EMT():
    from ase.calculators.emt import EMT
    from pyiron_nodes.atomistic.engine.generic import OutputEngine

    out = OutputEngine(calculator=EMT())

    return out


@as_function_node("engine")
def M3GNet(model: str = "M3GNet-MP-2021.2.8-PES"):
    """M3GNet: A universal neural network potential for atomistic simulations."""
    import matgl
    from pyiron_nodes.atomistic.engine.generic import OutputEngine
    from matgl.ext.ase import M3GNetCalculator

    out = OutputEngine(calculator=M3GNetCalculator(matgl.load_model(model)))
    return out


@as_function_node("engine")
def GRACE(model: str = "GRACE-1L-OAM_2Feb25"):
    """Universal Graph Atomic Cluster Expansion models."""
    from pyiron_nodes.atomistic.engine.generic import OutputEngine
    from tensorpotential.calculator import grace_fm

    out = OutputEngine(calculator=grace_fm(model))
    return out
