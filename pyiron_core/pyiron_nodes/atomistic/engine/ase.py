from __future__ import annotations

from pyiron_core.pyiron_workflow import as_function_node


@as_function_node("engine")
def EMT():
    from ase.calculators.emt import EMT

    from pyiron_core.pyiron_nodes.atomistic.engine.generic import OutputEngine

    out = OutputEngine(calculator=EMT())

    return out


@as_function_node("engine")
def GRACE(model: str = "GRACE-1L-MP-r6"):
    """Universal Graph Atomic Cluster Expansion models."""
    from tensorpotential.calculator import grace_fm

    from pyiron_core.pyiron_nodes.atomistic.engine.generic import OutputEngine

    out = OutputEngine(calculator=grace_fm(model))
    return out
