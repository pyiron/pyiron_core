from __future__ import annotations

from pyiron_workflow import as_function_node


@as_function_node("engine")
def EMT():
    """Create an OutputEngine wrapping the ASE EMT calculator.

    The EMT (Effective Medium Theory) potential provides a fast, generic
    empirical description of metallic systems. It is useful for quick
    prototyping, testing workflows, or when high‑throughput calculations
    require a low‑cost energy and force evaluator.

    Returns
    -------
    OutputEngine
        Engine containing an ASE EMT calculator.
    """
    from ase.calculators.emt import EMT
    from pyiron_nodes.atomistic.engine.generic import OutputEngine

    out = OutputEngine(calculator=EMT())

    return out


@as_function_node("engine")
def M3GNet(model: str = "M3GNet-MP-2021.2.8-PES"):
    """Create an OutputEngine using the M3GNet neural network potential.

    M3GNet is a universal graph‑based neural network trained on a large
    dataset of DFT calculations. It can be applied to a wide range of
    materials and provides DFT‑level accuracy at a fraction of the cost.

    Parameters
    ----------
    model : str, optional
        Identifier of the pretrained M3GNet model to load. Defaults to
        ``"M3GNet-MP-2021.2.8-PES"``.

    Returns
    -------
    OutputEngine
        Engine wrapping a ``M3GNetCalculator`` ready for ASE simulations.
    """
    import matgl
    from pyiron_nodes.atomistic.engine.generic import OutputEngine
    from matgl.ext.ase import M3GNetCalculator

    out = OutputEngine(calculator=M3GNetCalculator(matgl.load_model(model)))
    return out


@as_function_node("engine")
def GRACE(model: str = "GRACE-1L-MP-r6"):
    """Create an OutputEngine using a GRACE (Graph Atomic Cluster Expansion) model.

    GRACE provides a flexible, data‑driven potential based on the atomic
    cluster expansion formalism. It is suitable for systems where a
    machine‑learning model trained on DFT data is desired.

    Parameters
    ----------
    model : str, optional
        Name of the pretrained GRACE model to load. Defaults to
        ``"GRACE-1L-MP-r6"``.

    Returns
    -------
    OutputEngine
        Engine containing the GRACE calculator.
    """
    from pyiron_nodes.atomistic.engine.generic import OutputEngine
    from tensorpotential.calculator import grace_fm

    out = OutputEngine(calculator=grace_fm(model))
    return out
