from pyiron_core.pyiron_workflow import Workflow, as_function_node, as_macro_node


@as_macro_node("BulkStaticEnergy")
def BulkStaticEnergy(name: str, a: float = None, store: bool = True, _db=None) -> float:
    import pyiron_core.pyiron_nodes as pyiron_nodes

    wf = Workflow("subgraph")

    wf.Bulk = pyiron_nodes.atomistic.structure.build.Bulk(name=name, a=a)
    wf.M3GNet = pyiron_nodes.atomistic.engine.ase.M3GNet()
    wf.Static = pyiron_nodes.atomistic.calculator.ase.Static(
        structure=wf.Bulk, engine=wf.M3GNet
    )
    wf.GetEnergyLast = pyiron_nodes.atomistic.calculator.output.GetEnergyLast(
        calculator=wf.Static
    )

    return wf.GetEnergyLast.outputs.energy_last


@as_function_node("energy")
def BulkStaticEnergyF(name: str, a: float = None, store: bool = True, _db=None):

    import pyiron_core.pyiron_nodes as pyiron_nodes
    from pyiron_core.pyiron_workflow import Workflow

    wf = Workflow("subgraph")

    wf.Bulk = pyiron_nodes.atomistic.structure.build.Bulk(name=name, a=a)
    wf.M3GNet = pyiron_nodes.atomistic.engine.ase.M3GNet()
    wf.Static = pyiron_nodes.atomistic.calculator.ase.Static(
        structure=wf.Bulk, engine=wf.M3GNet
    )
    wf.GetEnergyLast = pyiron_nodes.atomistic.calculator.output.GetEnergyLast(
        calculator=wf.Static
    )

    out = wf.GetEnergyLast.pull()

    return out
