from dataclasses import dataclass, field
from collections.abc import Generator
from numbers import Integral
from itertools import product

from pyiron_workflow import Workflow

from ase import Atoms

from pyiron_workflow import as_inp_dataclass_node, as_out_dataclass_node
from pyiron_workflow import (
    as_function_node,
    as_macro_node,
)


@as_inp_dataclass_node
class ElementInput:
    element: str = "Al"
    num: int = 1


@as_inp_dataclass_node
class SpaceGroupInput:
    spacegroups: list[int] = field(default_factory=lambda: list(range(1, 231)))
    element1: ElementInput = None
    element2: ElementInput = None
    max_atoms: int = 10
    max_structures: int = 10


@as_function_node
def SpaceGroupSampling(input: SpaceGroupInput, store: bool = True):  # -> list[Atoms]:
    from warnings import catch_warnings
    from structuretoolkit.build.random import pyxtal
    from pyiron_core.pyiron_nodes.atomistic.calculator.data import OutputSEFS

    structures = []
    elements = []
    stoichiometry = []
    for inp in [input.element1, input.element2]:
        if inp is None:
            continue
        if inp.element is None:
            continue
        elements.append(inp.element)
        stoichiometry.append(inp.num)
        # el_num.append((elements, num_ions))
    print("elements: ", elements, stoichiometry)

    ions = filter(
        lambda x: 0 < sum(x) <= input.max_atoms,
        product(stoichiometry, repeat=len(elements)),
    )

    # print(list(ions))

    el_list, n_list = [], []
    for n_ions in ions:
        elements, num_ions = zip(
            *((el, ni) for el, ni in zip(elements, n_ions) if ni > 0)
        )
        el_list.append(elements)
        n_list.append(num_ions)

    with catch_warnings():
        for elements, num_ions in zip(el_list, n_list):
            print("crystal elements: ", elements, num_ions)
            structures += [
                s["atoms"] for s in pyxtal(input.spacegroups, elements, num_ions)
            ]
            if len(structures) > input.max_structures:
                structures = structures[: input.max_structures]
                break

    out_sefs = OutputSEFS().dataclass()
    out_sefs.structures = structures
    return out_sefs
