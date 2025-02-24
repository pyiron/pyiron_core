from dataclasses import dataclass
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
class SpaceGroupInput:
    # def __post_init__(self):
    #     if self.stoichiometry is None or len(self.stoichiometry) == 0:
    #         self.stoichiometry = list(range(1, self.max_atoms + 1))
    #     if self.spacegroups is None:
    #         self.spacegroups = list(range(1,231))

    elements: list[str] = "Al"
    max_atoms: int = 10
    stoichiometry: list[int] | list[tuple[int, ...]] | None = 1
    spacegroups: list[int] | None = None

    # can be either a single cutoff distance or a dictionary mapping chemical
    # symbols to min *radii*; you need to half the value if you go from using a
    # float to a dict
    min_dist: float | dict[str, float] | None = None

    # FIXME: just to restrict number of structures during testing
    max_structures: int = 20


@as_function_node
def get_stoichiometry(
    self: SpaceGroupInput,
) -> list[tuple[tuple[str, ...], tuple[int, ...]]]:
    """Yield pairs of str and int tuples."""
    if isinstance(self.stoichiometry[0], Integral):
        ions = filter(
            lambda x: 0 < sum(x) <= self.max_atoms,
            product(self.stoichiometry, repeat=len(self.elements)),
        )
    else:
        ions = iter(self.stoichiometry)

    stoichiometry = [
        zip(*((el, ni) for el, ni in zip(self.elements, num_ions) if ni > 0))
        for num_ions in ions
    ]
    return stoichiometry


# @as_function_node
# def get_distance_filter(self: SpaceGroupInput):
#     match self.min_dist:
#         case float():
#             return DistanceFilter({el: self.min_dist / 2 for el in self.elements})
#         case dict():
#             return DistanceFilter(self.min_dist)
#         case _:
#             assert (
#                 False
#             ), f"min_dist cannot by of type {type(self.min_dist)}: {self.min_dist}!"


@as_function_node
def SpaceGroupSampling(input: SpaceGroupInput) -> list[Atoms]:
    from warnings import catch_warnings
    from structuretoolkit.build.random import pyxtal

    structures = []
    with catch_warnings(category=UserWarning, action="ignore"):
        if isinstance(input.stoichiometry[0], Integral):
            ions = filter(
                lambda x: 0 < sum(x) <= input.max_atoms,
                product(input.stoichiometry, repeat=len(input.elements)),
            )
        else:
            ions = iter(input.stoichiometry)

        stoichiometry = [
            zip(*((el, ni) for el, ni in zip(input.elements, num_ions) if ni > 0))
            for num_ions in ions
    ]            

        for elements, num_ions in stoichiometry:
            structures += [
                s["atoms"] for s in pyxtal(input.spacegroups, elements, num_ions)
            ]
            if len(structures) > input.max_structures:
                structures = structures[: input.max_structures]
                break
    return structures
