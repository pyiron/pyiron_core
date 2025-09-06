from dataclasses import dataclass, field
from collections.abc import Generator, Sequence
from numbers import Integral
from itertools import product

from pyiron_core.pyiron_workflow import Workflow, as_inp_dataclass_node, as_function_node

import pandas as pd
from ase import Atoms


@dataclass(frozen=True)
class Stoichiometry(Sequence):
    stoichiometry: tuple[dict[str, int]]

    @property
    def elements(self) -> set[str]:
        """Set of elements present in stoichiometry."""
        e = set()
        for s in self.stoichiometry:
            s = e.union(s.keys())
        return s

    # FIXME: Self only availabe in >=3.11
    def __add__(self, other: "Stoichiometry") -> "Stoichiometry":
        """Extend underlying list of stoichiometries."""
        return Stoichiometry(self.stoichiometry + other.stoichiometry)

    def __or__(self, other: "Stoichiometry") -> "Stoichiometry":
        """Inner product of underlying stoichiometries.

        Must not share elements with other stoichiometry."""
        assert self.elements.isdisjoint(
            other.elements
        ), "Can only or stoichiometries of different elements!"
        s = ()
        for me, you in zip(self.stoichiometry, other.stoichiometry):
            s += (me | you,)
        return Stoichiometry(s)

    def __mul__(self, other: "Stoichiometry") -> "Stoichiometry":
        """Outer product of underlying stoichiometries.

        Must not share elements with other stoichiometry."""
        assert self.elements.isdisjoint(
            other.elements
        ), "Can only multiply stoichiometries of different elements!"
        s = ()
        for me, you in product(self.stoichiometry, other.stoichiometry):
            s += (me | you,)
        return Stoichiometry(s)

    # Sequence Impl'
    def __getitem__(self, index: int) -> dict[str, int]:
        return self.stoichiometry[index]

    def __len__(self) -> int:
        return len(self.stoichiometry)


@as_function_node
def ElementInput(
    element: str,
    min_ion: int = 1,
    max_ion: int = 10,
    step_ion: int = 1,
) -> Stoichiometry:
    stoichiometry = Stoichiometry(
        tuple({element: i} for i in range(min_ion, max_ion + 1, step_ion))
    )
    return stoichiometry


@as_function_node("df")
def StoichiometryTable(stoichiometry: Stoichiometry) -> pd.DataFrame:
    return pd.DataFrame(stoichiometry.stoichiometry)


@as_inp_dataclass_node
class SpaceGroupInput:
    spacegroups: list[int] = field(default_factory=lambda: list(range(1, 231)))
    stoichiometry: Stoichiometry = None
    max_atoms: int = 10

    # can be either a single cutoff distance or a dictionary mapping chemical
    # symbols to min *radii*; you need to half the value if you go from using a
    # float to a dict
    min_dist: float | dict[str, float] | None = None

    # FIXME: just to restrict number of structures during testing
    max_structures: int = 20


@as_function_node
def SpaceGroupSampling(input: SpaceGroupInput) -> list[Atoms]:
    from warnings import catch_warnings
    from structuretoolkit.build.random import pyxtal
    from tqdm.auto import tqdm

    structures = []
    with catch_warnings(category=UserWarning, action="ignore"):
        for stoich in (bar := tqdm(input.stoichiometry)):
            elements, num_ions = zip(*stoich.items())
            stoich_str = "".join(f"{s}{n}" for s, n in zip(elements, num_ions))
            bar.set_description(stoich_str)
            structures += [
                s["atoms"] for s in pyxtal(input.spacegroups, elements, num_ions)
            ]
            if len(structures) > input.max_structures:
                structures = structures[: input.max_structures]
                break
        bar.close()
    return structures


@as_function_node
def CombineStructures(
    set1: list[Atoms],
    set2: list[Atoms],
    set3: list[Atoms] | None,
    set4: list[Atoms] | None,
    set5: list[Atoms] | None,
) -> list[Atoms]:
    """Combine a number of structure lists into a single list."""
    set3 = set3 or []
    set4 = set4 or []
    set5 = set5 or []
    structures = set1 + set2 + set3 + set4 + set5
    return structures


@as_function_node
def SaveStructures(structures: list[Atoms], filename: str):
    """Save list of structures into a pickled dataframe.

    Columns are:
        'name': a structure label
        'ase_atoms': the ASE object for the actual structure
        'number_of_atoms': the number of atoms inside the structure

    If `filename` does not end with 'pckl.gz', it is added.

    Args:
        structures (list of Atoms): structures to save
        filename (str): path where the dataframe is written to
    """
    import pandas as pd
    import os.path

    df = pd.DataFrame(
        [
            {
                "name": s.info.get("label", f"structure_{i}"),
                "ase_atoms": s,
                "number_of_atoms": len(s),
            }
            for i, s in enumerate(structures)
        ]
    )
    if not filename.endswith("pckl.gz"):
        filename += ".pckl.gz"
    dirname = os.path.dirname(filename)
    os.makedirs(dirname, exist_ok=True)
    df.to_pickle(filename)
