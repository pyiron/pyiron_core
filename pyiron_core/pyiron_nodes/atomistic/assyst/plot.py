from typing import TYPE_CHECKING, Iterable

import pandas as pd
from ase import Atoms
from matplotlib.axis import Axis

from pyiron_core.pyiron_workflow import Workflow, as_function_node


@as_function_node
# def PlotSPG(structures: list[Atoms]) -> tuple[Axis, list[int]]:
def PlotSPG(structures: list[Atoms]) -> list[int]:
    """Plot a histogram of space groups in input list."""
    import matplotlib.pyplot as plt
    from structuretoolkit.analyse import get_symmetry

    spacegroups = []
    for structure in structures:
        spacegroups.append(get_symmetry(structure).info["number"])
    plt.hist(spacegroups)
    # ax = plt.gca()
    # return ax, spacegroups
    return spacegroups


@as_function_node("fig")
def PlotAtomsHistogram(structures: list[Atoms]):
    """
    Plot a histogram of the number of atoms in each structure.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    length = np.array([len(s) for s in structures])
    lo = length.min()
    hi = length.max()
    # make the bins fall in between whole numbers and include hi
    plt.hist(length, bins=np.arange(lo, hi + 2) - 0.5)
    plt.xlabel("#Atoms")
    plt.ylabel("Count")

    return plt.show()


@as_function_node("fig")
def PlotAtomsCells(
    structures: list[Atoms], angle_in_degrees: bool = True
) -> pd.DataFrame:
    """
    Plot histograms of cell parameters.

    Plotted are atomic volume, density, cell vector lengths and cell vector angles in separate subplots all on a
    log-scale.

    Args:
        structures (list of Atoms): structures to plot
        angle_in_degrees (bool): whether unit for angles is degree or radians

    Returns:
        `DataFrame`: contains the plotted information in the columns:
                        - a: length of first vector
                        - b: length of second vector
                        - c: length of third vector
                        - alpha: angle between first and second vector
                        - beta: angle between second and third vector
                        - gamma: angle between third and first vector
                        - V: volume of the cell
                        - N: number of atoms in the cell
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    N = np.array([len(s) for s in structures])
    C = np.array([s.cell.array for s in structures])

    # def get_angle(cell, idx=0):
    get_angle = lambda cell, idx=0: np.arccos(
        np.dot(cell[idx], cell[(idx + 1) % 3])
        / np.linalg.norm(cell[idx])
        / np.linalg.norm(cell[(idx + 1) % 3])
    )

    # def extract(n, c):
    extract = lambda n, c: {
        "a": np.linalg.norm(c[0]),
        "b": np.linalg.norm(c[1]),
        "c": np.linalg.norm(c[2]),
        "alpha": get_angle(c, 0),
        "beta": get_angle(c, 1),
        "gamma": get_angle(c, 2),
    }

    df = pd.DataFrame([extract(n, c) for n, c in zip(N, C)])
    df["V"] = np.linalg.det(C)
    df["N"] = N
    if angle_in_degrees:
        df["alpha"] = np.rad2deg(df["alpha"])
        df["beta"] = np.rad2deg(df["beta"])
        df["gamma"] = np.rad2deg(df["gamma"])

    plt.subplot(1, 4, 1)
    plt.title("Atomic Volume")
    plt.hist(df.V / df.N, bins=20, log=True)
    plt.xlabel(r"$V$ [$\AA^3$]")

    plt.subplot(1, 4, 2)
    plt.title("Density")
    plt.hist(df.N / df.V, bins=20, log=True)
    plt.xlabel(r"$\rho$ [$\AA^{-3}$]")

    plt.subplot(1, 4, 3)
    plt.title("Lattice Vector Lengths")
    plt.hist([df.a, df.b, df.c], log=True)
    plt.xlabel(r"$a,b,c$ [$\AA$]")

    plt.subplot(1, 4, 4)
    plt.title("Lattice Vector Angles")
    plt.hist([df.alpha, df.beta, df.gamma], log=True)
    if angle_in_degrees:
        label = r"$\alpha,\beta,\gamma$ [$^\circ$]"
    else:
        label = r"$\alpha,\beta,\gamma$ [rad]"
    plt.xlabel(label)

    # return df
    return plt.show()


@as_function_node("fig")
def PlotDistances(
    structures: list[Atoms],
    bins: int | Iterable[float] = 50,
    num_neighbors: int = 50,
    normalize: bool = True,
):
    """Plot radial distribution of a list of structures.

    Args:
        structures (list of Atoms): structures to plot
        bins (int or iterable of floats): if int number of bins; if iterable of floats bin edges
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from structuretoolkit import get_neighbors

    distances = []
    for structure in structures:
        distances.append(
            get_neighbors(structure, num_neighbors=num_neighbors).distances.ravel()
        )
    distances = np.concatenate(distances)

    if normalize:
        plt.hist(
            distances,
            bins=bins,
            weights=1 / (4 * np.pi * distances**2),
        )
        plt.ylabel(r"Neighbor density [$\mathrm{\AA}^{-2}$]")
    else:
        plt.hist(distances, bins=bins)
        plt.ylabel("Neighbor count")
    plt.xlabel(r"Distance [$\mathrm{\AA}$]")

    return plt.show()
