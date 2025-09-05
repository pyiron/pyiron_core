from __future__ import annotations

from typing import Optional

from ase import Atoms
from pyiron_workflow import as_function_node, as_macro_node
import numpy as np


@as_function_node
def Volume(structure: Optional[Atoms] = None, per_atom: bool = False) -> float:
    # print("Calculating volume of the structure", structure)
    volume = structure.get_volume()
    # if per_atom:
    volume = volume / structure.get_number_of_atoms()
    return volume


@as_function_node
def NumberOfAtoms(structure: Optional[Atoms] = None) -> int:
    number_of_atoms = structure.get_number_of_atoms()
    return number_of_atoms()


@as_function_node
def GetDistances(
    structure: Optional[Atoms] = None,
    num_neighbors: int = 12,
    flatten: bool = True,
    # vector: bool = False,
) -> np.ndarray:
    """
    Get distances between atoms in the structure.

    :param structure: Atoms object representing the structure.
    :param num_neighbors: Number of nearest neighbors to consider.
    :param flatten: If True, return a flat list of distances.
    # :param vector: If True, return distances as vectors.
    :return: List of distances between atoms.
    """
    from structuretoolkit import get_neighbors

    distances = get_neighbors(structure, num_neighbors=num_neighbors).distances
    if flatten:
        distances = distances.flatten()

    return distances


@as_function_node
def SplineDescriptor(
    structure: Optional[Atoms] = None,
    r_min: float = 2.5,
    r_max: float = 7,
    num_points: int = 51,
    degree: int = 3,
) -> np.ndarray:
    """
    Interpolate a descriptor linearly over a specified range.
    :param structure: Atoms object representing the structure.
    :param num_points: Number of points in the interpolation.
    :param r_min: Start value for interpolation.
    :param r_max: Stop value for interpolation.
    :return: Interpolated values of the descriptor.
    """
    from pyiron_core.pyiron_nodes.math import BSpline

    if structure is None:
        descriptor = None
    else:

        r_bins = np.linspace(r_min, r_max, num_points)

        distances = GetDistances()._func(
            structure=structure, num_neighbors=num_points, flatten=True
        )

        # r_bins = np.linspace(r_min, r_max, num_points)
        descriptor, deriv_descriptor, r_bins = BSpline()._func(
            x0_vals=distances,
            x_min=r_bins[0],
            x_max=r_bins[-1],
            steps=num_points,
            degree=degree,
        )

    return descriptor #, deriv_descriptor, r_bins


@as_function_node
def LinearInterpolationDescriptor(
    structure: Optional[Atoms] = None,
    # descriptor: str = "distance",
    r_bins: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Interpolate a descriptor linearly over a specified range.
    :param structure: Atoms object representing the structure.
    :param num_points: Number of points in the interpolation.
    :param r_min: Start value for interpolation.
    :param r_max: Stop value for interpolation.
    :return: Interpolated values of the descriptor.
    """
    from pyiron_core.pyiron_nodes.math import LinearBin

    if structure is None:
        counts = None
    else:
        if r_bins is None:
            r_bins = np.linspace(2.5, 7, 51)  # Default range if not provided

        num_points = r_bins.size
        distances = GetDistances()._func(
            structure=structure, num_neighbors=num_points, flatten=True
        )

        # r_bins = np.linspace(r_min, r_max, num_points)
        counts, _ = LinearBin()._func(data=distances, bin_centers=r_bins)

    return counts


@as_function_node
def GetNeighbors(
    structure: Optional[Atoms] = None,
    num_neighbors: int = 12,
    tolerance: int = 2,
    id_list: Optional[list] = None,
    cutoff_radius: float = 10.0,
    width_buffer: float = 1.2,
    mode: str = "filled",
    norm_order: int = 2,
):
    """
    Get the neighbors of each atom in the structure.

    :param structure: Atoms object representing the structure.
    :param num_neighbors: Number of nearest neighbors to consider.
    :return: List of neighbors for each atom.
    """
    from structuretoolkit import get_neighbors

    neighbors = get_neighbors(
        structure,
        num_neighbors=num_neighbors,
        tolerance=tolerance,
        id_list=id_list,
        cutoff_radius=cutoff_radius,
        width_buffer=width_buffer,
        mode=mode,
        norm_order=norm_order,
    )

    return neighbors


@as_macro_node(
    [
        "coefficients",
        "design_matrix",
        "r_bins",
        "diff_energy_per_atom",
        "fit_diff_energy_per_atom",
        "number_of_atoms",
    ]
)
def FitDiffPotential(
    file_path_0: str = "ASSYST/Al_LDA.pckl.gz",
    file_path_1: str = "ASSYST/Al_PBE.pckl.gz",
    r_min: float = 2.5,
    r_max: float = 7,
    num_points: int = 51,
    max_row_index: int = -1,
    store: bool = True,
):

    from pyiron_workflow import Workflow
    from pyiron_core.pyiron_nodes.atomistic.ml_potentials.fitting.linearfit import (
        ReadPickledDatasetAsDataframe,
    )
    from pyiron_core.pyiron_nodes.math import Linspace, Divide, DotProduct
    from pyiron_core.pyiron_nodes.atomistic.structure.calc import LinearInterpolationDescriptor
    from pyiron_core.pyiron_nodes.dataframe import (
        MergeDataFrames,
        GetRowsFromDataFrame,
        GetColumnFromDataFrame,
        ApplyFunctionToSeriesNew,
    )
    from pyiron_core.pyiron_nodes.math import Subtract, PseudoInverse, Sum, DotProduct

    wf = Workflow("assyst_linear_fit3")

    wf.ReadData = ReadPickledDatasetAsDataframe(
        file_path=file_path_0,
        compression="gzip",
    )

    wf.ReadRefData = ReadPickledDatasetAsDataframe(
        file_path=file_path_1,
        compression="gzip",
    )

    wf.Linspace = Linspace(x_min=r_min, x_max=r_max, num_points=num_points)

    wf.MergeDataFrames = MergeDataFrames(
        df1=wf.ReadData,
        df2=wf.ReadRefData,
        on="name",
        how="inner",
    )

    wf.LinearInterpolationDescriptor = LinearInterpolationDescriptor(r_bins=wf.Linspace)

    wf.GetRowsFromDataFrame = GetRowsFromDataFrame(
        df=wf.MergeDataFrames, max_index=max_row_index
    )

    wf.GetStructures = GetColumnFromDataFrame(
        df=wf.GetRowsFromDataFrame, column_name="ase_atoms_x"
    )

    wf.NumberOfAtoms = GetColumnFromDataFrame(
        df=wf.GetRowsFromDataFrame, column_name="NUMBER_OF_ATOMS_x"
    )

    wf.GetEnergy = GetColumnFromDataFrame(
        df=wf.GetRowsFromDataFrame, column_name="energy_corrected_y"
    )

    wf.GetRefEnergy = GetColumnFromDataFrame(
        df=wf.GetRowsFromDataFrame, column_name="energy_corrected_x"
    )

    wf.DesignMatrix = ApplyFunctionToSeriesNew(
        series=wf.GetStructures,
        function=wf.LinearInterpolationDescriptor,
        store=store,
    )

    wf.DiffEnergy = Subtract(x=wf.GetEnergy, y=wf.GetRefEnergy)

    wf.PseudoInverse = PseudoInverse(matrix=wf.DesignMatrix)

    wf.Sum = Sum(x=wf.DesignMatrix, axis=0)

    wf.Coeff = DotProduct(a=wf.PseudoInverse, b=wf.DiffEnergy, store=store)
    wf.DiffEnergyPerAtom = Divide(wf.DiffEnergy, wf.NumberOfAtoms)
    wf.FitEnergyDiff = DotProduct(a=wf.DesignMatrix, b=wf.Coeff)
    wf.FitEnergyDiffPerAtom = Divide(wf.FitEnergyDiff, wf.NumberOfAtoms)

    return (
        wf.Coeff,
        wf.DesignMatrix,
        wf.Linspace,
        wf.DiffEnergyPerAtom,
        wf.FitEnergyDiffPerAtom,
        wf.NumberOfAtoms,
    )


@as_macro_node(
    [
        "coefficients",
        "design_matrix",
        "r_bins",
        "diff_energy_per_atom",
        "fit_diff_energy_per_atom",
        "number_of_atoms",
    ]
)
def FitDiffPotential2(
    file_path_0: str = "ASSYST/Al_LDA.pckl.gz",
    file_path_1: str = "ASSYST/Al_PBE.pckl.gz",
    r_min: float = 2.5,
    r_max: float = 7,
    num_points: int = 51,
    degree: int = 3,
    max_row_index: int = -1,
    store: bool = True,
):

    from pyiron_workflow import Workflow
    from pyiron_core.pyiron_nodes.atomistic.ml_potentials.fitting.linearfit import (
        ReadPickledDatasetAsDataframe,
    )
    from pyiron_core.pyiron_nodes.math import Divide, DotProduct
    from pyiron_core.pyiron_nodes.atomistic.structure.calc import SplineDescriptor
    from pyiron_core.pyiron_nodes.dataframe import (
        MergeDataFrames,
        GetRowsFromDataFrame,
        GetColumnFromDataFrame,
        ApplyFunctionToSeriesNew,
    )
    from pyiron_core.pyiron_nodes.math import Subtract, PseudoInverse, Sum, DotProduct

    wf = Workflow("assyst_linear_fit3")

    wf.ReadData = ReadPickledDatasetAsDataframe(
        file_path=file_path_0,
        compression="gzip",
    )

    wf.ReadRefData = ReadPickledDatasetAsDataframe(
        file_path=file_path_1,
        compression="gzip",
    )

    wf.MergeDataFrames = MergeDataFrames(
        df1=wf.ReadData,
        df2=wf.ReadRefData,
        on="name",
        how="inner",
    )

    wf.Descriptor = SplineDescriptor(r_min=r_min, r_max=r_max, num_points=num_points, degree=degree)

    wf.GetRowsFromDataFrame = GetRowsFromDataFrame(
        df=wf.MergeDataFrames, max_index=max_row_index
    )

    wf.GetStructures = GetColumnFromDataFrame(
        df=wf.GetRowsFromDataFrame, column_name="ase_atoms_x"
    )

    wf.NumberOfAtoms = GetColumnFromDataFrame(
        df=wf.GetRowsFromDataFrame, column_name="NUMBER_OF_ATOMS_x"
    )

    wf.GetEnergy = GetColumnFromDataFrame(
        df=wf.GetRowsFromDataFrame, column_name="energy_corrected_y"
    )

    wf.GetRefEnergy = GetColumnFromDataFrame(
        df=wf.GetRowsFromDataFrame, column_name="energy_corrected_x"
    )

    wf.DesignMatrix = ApplyFunctionToSeriesNew(
        series=wf.GetStructures,
        function=wf.Descriptor,
        store=store,
    )

    wf.DiffEnergy = Subtract(x=wf.GetEnergy, y=wf.GetRefEnergy)

    wf.PseudoInverse = PseudoInverse(matrix=wf.DesignMatrix)

    wf.Sum = Sum(x=wf.DesignMatrix, axis=0)

    wf.Coeff = DotProduct(a=wf.PseudoInverse, b=wf.DiffEnergy, store=store)
    wf.DiffEnergyPerAtom = Divide(wf.DiffEnergy, wf.NumberOfAtoms)
    wf.FitEnergyDiff = DotProduct(a=wf.DesignMatrix, b=wf.Coeff)
    wf.FitEnergyDiffPerAtom = Divide(wf.FitEnergyDiff, wf.NumberOfAtoms)

    return (
        wf.Coeff,
        wf.DesignMatrix,
        wf.Descriptor,
        wf.DiffEnergyPerAtom,
        wf.FitEnergyDiffPerAtom,
        wf.NumberOfAtoms,
    )
