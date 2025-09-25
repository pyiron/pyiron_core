from dataclasses import asdict, dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from pyiron_core.pyiron_workflow import (
    Workflow,
    as_function_node,
    as_inp_dataclass_node,
)


@as_inp_dataclass_node
class EmbeddingsALL:
    fs_parameters: list[int] = field(default_factory=lambda: [1, 1])
    npot: str = "FinnisSinclairShiftedScaled"
    ndensity: int = 1


@as_inp_dataclass_node
class Embeddings:
    ALL: EmbeddingsALL = field(default_factory=EmbeddingsALL)


@as_inp_dataclass_node
class BondsALL:
    radparameters: list[float] = field(default_factory=lambda: [5.25])
    radbase: str = "SBessel"
    rcut: float = 7.0
    dcut: float = 0.01


@as_inp_dataclass_node
class Bonds:
    ALL: BondsALL = field(default_factory=BondsALL)


@as_inp_dataclass_node
class FunctionsALL:
    nradmax_by_orders: list[int] = field(default_factory=lambda: [15, 3, 2, 1])
    lmax_by_orders: list[int] = field(default_factory=lambda: [0, 3, 2, 1])
    number_of_functions_per_element: Optional[int] = None


@as_inp_dataclass_node
class Functions:
    ALL: FunctionsALL = field(default_factory=FunctionsALL)


@as_function_node
def Elements(element1: str = None, element2: str = None):
    elements = []
    if element1 is not None:
        elements.append(element1)
    if element2 is not None:
        elements.append(element2)
    return elements


@as_inp_dataclass_node
class PotentialConfig:
    embeddings: Embeddings = field(default_factory=Embeddings())
    bonds: Bonds = field(default_factory=Bonds())
    functions: Functions = field(default_factory=Functions())
    elements: list[str] = field(default_factory=lambda: [])
    deltaSplineBins: float = 0.001


def remove_none(d):
    """Recursively remove None values from dictionaries."""
    if isinstance(d, dict):
        return {k: remove_none(v) for k, v in d.items() if v is not None}
    elif isinstance(d, list):
        return [remove_none(v) for v in d if v is not None]
    else:
        return d


@as_function_node("empty_basis_config")
def CreateEmptyBasisFunctions(potential_config: PotentialConfig):

    from dataclasses import asdict

    from pyace import create_multispecies_basis_config

    potential_config_dict = remove_none(asdict(potential_config))
    bconf = create_multispecies_basis_config(potential_config_dict)

    return bconf


@as_function_node
def ReadPickledDatasetAsDataframe(file_path: str, compression: str = None):

    from ase.atoms import Atoms as aseAtoms
    from pyiron_atomistics import Atoms as pyironAtoms

    df = pd.read_pickle(file_path, compression=compression)

    # Handling the dataframe to format in an acceptable format to python-ace

    # Atoms check
    if "atoms" in df.columns:
        at = df.iloc[0]["atoms"]
        # Checking that the elements themselves have the correct atoms format
        if isinstance(at, pyironAtoms):
            df["ase_atoms"] = df["atoms"].map(lambda s: s.to_ase())
            df.drop(columns=["atoms"], inplace=True)
        else:
            assert isinstance(
                at, aseAtoms
            ), "The 'atoms' column is not a valid ASE Atoms object"
            df.rename(columns={"atoms": "ase_atom"}, inplace=True)
    elif "ase_atoms" not in df.columns:
        raise ValueError(
            "DataFrame should contain 'atoms' (pyiron Atoms) or 'ase_atoms' (ASE atoms) columns"
        )

    # NUMBER OF ATOMS check
    if "NUMBER_OF_ATOMS" not in df.columns and "number_of_atoms" in df.columns:
        df.rename(columns={"number_of_atoms": "NUMBER_OF_ATOMS"}, inplace=True)

    df["NUMBER_OF_ATOMS"] = df["NUMBER_OF_ATOMS"].astype(int)

    # energy corrected check
    if "energy_corrected" not in df.columns and "energy" in df.columns:
        df.rename(columns={"energy": "energy_corrected"}, inplace=True)
    return df


@as_function_node
def SplitTrainingAndTesting(df, training_frac: float = 0.2, random_state: int = 42):

    if training_frac > 0.95:
        assert (
            True
        ), "Can't have the training dataset more than 99 % of the dataset\n\
            Setting the value to 99%"
        print(
            "Can't have the training dataset more than 99 % of the dataset\n\
            Setting the value to 99%"
        )
        training_frac = 0.99
    df_train = df.sample(frac=training_frac, random_state=random_state)
    df_test = df.loc[(i for i in df.index if i not in df_train.index)]
    return df_train, df_test


import pandas as pd


@as_function_node
def PrepareLinearACEdataset(
    bconf, df_train: pd.DataFrame, df_test: pd.DataFrame, verbose: bool = False
):
    """
    Prepare the LinearACE dataset for training and testing.

    Parameters:
    bconf: Basis configuration object.
    df_train (pd.DataFrame): Training dataset.
    df_test (pd.DataFrame): Testing dataset.
    verbose (bool): Whether to print verbose output.

    Returns:
    tuple: A tuple containing the training and testing LinearACEDataset objects.
    """

    from pyace.linearacefit import LinearACEDataset

    train_ds = LinearACEDataset(bconf, df_train)
    test_ds = LinearACEDataset(bconf, df_test)

    train_ds.construct_design_matrix(verbose=verbose)
    test_ds.construct_design_matrix(verbose=verbose)
    return train_ds, test_ds


# linear_fit = LinearACEFit(train_dataset=train_ds)
@as_function_node
def LinearACEFit(train_dataset, fit: bool = True):
    from pyace.linearacefit import LinearACEFit

    linear_fit = LinearACEFit(train_dataset=train_dataset)
    if fit:
        linear_fit.fit()
    return linear_fit


# # linear_fit.fit()
# @as_function_node("fit")
# def Fit(linear_fit):
#     return linear_fit.fit()


# linear_fit.compute_errors(test_ds)
@as_function_node("errors")
def ComputeErrors(linear_fit, ds):
    return linear_fit.compute_errors(ds)


# basis = linear_fit.get_bbasis()
@as_function_node("basis")
def GetBasis(linear_fit):
    return linear_fit.get_bbasis()


# e_pred,f_pred = linear_fit.predict(test_ds, reshape_forces=True)
@as_function_node
def Predict(linear_fit, ds, reshape_forces: bool = True):
    e_pred, f_pred = linear_fit.predict(ds, reshape_forces=reshape_forces)
    return e_pred, f_pred


@as_function_node("df")
def DisplayDataFrame(obj) -> pd.DataFrame:
    return obj.df


@as_function_node
def PotentialEnergy(df: pd.DataFrame, atom_index: int, linear_fit):
    import pyace

    basis = linear_fit.get_bbasis()
    calc = pyace.PyACECalculator(basis)
    at = df.iloc[atom_index]["ase_atoms"].copy()
    at.set_calculator(calc)
    potential_energy = at.get_potential_energy()
    return potential_energy


# get column from dataframe
@as_function_node
def GetColumnFromDataFrame(df, column_name: str):
    column = df[column_name]
    return column


@as_function_node("design_matrix")
def DesignMatrix(
    df_train: pd.DataFrame,
    potential_config: PotentialConfig,
    energy_only: bool = False,
    verbose: bool = False,
    store: bool = True,
):
    """
    Constructs the design matrix for the training dataset using the provided potential configuration.
    Args:
        df_train (pd.DataFrame): The training dataset containing ASE atoms and other properties.
        potential_config (PotentialConfig): The configuration for the potential.
    Returns:
        LinearACEDataset: The constructed design matrix for the training dataset.
    """

    from pyace import create_multispecies_basis_config
    from pyace.linearacefit import LinearACEDataset
    from pyiron_snippets.logger import logger

    logger.setLevel(30)

    elements_set = set()
    for at in df_train["ase_atoms"]:
        elements_set.update(at.get_chemical_symbols())

    elements = sorted(elements_set)
    potential_config.elements = elements
    potential_config_dict = potential_config.to_dict()

    bconf = create_multispecies_basis_config(potential_config_dict)

    train_ds = LinearACEDataset(bconf, df_train)
    train_ds.construct_design_matrix(verbose=verbose)
    design_matrix = train_ds.design_matrix
    if energy_only:
        design_matrix = design_matrix[: len(df_train)]
    return design_matrix


@as_function_node("matrix")
def GetMatrix(
    df: pd.DataFrame,
    matrix,
    energy_only: bool = False,
    min_index: int = 0,
    max_index: int = None,
):
    import numpy as np

    num_structures = len(df)
    num_atoms = np.sum(df.NUMBER_OF_ATOMS)

    if max_index is None:
        max_index = num_structures
    new_matrix = matrix[min_index, max_index]
    return new_matrix


@as_function_node
def GetVector(
    df: pd.DataFrame,
    energies: bool = True,
    forces: bool = True,
    scale_energy_per_atom: bool = False,
):
    import numpy as np

    vec = np.array([])
    if energies:
        vec = np.append(vec, df.energy_corrected)
        if scale_energy_per_atom:
            vec /= df.NUMBER_OF_ATOMS
    if forces:
        # Flatten the forces array and append to the vector
        forces_vec = []
        for f in df.forces.apply(lambda x: x.flatten()):
            forces_vec += list(f)
        vec = np.append(vec, forces_vec)
    return vec
