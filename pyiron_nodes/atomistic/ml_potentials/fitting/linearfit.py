from pyiron_workflow import Workflow, as_function_node
import pandas as pd
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import Optional
import matplotlib.pyplot as plt


@dataclass
class EmbeddingsALL:
    npot: str = "FinnisSinclairShiftedScaled"
    fs_parameters: list[int] = field(default_factory=lambda: [1, 1])
    ndensity: int = 1


@dataclass
class Embeddings:
    ALL: EmbeddingsALL = field(default_factory=EmbeddingsALL)


@dataclass
class BondsALL:
    radbase: str = "SBessel"
    radparameters: list[float] = field(default_factory=lambda: [5.25])
    rcut: float | int = 7.0
    dcut: float = 0.01


@dataclass
class Bonds:
    ALL: BondsALL = field(default_factory=BondsALL)


@dataclass
class FunctionsALL:
    nradmax_by_orders: list[int] = field(default_factory=lambda: [15, 3, 2, 1])
    lmax_by_orders: list[int] = field(default_factory=lambda: [0, 3, 2, 1])


@dataclass
class Functions:
    number_of_functions_per_element: Optional[int | None] = None
    ALL: FunctionsALL = field(default_factory=FunctionsALL)


@dataclass
class PotentialConfig:
    deltaSplineBins: float = 0.001
    elements: list[str] | None = None

    embeddings: Embeddings = field(default_factory=Embeddings)
    bonds: Bonds = field(default_factory=Bonds)
    functions: Functions = field(default_factory=Functions)

    def __post_init__(self):
        if not isinstance(self.embeddings, Embeddings):
            self.embeddings = Embeddings()
        if not isinstance(self.bonds, Bonds):
            self.bonds = Bonds()
        if not isinstance(self.functions, Functions):
            self.functions = Functions()

    def to_dict(self):
        def remove_none(d):
            """Recursively remove None values from dictionaries."""
            if isinstance(d, dict):
                return {k: remove_none(v) for k, v in d.items() if v is not None}
            elif isinstance(d, list):
                return [remove_none(v) for v in d if v is not None]
            else:
                return d

        return remove_none(asdict(self))


@as_function_node
def ReadPickledDatasetAsDataframe(
    file_path: str = "", compression: Optional[str] = None
):

    from ase.atoms import Atoms as aseAtoms

    df = pd.read_pickle(file_path, compression=compression)

    # Atoms check
    if "atoms" in df.columns:
        at = df.iloc[0]["atoms"]
        # Checking that the elements themselves have the correct atoms format
        if isinstance(at, aseAtoms):
            df.rename(columns={"atoms": "ase_atoms"}, inplace=True)
    elif "ase_atoms" not in df.columns:
        raise ValueError(
            "DataFrame should contain 'atoms' or 'ase_atoms' (ASE atoms) columns"
        )

    # NUMBER OF ATOMS check
    if "NUMBER_OF_ATOMS" not in df.columns and "number_of_atoms" in df.columns:
        df.rename(columns={"number_of_atoms": "NUMBER_OF_ATOMS"}, inplace=True)

    df["NUMBER_OF_ATOMS"] = df["NUMBER_OF_ATOMS"].astype(int)

    # energy corrected check
    if "energy_corrected" not in df.columns and "energy" in df.columns:
        df.rename(columns={"energy": "energy_corrected"}, inplace=True)

    if "pbc" not in df.columns:
        df["pbc"] = df["ase_atoms"].map(lambda atoms: np.all(atoms.pbc))

    return df


@as_function_node
def ParameterizePotentialConfig(
    nrad_max: list = [15, 6, 4, 1],
    l_max: list = [0, 6, 5, 1],
    # number_of_functions_per_element: int | None = None,
    number_of_functions_per_element: int = 10,
    rcut: float = 7.0,
):

    potential_config = PotentialConfig()

    potential_config.bonds.ALL.rcut = rcut
    potential_config.functions.ALL.nradmax_by_orders = nrad_max
    potential_config.functions.ALL.lmax_by_orders = l_max
    potential_config.functions.number_of_functions_per_element = (
        number_of_functions_per_element
    )

    return potential_config


@as_function_node
def SplitTrainingAndTesting(
    data_df: pd.DataFrame, training_frac: float = 0.5, random_state: int = 42
):
    """
    Splits the filtered dataframe into training and testing sets based on a fraction of the dataset

    Args:
        data_df: A pandas.DataFrame of the filtered data DataFrame
        training_frac: A float number which dictates what is the precentage of the dataset to be used for training should be set between 0 to 1
        random_state (default = 42): Sets the random seed used to shuffle the data

    Returns:
        df_training: The training dataframe
        df_testing: The testing dataframe
    """
    if isinstance(training_frac, float):
        training_frac = np.abs(training_frac)

    if training_frac > 1:
        print(
            "Can't have the training dataset more than 100 % of the dataset\n\
            Setting the value to 100%"
        )
        training_frac = 1
    elif training_frac == 0:
        print("Can'fit with no training dataset\nSetting the value to 1%")
        training_frac = 0.01
    df_training = data_df.sample(frac=training_frac, random_state=random_state)
    df_testing = data_df.loc[(i for i in data_df.index if i not in df_training.index)]

    return df_training, df_testing


@as_function_node
def RunLinearFit(
    potential_config,
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    verbose: bool = False,
    store: bool = True,
):

    from pyace.linearacefit import LinearACEFit, LinearACEDataset
    from pyace import create_multispecies_basis_config

    from pyiron_snippets.logger import logger

    logger.setLevel(30)

    elements_set = set()
    for at in df_train["ase_atoms"]:
        elements_set.update(at.get_chemical_symbols())
    for at in df_test["ase_atoms"]:
        elements_set.update(at.get_chemical_symbols())

    elements = sorted(elements_set)
    potential_config.elements = elements
    potential_config_dict = potential_config.to_dict()

    bconf = create_multispecies_basis_config(potential_config_dict)

    train_ds = LinearACEDataset(bconf, df_train)
    train_ds.construct_design_matrix(verbose=verbose)
    if df_test.empty is False:
        test_ds = LinearACEDataset(bconf, df_test)
        test_ds.construct_design_matrix(verbose=verbose)
    else:
        test_ds = None

    linear_fit = LinearACEFit(train_dataset=train_ds)
    linear_fit.fit()

    training_dict = linear_fit.compute_errors(train_ds)
    training_e_rmse = round(training_dict["epa_rmse"] * 1000, 2)
    training_f_rmse = round(training_dict["f_comp_rmse"] * 1000, 2)
    print("====================== TRAINING INFO ======================")
    print(f"Training E RMSE: {training_e_rmse:.2f} meV/atom")
    print(f"Training F RMSE: {training_f_rmse:.2f} meV/A")

    if test_ds is not None:
        testing_dict = linear_fit.compute_errors(test_ds)
        testing_e_rmse = round(testing_dict["epa_rmse"] * 1000, 2)
        testing_f_rmse = round(testing_dict["f_comp_rmse"] * 1000, 2)
        print("======================= TESTING INFO =======================")
        print(f"Testing E RMSE: {testing_e_rmse:.2f} meV/atom")
        print(f"Testing F RMSE: {testing_f_rmse:.2f} meV/A")

    basis = linear_fit.get_bbasis()
    return basis


@as_function_node
def SavePotential(basis, filename: str = ""):
    import os

    if filename == "":
        filename = f"{'_'.join(basis.elements_name)}_linear_potential"
        folder_name = "Linear_ace_potentials"
    else:
        folder_name = os.path.dirname(filename)
        filename = os.path.basename(filename)

    folder_name = "Linear_ace_potentials"
    os.makedirs(folder_name, exist_ok=True)

    current_path = os.getcwd()
    folder_path = current_path + "/" + folder_name
    # Saving yaml and yace files
    print(
        f'Potentials "{filename}.yaml" and "{filename}.yace" are saved in "{folder_path}".'
    )

    yace_file_path = f"{folder_path}/{filename}.yace"
    basis.save(f"{folder_path}/{filename}.yaml")
    basis.to_ACECTildeBasisSet().save_yaml(yace_file_path)

    return basis, yace_file_path


@as_function_node
def PredictEnergiesAndForces(
    basis, df_train: pd.DataFrame, df_test: pd.DataFrame, store: bool = True
):

    from pyace import PyACECalculator

    data_dict = {}

    ace = PyACECalculator(basis)

    training_structures = df_train.ase_atoms

    # Reference data
    training_number_of_atoms = df_train.NUMBER_OF_ATOMS.to_numpy()
    training_energies = df_train.energy_corrected.to_numpy()

    training_epa = training_energies / training_number_of_atoms
    training_fpa = np.concatenate(df_train.forces.to_numpy()).flatten()
    data_dict["reference_training_epa"] = training_epa
    data_dict["reference_training_fpa"] = training_fpa

    # Predicted data
    training_predict = _get_predicted_energies_forces(
        ace=ace, structures=training_structures
    )
    data_dict["predicted_training_epa"] = (
        np.array(training_predict[0]) / training_number_of_atoms
    )
    data_dict["predicted_training_fpa"] = np.concatenate(training_predict[1]).flatten()

    if df_test.empty is False:

        testing_structures = df_test.ase_atoms

        # Reference data
        testing_number_of_atoms = df_test.NUMBER_OF_ATOMS.to_numpy()
        testing_energies = df_test.energy_corrected.to_numpy()

        testing_epa = testing_energies / testing_number_of_atoms
        testing_fpa = np.concatenate(df_test.forces.to_numpy()).flatten()
        data_dict["reference_testing_epa"] = testing_epa
        data_dict["reference_testing_fpa"] = testing_fpa

        # Predicted data
        testing_predict = _get_predicted_energies_forces(
            ace=ace, structures=testing_structures
        )
        data_dict["predicted_testing_epa"] = (
            np.array(testing_predict[0]) / testing_number_of_atoms
        )
        data_dict["predicted_testing_fpa"] = np.concatenate(
            testing_predict[1]
        ).flatten()

    return data_dict


def _get_predicted_energies_forces(ace, structures):
    forces = []
    energies = []

    for s in structures:
        s.calc = ace
        energies.append(s.get_potential_energy())
        forces.append(s.get_forces())
        s.calc = None
    return energies, forces


def _calc_rmse(array_1, array_2, rmse_in_milli: bool = True):
    """
    Calculates the RMSE value of two arrays

    Args:
    array_1: An array or list of energy or force values
    array_2: An array or list of energy or force values

    Returns:
    rmse_in_milli: (boolean, Default = True) Set False if you want the calculated RMSE value in decimals
    rmse: The calculated RMSE value
    """
    rmse = np.sqrt(np.mean((array_1 - array_2) ** 2))
    if rmse_in_milli == True:
        return rmse * 1000
    else:
        return rmse


def make_linearfit(
    workflow_name: str,
    delete_existing_savefiles=False,
    file_path: str = "mgca.pckl.tgz",
    compression: str | None = None,
    training_frac: float | int = 0.5,
    number_of_functions_per_element: int | None = 10,
    rcut: float | int = 6.0,
):

    wf = Workflow(workflow_name, delete_existing_savefiles=delete_existing_savefiles)
    if wf.has_saved_content():
        return wf

    # Workflow connections
    wf.load_dataset = ReadPickledDatasetAsDataframe(
        file_path=file_path, compression=compression
    )
    wf.split_dataset = SplitTrainingAndTesting(
        data_df=wf.load_dataset.outputs.df, training_frac=training_frac
    )
    wf.parameterize_potential = ParameterizePotentialConfig(
        number_of_functions_per_element=number_of_functions_per_element, rcut=rcut
    )
    wf.run_linear_fit = RunLinearFit(
        potential_config=wf.parameterize_potential,
        df_train=wf.split_dataset.outputs.df_training,
        df_test=wf.split_dataset.outputs.df_testing,
        verbose=False,
    )
    wf.save_potential = SavePotential(basis=wf.run_linear_fit.outputs.basis)
    wf.predict_energies_forces = PredictEnergiesAndForces(
        basis=wf.save_potential.outputs.basis,
        df_train=wf.split_dataset.outputs.df_training,
        df_test=wf.split_dataset.outputs.df_testing,
    )

    # Input mapping
    wf.inputs_map = {
        "run_linear_fit__verbose": "verbose",
        "save_potential__filename": "filename",
        "parameterize_potential__number_of_functions_per_element": "number_of_functions_per_element",
        "parameterize_potential__rcut": "rcut",
    }

    # Output maping
    wf.outputs_map = {
        "save_potential__yace_file_path": "yace_file_path",
        "predict_energies_forces__data_dict": "data_dict",
    }

    return wf


########################## PLOTTING NODES ##########################


# HISTOGRAM FOR ENERGY DISTRIBUTION
@as_function_node("plot")
def PlotEnergyHistogram(df: pd.DataFrame, bins: int = 100, log_scale: bool = True):

    # Calculate energy_per_atom
    df["energy_per_atom"] = df["energy_corrected"] / df["NUMBER_OF_ATOMS"]

    plt.hist(df["energy_per_atom"], bins=bins, log=log_scale)
    plt.ylabel("Count")
    plt.xlabel("Energy per atom (meV/atom)")
    return plt.show()


# HISTOGRAM FOR FORCE DISTRIBUTION
@as_function_node("plot")
def PlotForcesHistogram(df: pd.DataFrame, bins: int = 100, log_scale: bool = True):

    array = np.concatenate(df.forces.values).flatten()

    plt.hist(array, bins=bins, log=log_scale)
    plt.ylabel("Count")
    plt.xlabel(r"Force (eV/$\mathrm{\AA}$)")
    return plt.show()


@as_function_node("plot")
def PlotEnergyFittingCurve(data_dict: dict):

    fig, axe = plt.subplots()

    lims = [
        data_dict["reference_training_epa"].min(),
        data_dict["reference_training_epa"].max(),
    ]
    axe.plot(lims, lims, ls="--", color="C0")

    if "reference_testing_epa" in data_dict.keys():
        rmse_testing = _calc_rmse(
            data_dict["reference_testing_epa"], data_dict[f"predicted_testing_epa"]
        )
        axe.scatter(
            data_dict["reference_testing_epa"],
            data_dict["predicted_testing_epa"],
            color="black",
            s=30,
            marker="+",
            label=f"Testing RMSE = {rmse_testing:.2f} (meV/atom)",
        )

    rmse_training = _calc_rmse(
        data_dict["reference_training_epa"], data_dict["predicted_training_epa"]
    )
    axe.scatter(
        data_dict["reference_training_epa"],
        data_dict["predicted_training_epa"],
        color="C0",
        s=30,
        label=f"Training RMSE = {rmse_training:.2f} (meV/atom)",
    )

    axe.set_xlabel("DFT E (eV/atom)")
    axe.set_ylabel("Predicted E (eV/atom)")
    axe.set_title("Predicted Energy Vs Reference Energy")
    axe.legend()

    return plt.show()


@as_function_node("plot")
def PlotForcesFittingCurve(data_dict: dict):

    fig, axe = plt.subplots()

    lims = [
        data_dict["reference_training_fpa"].min(),
        data_dict["reference_training_fpa"].max(),
    ]
    axe.plot(lims, lims, ls="--", color=f"C1")

    if "reference_testing_epa" in data_dict.keys():
        rmse_testing = _calc_rmse(
            data_dict["reference_testing_fpa"], data_dict["predicted_testing_fpa"]
        )
        axe.scatter(
            data_dict["reference_testing_fpa"],
            data_dict["predicted_testing_fpa"],
            color="black",
            s=30,
            marker="+",
            label=f"Testing RMSE = {rmse_testing:.2f}" + r"(meV/$\AA$)",
        )

    rmse_training = _calc_rmse(
        data_dict["reference_training_fpa"], data_dict["predicted_training_fpa"]
    )
    axe.scatter(
        data_dict["reference_training_fpa"],
        data_dict["predicted_training_fpa"],
        color="C1",
        s=30,
        label=f"Training RMSE = {rmse_training:.2f}" + r"(meV/$\AA$)",
    )

    axe.set_xlabel("DFT $F_i$" + r"(eV/$\AA$)")
    axe.set_ylabel("Predicted $F_i$" + r"(eV/$\AA$)")
    axe.set_title("Predicted Force Vs Reference Force")
    axe.legend()

    return plt.show()


@as_function_node("design_matrix")
def DesignMatrix(
    df: pd.DataFrame,
    potential_config: PotentialConfig,
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

    from pyace.linearacefit import LinearACEDataset
    from pyace import create_multispecies_basis_config

    from pyiron_snippets.logger import logger

    logger.setLevel(30)

    elements_set = set()
    for atoms in df["ase_atoms"]:
        elements_set.update(atoms.get_chemical_symbols())

    elements = sorted(elements_set)
    potential_config.elements = elements
    potential_config_dict = potential_config.to_dict()

    bconf = create_multispecies_basis_config(potential_config_dict)

    ds = LinearACEDataset(bconf, df)
    ds.construct_design_matrix(verbose=verbose)
    return ds.design_matrix


@as_function_node("matrix")
def SliceArray(matrix, indices):
    return matrix[indices]


@as_function_node("vector")
def GetVector(
    df: pd.DataFrame,
    indices,
    scale_energy_per_atom: bool = False,
):
    import numpy as np

    vec = df.energy_corrected
    if scale_energy_per_atom:
        vec /= df.NUMBER_OF_ATOMS

    forces_vec = []
    for f in df.forces.apply(lambda x: x.flatten()):
        forces_vec += list(f)
    vec = np.append(vec, forces_vec)
    return vec[indices]


@as_function_node
def MinMaxIndices(
    df: pd.DataFrame,
    i_min: int = 0,
    i_max: int = None,
    energy_only: bool = False,
):
    num_structures = len(df)
    num_atoms = np.sum(df.NUMBER_OF_ATOMS)

    indices = np.arange(num_structures + 3 * num_atoms)
    if i_max is None or i_max == "":
        i_max = num_structures
    energies = indices[i_min:i_max]
    forces = indices[num_atoms + 3 * i_min : num_atoms + 3 * i_max]
    if energy_only:
        indices = energies
    else:
        indices = np.append(energies, forces, axis=0)
    return indices
