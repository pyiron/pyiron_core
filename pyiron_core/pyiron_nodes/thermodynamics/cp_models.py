from dataclasses import dataclass, field

import numpy as np

from pyiron_core.pyiron_workflow import (
    Workflow,
    as_function_node,
    as_macro_node,
    as_out_dataclass_node,
)

kB_meV = 8.617333262145e-2  # Boltzmann's constant in eV/K


@as_function_node("slice")
def Slice(start: int = 0, stop: int = -1, step=1):
    return slice(start, stop, step)


# @as_inp_dataclass_node
@dataclass
class DataVector:
    label: str = ""
    value: list | np.ndarray = field(default_factory=lambda: np.array([]))
    unit: str = ""
    val_slice: slice = slice(0, -1, 1)


@as_out_dataclass_node
class GibbsData:
    temperature: DataVector = field(
        default_factory=lambda: DataVector("Temperature", unit="K")
    )
    gibbs_energy: DataVector = field(
        default_factory=lambda: DataVector("Gibbs Energy", unit="eV")
    )
    label: str = ""


@as_out_dataclass_node
class CpData:
    temperature: DataVector = field(
        default_factory=lambda: DataVector("Temperature", unit="K")
    )
    cp: DataVector = field(default_factory=lambda: DataVector("Cp", unit="eV/K"))
    label: str = ""


@as_function_node
def fit_G(
    data: GibbsData,
    degree: int = 4,
    T_start: int = None,
    T_stop: int = None,
    T_step: int = 1,
):
    if T_start is None:
        T_start = data.temperature.value.min()
    if T_stop is None:
        T_stop = data.temperature.value.max()
    num_steps = int((T_stop - T_start) / T_step)
    temperatures = np.linspace(T_start, T_stop, num_steps)

    coefficients_G = np.polyfit(data.temperature.value, data.gibbs_energy.value, degree)
    fit = np.poly1d(coefficients_G)(temperatures)
    return temperatures, fit


@as_function_node("enthalpy")
def compute_enthalpy(gibbs_energy, temperature, entropy):
    return gibbs_energy + kB_meV * temperature * entropy


@as_function_node
def compute_entropy(gibbs_energy, temperature):
    delta_T = np.gradient(temperature)
    delta_G = np.gradient(gibbs_energy)
    entropy = -delta_G / delta_T / kB_meV
    return entropy


@as_function_node
def compute_cp(enthalpy, temperature, label: str = "cp"):
    delta_T = np.gradient(temperature)
    delta_H = np.gradient(enthalpy)
    cp = delta_H / delta_T / kB_meV

    out = CpData().dataclass()
    out.temperature.value = temperature[:-2]
    out.cp.value = cp[:-2]
    out.label = "Cp"
    return out


@as_macro_node("cp")
def compute_cp_from_G(data: GibbsData, degree: int = 4):
    wf = Workflow("macro")
    wf.fit = fit_G(data, degree)
    wf.entropy = compute_entropy(wf.fit.outputs.fit, wf.fit.outputs.temperatures)
    wf.enthalpy = compute_enthalpy(
        wf.fit.outputs.fit, wf.fit.outputs.temperatures, wf.entropy
    )
    print(data, type(data))
    wf.cp = compute_cp(wf.enthalpy, wf.fit.outputs.temperatures)

    return wf.cp


@as_function_node("plot")
def DataPlot(
    data_1: GibbsData,
    data_2: GibbsData = None,
    data_3: GibbsData = None,
    title: str = None,
):
    from matplotlib import pyplot as plt

    if title is not None:
        plt.title(title)
    plt.plot(data_1.temperature.value, data_1.gibbs_energy.value, label=data_1.label)
    plt.xlabel = f"{data_1.temperature.label} [{data_1.temperature.unit}]"
    plt.ylabel = data_1.gibbs_energy.label

    if data_2 is not None:
        plt.plot(
            data_2.temperature.value, data_2.gibbs_energy.value, label=data_2.label
        )
    if data_3 is not None:
        plt.plot(
            data_3.temperature.value, data_3.gibbs_energy.value, label=data_3.label
        )

    plt.legend()
    plt.title(title)

    return plt.show()


@as_function_node("plot")
def DataPlotCp(
    data_1: GibbsData,
    data_2: GibbsData = None,
    data_3: GibbsData = None,
    title: str = None,
):
    from matplotlib import pyplot as plt

    if title is not None:
        plt.title(title)
    plt.plot(data_1.temperature.value, data_1.cp.value, label=data_1.label)
    plt.xlabel = f"{data_1.temperature.label} [{data_1.temperature.unit}]"
    plt.ylabel = data_1.cp.label

    if data_2 is not None:
        plt.plot(data_2.temperature.value, data_2.cp.value, label=data_2.label)
    if data_3 is not None:
        plt.plot(data_3.temperature.value, data_3.cp.value, label=data_3.label)

    plt.legend()
    plt.title(title)

    return plt.show()


@as_function_node
def LoadGibbsData(file_name: str, label: str = "", range: slice = None):
    import pylab as plb

    data = plb.loadtxt(file_name)

    out = GibbsData().dataclass()
    if label == "":
        out.label = file_name.split("/")[-1]
    else:
        out.label = label
    out.temperature.value = data.T[0]
    out.gibbs_energy.value = data.T[1]
    return out
