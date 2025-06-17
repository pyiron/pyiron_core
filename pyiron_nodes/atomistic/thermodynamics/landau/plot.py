import numpy as np

import landau

from pyiron_workflow import as_function_node, as_macro_node, Workflow


@as_function_node(use_cache=False)
def TransitionTemperature(
    phase1,
    phase2,
    Tmin: float,
    Tmax: float,
    dmu: float = 0,
    plot: bool = True,
) -> float:
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np
    from IPython.display import display

    df = landau.calculate.calc_phase_diagram(
        [phase1, phase2], np.linspace(Tmin, Tmax), dmu, keep_unstable=True
    )
    try:
        fm, Tm = (
            df.query("border and T!=@Tmin and T!=@Tmax")[["f", "T"]].iloc[0].tolist()
        )
    except IndexError:
        display("Transition Point not found!")
        fm, Tm = np.nan, np.nan
    if plot:
        sns.lineplot(
            data=df,
            x="T",
            y="f",
            hue="phase",
            style="stable",
            style_order=[True, False],
        )
        plt.axvline(Tm, color="k", linestyle="dotted", alpha=0.5)
        plt.scatter(Tm, fm, marker="o", c="k", zorder=10)

        dfa = np.ptp(df["f"].dropna())
        dft = np.ptp(df["T"].dropna())
        plt.text(
            Tm + 0.05 * dft,
            fm + dfa * 0.1,
            rf"$T_m = {Tm:.0f}\,\mathrm{{K}}$",
            rotation="vertical",
            ha="center",
        )
        plt.xlabel("Temperature [K]")
        plt.ylabel("Free Energy [eV/atom]")
        plt.show()
    return Tm


def guess_mu_range(phases, Tmax, samples):
    """Guess chemical potential window from the ideal solution.

    Searches numerically for chemical potentials which stabilize
    concentrations close to 0 and 1 and then use the concentrations
    encountered along the way to numerically invert the c(mu) mapping.
    Using an even c grid with mu(c) then yields a decent sampling of mu
    space so that the final phase diagram is described everywhere equally.

    Args:
        phases: list of phases to consider
        Tmax: temperature at which to estimate
        samples: how many mu samples to return

    Returns:
        array of chemical potentials that likely cover the whole concentration space
    """

    import landau
    import scipy.optimize as so
    import scipy.interpolate as si
    import numpy as np

    # semigrand canonical "average" concentration
    # use this to avoid discontinuities and be phase agnostic
    def c(mu):
        phis = np.array([p.semigrand_potential(Tmax, mu) for p in phases])
        conc = np.array([p.concentration(Tmax, mu) for p in phases])
        phis -= phis.min()
        beta = 1 / (Tmax * 8.6e-5)
        prob = np.exp(-beta * (phis - conc * mu))
        prob /= prob.sum()
        return (prob * conc).sum()

    cc, mm = [], []
    mu0, mu1 = 0, 0
    while (ci := c(mu0)) > 0.001:
        cc.append(ci)
        mm.append(mu0)
        mu0 -= 0.05
    while (ci := c(mu1)) < 0.999:
        cc.append(ci)
        mm.append(mu1)
        mu1 += 0.05
    cc = np.array(cc)
    mm = np.array(mm)
    I = cc.argsort()
    cc = cc[I]
    mm = mm[I]
    return si.interp1d(cc, mm)(np.linspace(min(cc), max(cc), samples))


@as_function_node("phase_data")
def CalcPhaseDiagram(
    phases: list,
    temperatures: list[float] | np.ndarray,
    chemical_potentials: list[float] | np.ndarray | int = 100,
    refine: bool = True,
):
    """Calculate thermodynamic potentials and respective stable phases in a range of temperatures.

    The chemical potential range is chosen automatically to cover the full concentration space.

    Args:
        phases: list of phases to consider
        temperatures: temperature samples
        mu_samples: number of samples in chemical potential space
        refine (bool): add additional sampling points along exact phase transitions

    Returns:
        dataframe with phase data
    """
    import matplotlib.pyplot as plt
    import landau

    if isinstance(chemical_potentials, int):
        mus = guess_mu_range(phases, max(temperatures), chemical_potentials)
    else:
        mus = chemical_potentials
    df = landau.calculate.calc_phase_diagram(
        phases, np.asarray(temperatures), mus, refine=refine, keep_unstable=False
    )
    return df


@as_macro_node("phase_data")
def ComputePhaseDiagram(
    filename: str = "MgCaFreeEnergies.pckl.gz",
    T_min: int = 300,
    T_max: int = 1100,
    T_steps=20,
):
    import pyiron_nodes as pn
    from pyiron_nodes.atomistic.thermodynamics.landau.phases import PhasesFromDataFrame

    wf = Workflow("PhaseDiagram")
    wf.read_data = pn.dataframe.ReadDataFrame(filename=filename, compression="gzip")
    wf.phases_from_df = PhasesFromDataFrame(
        dataframe=wf.read_data
    )
    wf.temperatures = pn.math.Linspace(
        x_min=T_min, x_max=T_max, num_points=T_steps, endpoint=True
    )
    wf.calc_phase_diagram = pn.atomistic.thermodynamics.landau.plot.CalcPhaseDiagram(
        phases=wf.phases_from_df.outputs.phase_list,
        temperatures=wf.temperatures,
        refine=True,
    )
    return wf.calc_phase_diagram


@as_function_node("plot", use_cache=False)
def PlotConcPhaseDiagram(
    phase_data,
    plot_samples: bool = False,
    plot_isolines: bool = False,
    plot_tielines: bool = True,
    linephase_width: float = 0.01,
):
    """Plot a concentration-temperature phase diagram.

    phase_data should originate from CalcPhaseDiagram.

    Args:
        phases: list of phases to consider
        plot_samples (bool): overlay points where phase data has been sampled
        plot_isolines (bool): overlay lines of constance chemical potential
        plot_tielines (bool): add grey lines connecting triple points
        linephase_width (float): phases that have a solubility less than this
            will be plotted as a rectangle
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import landau

    landau.plot.plot_phase_diagram(
        phase_data.drop("refined", errors="ignore", axis="columns"),
        min_c_width=linephase_width,
    )
    if plot_samples:
        sns.scatterplot(data=phase_data, x="c", y="T", hue="phase", legend=False, s=1)
    if plot_isolines:
        sns.lineplot(
            data=phase_data.loc[np.isfinite(phase_data.mu)],
            x="c",
            y="T",
            hue="mu",
            units="phase",
            estimator=None,
            legend=False,
            sort=False,
        )
    if plot_tielines and "refined" in phase_data.columns:
        # hasn't made it upstream yet
        for T, dd in phase_data.query('refined=="delaunay-triple"').groupby("T"):
            plt.plot(dd.c, [T] * 3, c="k", alpha=0.5, zorder=-10)
    plt.xlabel("Concentration")
    plt.ylabel("Temperature [K]")
    return plt.show()


@as_function_node("plot", use_cache=False)
def PlotMuPhaseDiagram(phase_data):
    """Plot a chemical potential-temperature phase diagram.

    phase_data should originate from CalcPhaseDiagram.
    """
    import seaborn as sns
    import matplotlib.pyplot as plt

    border = None
    if "border" not in phase_data.columns:
        body = phase_data.query("not border")
    else:
        border = phase_data.query("border")
        body = phase_data.query("not border")
    sns.scatterplot(
        data=body,
        x="mu",
        y="T",
        hue="phase",
        s=5,
    )
    if border is not None:
        sns.scatterplot(
            data=border,
            x="mu",
            y="T",
            c="k",
            s=5,
        )
    plt.xlabel("Chemical Potential Difference [eV]")
    plt.ylabel("Temperature [K]")
    return plt.show()


@as_function_node("plot", use_cache=False)
def PlotIsotherms(phase_data):
    """Plot concentration isotherms in stable phases.

    phase_data should originate from CalcPhaseDiagram.
    """
    import seaborn as sns
    import matplotlib.pyplot as plt

    sns.lineplot(
        data=phase_data.query("stable"),
        x="mu",
        y="c",
        style="phase",
        hue="T",
    )
    plt.xlabel("Chemical Potential Difference [eV]")
    return plt.show()


@as_function_node("plot", use_cache=False)
def PlotPhiMuDiagram(phase_data):
    """Plot dependence of semigrand-potential on chemical potential in stable phases.

    phase_data should originate from CalcPhaseDiagram.
    """
    import seaborn as sns
    import matplotlib.pyplot as plt

    sns.lineplot(
        data=phase_data.query("stable"),
        x="mu",
        y="phi",
        style="phase",
        hue="T",
    )
    plt.xlabel("Chemical Potential Difference [eV]")
    plt.ylabel("Semigrand Potential [eV/atom]")
    return plt.show()


@as_function_node("plot", use_cache=False)
def CheckTemperatureInterpolation(
    phase: landau.phases.TemperatureDependentLinePhase,
    Tmin: float | None = None,
    Tmax: float | None = None,
):
    import numpy as np
    import matplotlib.pyplot as plt

    if Tmin is None:
        Tmin = np.min(phase.temperatures) * 0.9
    if Tmax is None:
        Tmax = np.max(phase.temperatures) * 1.1
    Ts = np.linspace(Tmin, Tmax, 50)
    (l,) = plt.plot(Ts, phase.line_free_energy(Ts), label="interpolation")
    # try to plot about 50 points
    n = max(int(len(phase.temperatures) // 50), 1)
    plt.scatter(
        phase.temperatures[::n],
        phase.free_energies[::n],
        c=l.get_color(),
        label="data",
    )

    return plt.show()
