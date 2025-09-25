from typing import Literal
from warnings import warn

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import landau
from pyiron_core.pyiron_workflow import Workflow, as_function_node, as_macro_node


def plot_phase_diagram(
    df,
    alpha=0.1,
    element=None,
    min_c_width=5e-3,
    color_override: dict[str, str] = {},
    tielines=False,
    poly_method: Literal["concave", "segments"] = "concave",
    ax=None,
):
    """
    Plot a concentration-temperature phase diagram.

    Parameters
    ----------
    df : pandas.DataFrame
        Stable phase diagram data.
    alpha : float
        Alpha parameter for concave hull construction.
    element : str or None
        If given, plot concentration axis with element name.
    min_c_width : float
        Minimum concentration width for phase polygon display.
    color_override : dict[str, str]
        Mapping of phase name to color hex code.
    tielines : bool
        Whether to draw tielines between phases.
    poly_method : {"concave", "segments"}
        Method to construct phase polygons.
    ax : matplotlib.axes.Axes or None
        If provided, plot will be drawn in this axis. Otherwise, a new figure is created.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object containing the plot.
    """
    from landau.plot import cluster_phase, make_concave_poly, make_poly

    df = df.query("stable").copy()

    # Create fig/ax if not provided
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    # The default color map
    color_map = dict(zip(df.phase.unique(), sns.palettes.SEABORN_PALETTES["pastel"]))
    color_override = {p: c for p, c in color_override.items() if p in color_map}

    duplicates_map = {c: color_map[o] for o, c in color_override.items()}
    diff = {k: duplicates_map[c] for k, c in color_map.items() if c in duplicates_map}
    color_map.update(diff | color_override)

    # Cluster the phase data
    df = cluster_phase(df)
    if (df.phase_unit == -1).any():
        warn("Clustering of phase points failed for some points, dropping them.")
        df = df.query("phase_unit >= 0")

    # Polygon construction
    if "refined" in df.columns and poly_method == "segments":
        df.loc[:, "phase"] = df.phase_id
        tdf = landau.plot.get_transitions(df)
        tdf["phase_unit"] = tdf.phase.str.rsplit("_", n=1).map(lambda x: int(x[1]))
        tdf["phase"] = tdf.phase.str.rsplit("_", n=1).map(lambda x: x[0])
        polys = tdf.groupby(["phase", "phase_unit"]).apply(
            make_poly, min_c_width=min_c_width
        )
    else:
        polys = (
            df.groupby(["phase", "phase_unit"])
            .apply(make_concave_poly, alpha=alpha, min_c_width=min_c_width)
            .dropna()
        )

    # Draw polygons
    for i, (phase, p) in enumerate(polys.items()):
        p.zorder = 1 / p.get_extents().size.prod()
        rep = phase[1] if isinstance(phase, tuple) else 0
        phase_name = phase[0] if isinstance(phase, tuple) else phase
        p.set_color(color_map[phase_name])
        p.set_edgecolor("k")
        p.set_label(phase_name + "'" * rep)
        ax.add_patch(p)

    # Tielines
    if tielines:
        if "refined" in df.columns:
            tdf = landau.plot.get_transitions(df)

            def plot_tie(dd):
                Tmin = dd["T"].min()
                Tmax = dd["T"].max()
                di = dd.query("T==@Tmin")
                da = dd.query("T==@Tmax")
                if len(dd.phase.unique()) in [1, 2]:
                    return
                ax.hlines(
                    Tmin, di.c.min(), di.c.max(), color="k", zorder=-2, alpha=0.5, lw=4
                )
                if Tmin != Tmax:
                    ax.hlines(
                        Tmax,
                        da.c.min(),
                        da.c.max(),
                        color="k",
                        zorder=-2,
                        alpha=0.5,
                        lw=4,
                    )

            tdf.groupby("border_segment").apply(plot_tie)
        else:
            chg = df.groupby("T").size().diff()
            T_tie = chg.loc[chg != 0].index[1:]

            def plot_tie(dd):
                if dd["T"].iloc[0].round(3) not in T_tie.round(3):
                    return
                if len(dd) != 2:
                    return
                cl, cr = sorted(dd.c)
                ax.plot([cl, cr], dd["T"], color="k", zorder=-2, alpha=0.5, lw=4)

            df.groupby(["T", "mu"]).apply(plot_tie)

    # Axis labels and limits
    ax.set_xlim(0, 1)
    ax.set_ylim(df["T"].min(), df["T"].max())
    ax.legend(ncols=2)
    if element is not None:
        ax.set_xlabel(rf"$c_\mathrm{{{element}}}$")
    else:
        ax.set_xlabel("$c$")
    ax.set_ylabel("$T$ [K]")

    return fig


@as_function_node(use_cache=False)
def TransitionTemperature(
    phase1,
    phase2,
    Tmin: float,
    Tmax: float,
    dmu: float = 0,
    plot: bool = True,
) -> float:
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
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

    import numpy as np
    import scipy.interpolate as si

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
def ComputPhaseDiagram(
    filename: str = "MgCaFreeEnergies.pckl.gz",
    T_min: int = 300,
    T_max: int = 1100,
    T_steps=20,
):
    import pyiron_core.pyiron_nodes as pn

    wf = Workflow("PhaseDiagram")
    wf.read_data = pn.utilities.ReadDataFrame(filename=filename, compression="gzip")
    wf.phases_from_df = pn.atomistic.thermodynamics.landau.phases.PhasesFromDataFrame(
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
    """
    Plot a concentration-temperature phase diagram.

    Args:
        phase_data: DataFrame from CalcPhaseDiagram
        plot_samples (bool): overlay points where phase data has been sampled
        plot_isolines (bool): overlay lines of constant chemical potential
        plot_tielines (bool): add grey lines connecting triple points
        linephase_width (float): phases that have a solubility less than this
            will be plotted as a rectangle
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns

    # Create a clean figure and axis
    fig, ax = plt.subplots()

    # Use specified axis for plotting
    plot_phase_diagram(
        phase_data.drop("refined", errors="ignore", axis="columns"),
        min_c_width=linephase_width,
        ax=ax,
    )

    if plot_samples:
        sns.scatterplot(
            data=phase_data, x="c", y="T", hue="phase", legend=False, s=1, ax=ax
        )

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
            ax=ax,
        )

    if plot_tielines and "refined" in phase_data.columns:
        for T, dd in phase_data.query('refined=="delaunay-triple"').groupby("T"):
            ax.plot(dd.c, [T] * 3, c="k", alpha=0.5, zorder=-10)

    ax.set_xlabel("Concentration")
    ax.set_ylabel("Temperature [K]")

    # Return the figure so IPython.display can render it
    return fig


@as_function_node("plot", use_cache=False)
def PlotMuPhaseDiagram(phase_data):
    """Plot a chemical potential-temperature phase diagram.

    phase_data should originate from CalcPhaseDiagram.
    Returns a matplotlib Figure object.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, ax = plt.subplots()

    border = None
    if "border" not in phase_data.columns:
        body = phase_data.query("not border")
    else:
        border = phase_data.query("border")
        body = phase_data.query("not border")

    sns.scatterplot(data=body, x="mu", y="T", hue="phase", s=5, ax=ax)

    if border is not None:
        sns.scatterplot(data=border, x="mu", y="T", c="k", s=5, ax=ax)

    ax.set_xlabel("Chemical Potential Difference [eV]")
    ax.set_ylabel("Temperature [K]")
    return fig


@as_function_node("plot", use_cache=False)
def PlotIsotherms(phase_data):
    """Plot concentration isotherms in stable phases.

    phase_data should originate from CalcPhaseDiagram.
    Returns a matplotlib Figure object.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, ax = plt.subplots()

    sns.lineplot(
        data=phase_data.query("stable"), x="mu", y="c", style="phase", hue="T", ax=ax
    )

    ax.set_xlabel("Chemical Potential Difference [eV]")
    ax.set_ylabel("Concentration")
    return fig


@as_function_node("plot", use_cache=False)
def PlotPhiMuDiagram(phase_data):
    """Plot dependence of semigrand-potential on chemical potential in stable phases.

    phase_data should originate from CalcPhaseDiagram.
    Returns a matplotlib Figure object.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, ax = plt.subplots()

    sns.lineplot(
        data=phase_data.query("stable"), x="mu", y="phi", style="phase", hue="T", ax=ax
    )

    ax.set_xlabel("Chemical Potential Difference [eV]")
    ax.set_ylabel("Semigrand Potential [eV/atom]")
    return fig


@as_function_node("plot", use_cache=False)
def CheckTemperatureInterpolation(
    phase: "landau.phases.TemperatureDependentLinePhase",
    Tmin: float | None = None,
    Tmax: float | None = None,
):
    """Check and visualize temperature interpolation of line phase free energies.
    Returns a matplotlib Figure object.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    if Tmin is None:
        Tmin = np.min(phase.temperatures) * 0.9
    if Tmax is None:
        Tmax = np.max(phase.temperatures) * 1.1

    fig, ax = plt.subplots()

    Ts = np.linspace(Tmin, Tmax, 50)
    (l,) = ax.plot(Ts, phase.line_free_energy(Ts), label="interpolation")

    # Try to plot about 50 points
    n = max(int(len(phase.temperatures) // 50), 1)
    ax.scatter(
        phase.temperatures[::n],
        phase.free_energies[::n],
        c=l.get_color(),
        label="data",
    )

    ax.set_xlabel("Temperature [K]")
    ax.set_ylabel("Free Energy")
    ax.legend()

    return fig
