from pyiron_core.pyiron_workflow import as_function_node, Node, as_out_dataclass_node
from dataclasses import field
from typing import Optional
import numpy as np


@as_out_dataclass_node
class OutputResistanceMeasurement:
    coordinates: list | np.ndarray = field(default_factory=lambda: [])
    measurements: list | np.ndarray = field(default_factory=lambda: [])
    index: int = 0
    value: float = 0


@as_function_node
def MeasureResistance(index: int = 0, sample: str = "demo", n_radius: int = 5):
    import numpy as np

    resistance = OutputResistanceMeasurement().dataclass()

    for i in range(-n_radius, n_radius):
        for j in range(-n_radius, n_radius):
            if i**2 + j**2 < n_radius**2:
                resistance.measurements.append(i * 0.1 + np.cos(j/n_radius))
                resistance.coordinates.append((i, j))

    resistance.index = index
    resistance.value = resistance.measurements[index]

    return resistance


@as_function_node
def OptimizeMeasurements(
    experiment: Node, method: str = "random", max_iterations: int = 100
):
    import numpy as np

    data = experiment.run()
    mae = []
    for i in range(max_iterations):
        fit = data.measurements - np.random.rand(len(data.measurements)) * (
            0.01 + np.exp(-i / 10)
        )
        mae.append(np.var(data.measurements - fit))

    return fit, mae


@as_function_node("plot")
def PlotData(coordinates, values, size: int = 1):
    import numpy as np
    import matplotlib.pylab as plt

    x, y = np.array(coordinates).T
    plt.scatter(x, y, c=values, s=size)

    return plt.show()
