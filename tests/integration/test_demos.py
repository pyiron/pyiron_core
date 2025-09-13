import contextlib
import pathlib
import unittest
import urllib

import numpy as np
import pandas as pd
import pandas.testing as pdt

try:
    import pyace
except ImportError:
    pyace = None

import pyiron_core as pc

DEMOS_DIR = pathlib.Path(__file__).parent.parent.parent / "stored_workflows" / "demos"
ALL_DEMOS = list(f.stem for f in DEMOS_DIR.glob("*.json"))


class TestDemoWorkflows(unittest.TestCase):
    @contextlib.contextmanager
    def _download_then_delete(self, url: str) -> None:
        fname = url.split("/")[-1]
        try:
            with urllib.request.urlopen(url) as response:
                with open(fname, "wb") as f:
                    f.write(response.read())
            yield None
        finally:
            pathlib.Path(fname).unlink()

    def _mock_run(self, wf_name: str, node_name: str, port_name: str):
        # Instantiate the gui like a user
        pf = pc.PyironFlow([wf_name], workflow_path=DEMOS_DIR, load_from_compact=True)
        # Mock a "run" click at the PyironFlowWidget level
        pf.wf_widgets[0].on_value_change(
            {"new": f"run: {node_name}", "old": None, "name": "mock"}
        )
        # The result gets passed directly to a display widget, and is hard to parse
        # As a fallback, inspect the underlying graph
        node_instance = pf.wf_widgets[0].graph.nodes[node_name].node
        output_value = getattr(node_instance.outputs, port_name).value
        # Return the target data
        return output_value

    def test_loading(self):
        pc.PyironFlow(
            ALL_DEMOS,
            workflow_path=DEMOS_DIR,
            load_from_compact=True,
        )

    def test_landau(self):
        with self._download_then_delete(
                "https://github.com/pyiron-workshop/DPG-tutorial-2025/raw/351eeca736cce45f9bc3bfca84ab05de049e38c2/data/MgCaFreeEnergies.pckl.gz"
        ) as _:
            output = self._mock_run("landau", "CalcPhaseDiagram", "df")
            tail = output.tail(3)

            # And compare it to results grabbed from a manual run where the plot is ok
            expected = pd.DataFrame({
                'T': [528.781028, 528.781028, 528.781028],
                'phase': ['fcc Ca', 'liquid', 'Mg$_2$Ca'],  # Note the newline in the third entry
                'phi': [-1.78757, -1.78757, -1.78757],
                'mu': [-0.316953, -0.316953, -0.316953],
                'c': [1.000000, 0.721217, 0.333333],
                'stable': [True, True, True],
                'border': [True, True, True],
                'refined': ['delaunay-triple', 'delaunay-triple', 'delaunay-triple'],
                'f': [-2.104523, -2.016162, -1.893221],
                'f_excess': [0.000000, 0.029454, 0.070434]
            })


            expected_sorted = expected.sort_values(by=list(expected.columns)).reset_index(drop=True)
            actual_sorted = tail.sort_values(by=list(tail.columns)).reset_index(drop=True)

            pdt.assert_frame_equal(
                expected_sorted,
                actual_sorted,
                rtol=1e-3,
                atol=1e-4,
                check_dtype=False
            )

    @unittest.skipIf(
        pyace is None, "pyace not available -- skipping linearfit test"
    )
    def test_linearfit(self):
        with self._download_then_delete(
                "https://github.com/pyiron-workshop/DPG-tutorial-2025/raw/351eeca736cce45f9bc3bfca84ab05de049e38c2/data/mgca.pckl.tgz"
        ) as _:
            output = self._mock_run("linearfit", "PredictEnergiesAndForces", "data_dict")

        reference = {  # Last items of each entry from a manual run
            "reference_training_epa": np.array([-1.6219105069933333, -1.6426530701, -1.4760515700066668, -1.449002318375, -1.204447103705]),
            "reference_training_fpa": np.array([-0.00052099, -3.075e-05, 5.55e-06, 0.00052099, 3.075e-05]),
            "predicted_training_epa": np.array([-1.782124353671963, -1.709263676346597, -1.6974051358740896, -1.6446346861135153, -1.1999858821779013]),
            "predicted_training_fpa": np.array([-0.0006600464048357077, -5.380634712150788e-05, -4.127741166488192e-06, 0.0006600464048357077, 5.38063471215287e-05]),
            "reference_testing_epa": np.array([-0.48387109904111114, -0.3011008274525, -0.8320895942100001, -0.257647389155, -0.27296485680125]),
            "reference_testing_fpa": np.array([0.06379655, -1.23912384, -0.05524943, -0.03189827, -1.23912384]),
            "predicted_testing_epa": np.array([-0.2911790725598916, -0.23546299226525025, -0.7422808416362213, -0.16393019733556505, -0.09089519915920707]),
            "predicted_testing_fpa": np.array([-0.026306930659412288, -1.0806129186000701, 0.02278246992192133, 0.013153465125682134, -1.0806129185952347]),
        }

        for key, data in output.items():
            expected = reference[key]
            actual = data[-len(expected):]
            with self.subTest(key):
                self.assertTrue(
                    np.allclose(expected, actual),
                    msg=f"expected {expected}\ngot {actual}"
                )
