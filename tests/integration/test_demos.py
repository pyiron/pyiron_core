import contextlib
import pathlib
import unittest
import urllib

import pandas as pd
import pandas.testing as pdt

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
            # Instantiate the gui like a user
            pf = pc.PyironFlow(["landau"], workflow_path=DEMOS_DIR, load_from_compact=True)

            # Mock a "run" click
            pf.wf_widgets[0].on_value_change(
                {"new": "run: CalcPhaseDiagram", "old": None, "name": "manual"}
            )

            # The result gets passed directly to a display widget, and is hard to parse
            # As a fallback, inspect the underlying graph
            output = pf.wf_widgets[0].graph.nodes["CalcPhaseDiagram"].node.outputs.df.value
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
