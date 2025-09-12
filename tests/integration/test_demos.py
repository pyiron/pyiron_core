import pathlib
import unittest

import pyiron_core as pc

DEMOS_DIR = pathlib.Path(__file__).parent.parent.parent / "stored_workflows" / "demos"
ALL_DEMOS = list(f.stem for f in DEMOS_DIR.glob("*.json"))


class TestDemoWorkflows(unittest.TestCase):
    def test_loading(self):

        pf = pc.PyironFlow(
            ALL_DEMOS,
            workflow_path=DEMOS_DIR,
            load_from_compact=True,
        )
