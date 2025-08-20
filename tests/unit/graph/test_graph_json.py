import glob
import pathlib
import shutil
import unittest

from pyiron_workflow.graph import graph_json

class TestGraphJson(unittest.TestCase):
    def test_compact_failure(self):
        """
        Demonstrates the failure of compact storage for certain workflows in our corpus.
        """

        storage_directory = pathlib.Path(__file__).parent.parent.parent.parent / "stored_workflows"
        storage_files = glob.glob(str(storage_directory / "*.json"))

        target_dir = pathlib.Path("compact_failure")
        target_dir.mkdir(exist_ok=True)
        skip = [
            "calphy2.json",  # Never loaded
        ]

        for filepath in storage_files:
            name = filepath.rsplit("/", 1)[-1]
            if name in skip:
                continue
            g = graph_json._load_graph(filename=name, workflow_dir=storage_directory, compact=False)
            graph_json._save_graph(g, filename=name, workflow_dir=target_dir, compact=True)
            try:
                graph_json._load_graph(filename=name, workflow_dir=target_dir, compact=True)
                print("Loaded", name)
            except Exception as e:
                print(f"Failed to load {name}: {type(e)}: {e}")

        shutil.rmtree(target_dir)