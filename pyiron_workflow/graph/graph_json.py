import json
import pathlib

from pyiron_workflow.graph.base import Graph


def _save_graph(
    graph: Graph,
    filename: str | pathlib.Path = None,
    workflow_dir: str = ".",
    overwrite: bool = False,
):
    if filename is None:
        filename = f"{graph.label}.json"

    if isinstance(filename, str):
        # check if filename has extension json, if not add it
        if not filename.endswith(".json"):
            filename = f"{filename}.json"

        filename = pathlib.Path(filename)

    file = pathlib.Path(workflow_dir) / filename
    if file.exists() and not overwrite:
        raise FileExistsError(
            f"File '{filename}' already exists in dir {workflow_dir}."
        )

    with open(file, "w") as f:
        f.write(json.dumps((graph.__getstate__()), indent=4))

    return True


def _load_graph(filename: str | pathlib.Path, workflow_dir: str = "."):
    # check if filename has extension json, if not add it
    if isinstance(filename, str):
        if not filename.endswith(".json"):
            filename = f"{filename}.json"

    if isinstance(filename, str):
        filename = pathlib.Path(filename)

    wf_file = workflow_dir / filename
    if not wf_file.exists():
        raise FileNotFoundError(f"File '{filename}' not found in dir {workflow_dir}.")

    with open(wf_file, "r") as f:
        graph = Graph().__setstate__(json.load(f))

    return graph
