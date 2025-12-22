# Graph Engine

The workflow is represented as a **directed acyclic graph (DAG)** where nodes are vertices and edges describe data flow. The core implementation resides in `pyiron_core/pyiron_workflow/graph/`.

## Core Classes
* `BaseNode` – wraps a decorated function, stores input/output specs and cached results.
* `BaseEdge` – connects an output port of one node to an input port of another.
* `Graph` – container that holds nodes, edges and provides serialization to JSON.

### Creating a Graph Programmatically

```python
wf = pc.SimpleWorkflow()
wf.add_node(add_one, inputs={"x": 5})
wf.add_node(multiply, inputs={"x": wf.last_output, "y": 2})
```

### Execution & Caching
When `wf.run()` is called the engine:
1. Topologically sorts the DAG.
2. Computes a **hash** from input data and the graph topology.
3. Looks up cached results – if a sub‑graph with the same hash exists, it is skipped.

The graph can be exported/imported via JSON (`graph_json.py`). This is used by the visual editor to persist the layout.
