# Getting Started

## Installation

```bash
git clone https://github.com/pyiron/pyiron_core.git
cd pyiron_core
conda env create -f project-env.yml   # or use pip install .
conda activate pyiron_core
npm install && npm run build           # build the React UI widgets
```


## Basic Python Usage

```python
import pyiron_core as pc

@pc.as_function_node("result")
def add_one(x: float) -> float:
    """Return *x* plus one."""
    return x + 1

wf = pc.Workflow("MyWorkflowName")
wf.add_one = add_one(x=2)
result = wf.run()
print(result)   # → 3
```

## Launch the Visual Editor (Jupyter)

```python
pc.PyironFlow([wf]).gui   # opens the React‑Flow UI in the current notebook cell
```

Explore the demo notebooks in `demo/` for more complex examples.
