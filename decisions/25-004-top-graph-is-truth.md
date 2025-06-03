# Decision 25-004 -- The top-most graph is the source of truth

## Status

- [Adopted](https://github.com/JNmpi/pyiron_core/pull/74)
  - No implementations


## Context

- A graph consists of nodes and edges
- Graphs may be nested, i.e. we allow sub-graphs
- A graph holding other graphs directly holds all the nodes and edges of its sub-graphs in its own nodes and edges


## Decision

- The top-level graph is the source of truth
- Only the top-level graph is editable
  - All other objects derived from the top-level graph (e.g., sub-graphs, representations for the GUI) are strictly non-editable views
- Views are always computed from or references to the current state of the top-level graph
- When a de-parented copy of a sub-graph is created, it becomes the top-level graph of its new context

## Consequences

- Any changes to the top-level graph are automatically reflected in derived views
- Explicit updates to sub-graphs or other representations are never required
- Operations in the GUI or other interfaces that provide the appearance of editing sub-graphs are merely acting as shortcuts to operations on the top-level graph itself.