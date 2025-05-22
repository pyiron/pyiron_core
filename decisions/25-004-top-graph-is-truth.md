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
  - We will not allow the top-level graph to become invalid due to changes made to a sub-graph
- When a de-parented copy of a sub-graph is created, it becomes the top-level graph of its new context

## Consequences

- Changes to the top-level graph should be propagated to relevant sub-graphs
  - I.e., after adding an edge in the top-level graph that connects two nodes in a sub-graph, this edge should be visible when viewing the edges of that sub-graph
- Either changes to sub-graphs must be propagated back up to the top-most graph, or they must be disallowed
  - For the time being, we opted to disallow modifications to nodes and edges in graphs that have a `parent_id`
- Syntactic sugar (either text-based or graphical) that gives the appearance of transiently violating this principle is not forbidden
  - E.g., "Editing" a node in the GUI may _appear_ to be making adding connection in a sub-graph, but might simply be a shortcut to modifications in the top-most graph