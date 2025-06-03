# Decision 25-003 -- The role of graphs in managing the flow of data into and out of their context

## Status

- [Adopted](https://github.com/JNmpi/pyiron_core/pull/58)
  - No implementations


## Context

- Graphs are allowed to hold other graphs as child nodes, i.e., sub-graphs are permitted
- This nesting is tracked by the parent-most graph in the "parent_id" columns
- Sub-graphs can be created by grouping existing nodes, or added ex-nihilo from a macro definition
- Input and Output of Graphs are represented by virtual child nodes (which are the equivalent of node ports)


## Decision

- It is the responsibility of graphs to manage the flow of all data between their (non-virtual) child nodes and non-child nodes


## Consequences

- The graph shall not hold edges between a (non-virtual) child of a particular subgraph and an external (non-child) node of that subgraph.
- If the user requests the formation of such an edge, it shall either... 
  - be denied (i.e., raise an error), or
  - (preferred) be coerced to pass through the graph's interface (i.e., to be transformed into a pair of edges passing through the graph's virtual node interface)
