# Decision 25-002 -- The role of collapsed and expanded in execution flow 

## Status

- [Adopted](https://github.com/JNmpi/pyiron_core/pull/57)
  - No implementations


## Context

- We have a parent-most graph tracking information about all nodes, including whether they are "expanded" or "collapsed" (i.e. not expanded), and whether they are in a group (have a "parent_id") or not (no "parent_id"/matches top-most graph)


## Decision

- The "expanded" column will not be used to compute execution flow, only (un)grouped status
- We will provide a graph-copying tool where all "expanded" groups are explicitly un-grouped in the copy


## Consequences

- The execution flow depends on a smaller set of variables
- We can easily modify execution flow via the GUI by invoking the above graph-copying tool and executing on the graph copy
