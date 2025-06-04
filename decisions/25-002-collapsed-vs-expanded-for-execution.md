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

- We can ignore data in the "expanded" column when computing execution flows for a given graph
- We have a path to modify and represent modifications to the execution flow in the GUI by expanding/collapsing groups and using copies prior to execution
