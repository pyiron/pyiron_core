# Decision 25-002 -- Collapsed and expanded do not impact execution representations

## Status

- [Adopted](https://github.com/JNmpi/pyiron_core/pull/57)
  - No implementations


## Context

- We have a parent-most graph tracking information about all nodes, including whether they are "expanded" or "collapsed" (i.e. not expanded), and whether they are in a group (have a "parent_id") or not (no "parent_id"/matches top-most graph)


## Decision

- The boolean "expanded" data should impact only the visual representation, and have no impact on execution flow
- The "expanded" data should be retained in the graph table, not shunted to the GUI, so that we can use it to modify group status in graph copies and thus impact execution flow


## Consequences

- The scope of use for "expanded" is concretely narrowed to visualization
- Impacts of the "expanded" state on execution flows can be treated as bugs
