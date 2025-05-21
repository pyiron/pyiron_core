# Decision 25-002 -- Collapsed and expanded do not impact execution representations

## Status

- [Adopted](https://github.com/JNmpi/pyiron_core/pull/57)
  - No implementations


## Context

- We have a parent-most graph tracking information about all nodes, including whether they are "expanded" or "collapsed" (i.e. not expanded)


## Decision

- The boolean "expanded" data should impact only the visual representation, and have no impact on execution representations
- The "expanded" data should be retained in the graph table, not shunted to the GUI, in case we want to use it to modify group status in graph copies


## Consequences

- The scope of use for "expanded" is concretely narrowed to visualization
- Impacts of the "expanded" state on execution flows can be treated as bugs
