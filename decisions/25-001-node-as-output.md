# Decision 25-001 -- Return node objects themselves as output

## Status

- [Adopted](https://github.com/JNmpi/pyiron_core/pull/56)
  - No implementations


## Context

- Functional programming for graphs requires passing an entire node along an edge
- The current implementation leverages any regular data output to carry node information
  - Interpretation as a node relies on the type hinting of the receiver
- Subgraph nodes formed by creating a group currently have no data output
  - Their interface is managed instead by "virtual" nodes
- In the functional programming context, node data for a subgraph is carried by a connection to a virtual node
- This leads to different GUI edge representations between "collapsed" and "expanded" subgraphs


## Decision

- Interpreting a node-as-input will remain be the responsibility of the receiver
  - Interpretation of node input for functional programming will be controlled by type-hinting the receiving location as node-type
  - In the absence of a node-type type hint, the receiver will attempt to interpret the node object as a reference to that node's output, failing if there is more than out output to choose from
- A port-like element will be added to the GUI to expose a node-object handle for creating functional programming connections
- The data structure for defining an edge will be expanded from a four-tuple of (source, target)x(source handle, target handle) to also allow three-tuple of (source, target, target handle)


## Consequences

- GUI edge representations for groups being used as functional input always go from the group's "node" graphical element to the receiving port graphical element, completely independent of the group's collapsed/expanded state