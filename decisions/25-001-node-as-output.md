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

- All nodes will be assigned a regular output port carrying that node's node data


## Consequences

- Improved symmetry between subgraph nodes and other nodes
- Improved clarity between collapsed and expanded views of subgraphs
  - I.e., node data always comes from a port on the node to which that data belongs, never from a different node
