# Overview

# Transient technical details

This project amalgamates multiple sources, including modified versions of [`pyironflow`](https://github.com/pyiron/pyironFlow/) and [`pyiron_nodes`](https://github.com/pyiron/pyiron_nodes).
Python requirements across all components are collected in `project.env`.
With a suitable python environment in place, generate a development build of the Javascript components with `npm install; npm run dev`.


# Transplanted README from the notebooks directory

This is a directy copy-paste. 
This content should be reviewed for completion, and any outstanding issues still desired should be migrated to repo issues.

## TODO list

- introduce a copy for nodes 
- copy node in Node input arguments / or update node input
- allow to spread input of a macro node over several child nodes
- don't show a node input type as editable box in Gui
- add save and load! 
- check whether we need to store both node and graph in GraphNode.__setstate__ 
- deepcopy fails to correctly reproduce the node input (copy_graph works), make sure that that __getstate__ works down to the node level!!



## Concept

### Graph related functions

This section contains functions related to the manipulation and management of graph structures within the application.

### Graph related functions

- **GraphNode**
#### Build graph
This section covers functions used to construct and modify the graph structure, including adding and removing nodes and edges.

#### Build graph
- copy_graph
- add_node
- add_edge
- remove_node
- remove_edge
- remove_node_with_reconnected_edges

- get_unique_label
- _update_target_port 

#### Transfer graph to workflow or code
- **get_wf_from_graph**
- **get_code_from_graph**

#### Collapse and expansion of nodes
Build a full graph with all nodes and virtual nodes for the input and output ports of macro nodes. Extract from the full graph the actual state of collapsed/expanded macro nodes.
- get_updated_graph (call after node expansion/collaps)
- expand_node
- collapse_node
- _mark_node_as_collapsed
- _mark_node_as_expanded
- _get_active_nodes (all nodes that are not inside a collapsed node)
- _get_active_edges
- _rewire_edge
- _remove_virtual_nodes
- _remove_virtual_edges
- _remove_edges_to_hidden_nodes


#### Graph to node
For virtual nodes that are not stored in a module but only exist as a subgraph
- **graph_to_node**
- get_non_default_input
- get_unconnected_ports
- get_externally_connected_input_ports
- get_unconnected_input_ports
- get_inputs_of_graph
- get_node_input_port
- get_outputs_of_graph

- is_port_external_to_graph

#### Gui related graph functions
This section contains functions that handle the graphical user interface (GUI) representation of the graph, converting graph structures into GUI elements.
- **GuiGraph** (graph -> gui)
- _nodes_to_gui
- get_child_dict
- _gui_children
- _graph_to_gui


## Workflow related functions

### transfer
- get_full_graph_from_wf