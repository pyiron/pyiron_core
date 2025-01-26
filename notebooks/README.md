TODO list

- introduce a copy for nodes 
- copy node in Node input arguments / or update node input
- allow to spread input of a macro node over several child nodes
- don't show a node input type as editable box in Gui
- add save and load! 
- check whether we need to store both node and graph in GraphNode.__setstate__ 
- deepcopy fails to correctly reproduce the node input (copy_graph works), make sure that that __getstate__ works down to the node level!!