/**
 * Author: Joerg Neugebauer
 * Copyright: Copyright 2024, Max-Planck-Institut for Sustainable Materials GmbH - Computational Materials Design (CM) Department
 * Version: 0.2
 * Maintainer:
 * Email:
 * Status: development
 * Date: Aug 1, 2024
 */

import React, {
  useCallback,
  useState,
  createContext,
  useSelection,
} from "react";
import { createRender, useModel } from "@anywidget/react";
import { getLayoutedNodes } from "./useElkLayout";
// import ELK from "elkjs/lib/elk.bundled.js";
import {
  ReactFlow,
  ReactFlowProvider,
  Controls,
  ControlButton,
  MiniMap,
  Background,
  applyEdgeChanges,
  applyNodeChanges,
  addEdge,
  Panel,
  useOnSelectionChange,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import {
  SymbolIcon,
  UploadIcon,
  DownloadIcon,
  Cross1Icon,
  Pencil1Icon,
  GroupIcon,
} from "@radix-ui/react-icons";

import TextUpdaterNode from "./TextUpdaterNode.jsx";
import CustomNode from "./CustomNode.jsx";

import "./text-updater-node.css";

const rfStyle = {
  //backgroundColor: '#B8CEFF',
  backgroundColor: "#dce1ea",
  //backgroundColor: 'white',
};

export const UpdateDataContext = createContext(null);
let globalStatus = "initial";
// const nodeTypes = { textUpdater: TextUpdaterNode, customNode: CustomNode };

const render = createRender(() => {
  console.log("rendering");
  const model = useModel();

  //   const initialNodes = JSON.parse(model.get("nodes"));
  //   const initialEdges = JSON.parse(model.get("edges"));
  const initialNodes = [];
  const initialEdges = [];
  console.log("initialData: ", initialNodes, initialEdges);

  const [nodes, setNodes] = useState(initialNodes);
  const [edges, setEdges] = useState(initialEdges);

  const [selectedNodes, setSelectedNodes] = useState([]);

  const nodeTypes = {
    textUpdater: TextUpdaterNode,
    customNode: CustomNode,
  };

  const updateData = (nodeLabel, handleIndex, newValue) => {
    const var_data = { label: nodeLabel, handle: handleIndex, value: newValue };
    model.set("commands", `change_node_value: ` + JSON.stringify(var_data));
    model.save_changes();
    console.log("updateData: ", nodeLabel, handleIndex, newValue);

    setNodes((prevNodes) =>
      prevNodes.map((node, idx) => {
        // console.log('updatedDataNodes: ', nodeLabel, handleIndex, newValue, node.id);
        if (node.id !== nodeLabel) {
          return node;
        }

        // This line assumes that node.data.target_values is an array
        const updatedTargetValues = [...node.data.target_values];
        updatedTargetValues[handleIndex] = newValue;
        console.log("updatedData2: ", updatedTargetValues);

        return {
          ...node,
          data: {
            ...node.data,
            target_values: updatedTargetValues,
          },
        };
      })
    );
  };

  //   // for test only, can be later removed
  //   useEffect(() => {
  //     console.log("nodes_test:", nodes);
  //     model.set("nodes", JSON.stringify(nodes)); // TODO: maybe better do it via command changeValue(nodeID, handleID, value)
  //     model.save_changes();
  //   }, [nodes]);

  model.on("change:nodes", () => {
    const new_nodes = model.get("nodes");
    console.log("load nodes: ", JSON.parse(new_nodes));
    setNodes(JSON.parse(new_nodes));
  });

  model.on("change:edges", () => {
    const new_edges = model.get("edges");
    console.log("load edges: ", JSON.parse(new_edges));
    setEdges(JSON.parse(new_edges));
  });

  model.on("change:mydata", () => {
    // it appears that there is a bug in the interplay between traitlets and setNodes and setEdges.
    // Calling setNodes and setEdges triggers a change event, which in turn triggers a new mydata event.
    // This leads to an increasing number of mydata events, which probably causes a slow down and eventually a crash.
    // To avoid this partly, we need to check if we are already running and return if this is the case.
    // This avoids repeating the expensive layouting process, but still triggers the creation of many objects.
    // This is not a solution, but a workaround. A possible way would be to make the data communication via files.
    // If this problem becomes a practical issue, we need to consider/switch to a file based communication.
    if (globalStatus === "running") {
      console.log("data changed, but we are running");
      return;
    }
    globalStatus = "running";

    console.log("data changed");
    const new_data = model.get("mydata");
    // console.log("load data: ", new_data);
    const data = JSON.parse(new_data);

    console.log("load data: ", data);

    const new_nodes = data.nodes;
    const new_edges = data.edges;
    const new_graph = data.graph;

    // give start and end time for the layouting

    console.log("start: ", new Date().getTime());
    getLayoutedNodes(new_nodes, new_edges, new_graph).then((layoutedNodes) => {
      console.log("layoutedNodes: ", layoutedNodes);
      setNodes(layoutedNodes);
      setEdges(new_edges);
      console.log("finished: ", new Date().getTime());
      globalStatus = "finished";
    });
    // since this is async, we need to wait for the layouted nodes
    // it appears that we never get here
  });

  model.on("change:commands", () => {
    const new_commands = model.get("commands");
    console.log("load commands: ", new_commands);
  });

  const onNodesChange = useCallback(
    (changes) => {
      setNodes((nds) => {
        const new_nodes = applyNodeChanges(changes, nds);
        // clear selectedNodes
        setSelectedNodes([]);
        for (const i in changes) {
          if (Object.hasOwn(changes[i], "selected")) {
            if (changes[i].selected) {
              for (const k in new_nodes) {
                if (new_nodes[k].id == changes[i].id) {
                  selectedNodes.push(new_nodes[k]);
                }
              }
            } else {
              for (const j in selectedNodes) {
                if (selectedNodes[j].id == changes[i].id) {
                  selectedNodes.splice(j, 1);
                }
              }
            }
          }
        }
        console.log("onNodesChange: ", changes, new_nodes);
        console.log("selectedNodes: ", selectedNodes);
        // convert list of selectedNodes to a string of comma separated node ids
        // and submit it to GUI
        let selectedNodesStr = selectedNodes.map((node) => node.id).join(",");
        model.set(
          "commands",
          `selected_nodes: ${selectedNodesStr} - ${new Date().getTime()}`
        );
        model.save_changes();

        model.set("nodes", JSON.stringify(new_nodes));
        model.save_changes();
        return new_nodes;
      });
    },
    [setNodes]
  );

  const onEdgesChange = useCallback(
    (changes) => {
      setEdges((eds) => {
        console.log("onEdgesChange: ", changes);
        const new_edges = applyEdgeChanges(changes, eds);
        model.set("edges", JSON.stringify(new_edges));
        model.save_changes();
        return new_edges;
      });
    },
    [setEdges]
  );

  const onConnect = useCallback(
    (params) => {
      setEdges((eds) => {
        const new_edges = addEdge(params, eds);
        model.set("edges", JSON.stringify(new_edges));
        model.save_changes();
        console.log("onConnect: ", params);
        model.set(
          "commands",
          `add_edge: ${params.source}/${params.sourceHandle} > ${
            params.target
          }/${params.targetHandle} - ${new Date().getTime()}`
        );
        model.save_changes();
        return new_edges;
      });
    },
    [setEdges]
  );

  const deleteNode = (id) => {
    // direct output of node to output widget
    console.log("output: ", id);
    if (model) {
      model.set("commands", `delete_node: ${id} - ${new Date().getTime()}`);
      model.save_changes();
    } else {
      console.error("model is undefined");
    }
  };

  const onNodesDelete = useCallback((deleted) => {
    console.log("onNodesDelete: ", deleted);
    deleteNode(deleted[0].id);
  });

  const deleteEdge = (params) => {
    console.log("delete edge: ", params);
    model.set(
      "commands",
      `delete_edge: ${params.source}/${params.sourceHandle} > ${
        params.target
      }/${params.targetHandle} - ${new Date().getTime()}`
    );
    model.save_changes();
  };

  const onEdgesDelete = useCallback((deleted) => {
    console.log("onEdgesDelete: ", deleted);
    deleteEdge(deleted[0]);
  });

  const refreshGraphView = (id) => {
    // refresh data on python side
    console.log("refreshGraphView: ", id);
    model.set(
      "commands",
      `refreshGraphView: __global__ - ${new Date().getTime()}`
    );
    model.save_changes();
  };

  const saveFlow = (id) => {
    // save data on python side
    console.log("saveFlow: ", id);
    model.set("commands", `saveFlow: __global__ - ${new Date().getTime()}`);
    model.save_changes();
  };

  const restoreFlow = (id) => {
    // restore data on python side
    console.log("restoreFlow: ", id);
    model.set("commands", `restoreFlow: __global__ - ${new Date().getTime()}`);
    model.save_changes();
  };

  const clearFlow = (id) => {
    // delete data on python side
    console.log("clearFlow: ", id);
    model.set("commands", `clearFlow: __global__ - ${new Date().getTime()}`);
    model.save_changes();
  };

  const groupNodes = (id) => {
    // group nodes
    console.log("groupNodes: ", id);
    // convert list of selectedNodes to a string of comma separated node ids
    // let selectedNodesStr = selectedNodes.map((node) => node.id).join(",");
    // console.log("selectedNodesStr: ", selectedNodesStr, selectedNodes);
    // model.set(
    //   "commands",
    //   `group_nodes: ${selectedNodesStr} - ${new Date().getTime()}`
    // );
    // model.save_changes();
    model.set(
      "commands",
      `groupSelectedNodes: __global__ - ${new Date().getTime()}`
    );
    model.save_changes();
  };

  const renameWorkflow = (id) => {
    // rename data on python side
    console.log("renameWorkflow: ", id);
    model.set(
      "commands",
      `renameWorkflow: __global__ - ${new Date().getTime()}`
    );
    model.save_changes();
  };

  const saveWorkflowName = (name) => {
    // rename data on python side
    console.log("saveWorkflowName: ", name);
    model.set(
      "commands",
      `saveWorkflowName: ${name} - ${new Date().getTime()}`
    );
    model.save_changes();
  };

  const setPosition = useCallback(
    (pos) =>
      setNodes((nodes) =>
        nodes.map((node) => ({
          ...node,
          data: { ...node.data, toolbarPosition: pos },
        }))
      ),
    [setNodes]
  );

  const forceToolbarVisible = useCallback((enabled) =>
    setNodes((nodes) =>
      nodes.map((node) => ({
        ...node,
        data: { ...node.data, forceToolbarVisible: enabled },
      }))
    )
  );

  return (
    <div style={{ position: "relative", height: "800px", width: "100%" }}>
      <UpdateDataContext.Provider value={updateData}>
        <ReactFlow
          nodes={nodes}
          edges={edges.map((edge) => ({
            ...edge,
            style: { stroke: "black", strokeWidth: 1 },
          }))}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          onConnect={onConnect}
          onNodesDelete={onNodesDelete}
          onEdgesDelete={onEdgesDelete}
          nodeTypes={nodeTypes}
          fitView
          style={rfStyle}
        >
          <Background variant="dots" gap={12} size={1} />
          <MiniMap />
          <Controls orientation="horizontal" position="top-left">
            <ControlButton
              onClick={refreshGraphView}
              title="Refresh Graph View"
            >
              <SymbolIcon />
            </ControlButton>
            <ControlButton onClick={saveFlow} title="Save Flow">
              <UploadIcon />
            </ControlButton>
            <ControlButton onClick={restoreFlow} title="Restore Flow">
              <DownloadIcon />
            </ControlButton>
            <ControlButton onClick={clearFlow} title="Delete Workflow">
              <Cross1Icon />
            </ControlButton>
            <ControlButton onClick={groupNodes} title="Group Nodes">
              <GroupIcon />
            </ControlButton>
            {/* <ControlButton onClick={renameWorkflow} title="Rename Workflow">
              <Pencil1Icon />
            </ControlButton> */}
          </Controls>
          <Panel position="top-right">
            <input
              type="text"
              placeholder="Enter workflow name"
              onKeyDown={(e) => {
                if (e.key === "Enter") {
                  saveWorkflowName(e.target.value);
                }
              }}
            />
          </Panel>
        </ReactFlow>
      </UpdateDataContext.Provider>
    </div>
  );
});

export default { render };
