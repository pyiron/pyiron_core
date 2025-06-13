/**
 * Author: Joerg Neugebauer
 * Copyright: Copyright 2024, Max-Planck-Institut for Sustainable Materials GmbH - Computational Materials Design (CM) Department
 * Version: 0.2
 * Status: development
 * Date: Aug 1, 2024
 */

import React, { useCallback, useState, createContext, useEffect } from "react";
import { createRender, useModel } from "@anywidget/react";
import { getLayoutedNodes } from "./useElkLayout";
import {
  ReactFlow,
  Controls,
  ControlButton,
  MiniMap,
  Background,
  applyEdgeChanges,
  applyNodeChanges,
  addEdge,
  Panel,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import {
  SymbolIcon,
  UploadIcon,
  DownloadIcon,
  Cross1Icon,
  GroupIcon,
} from "@radix-ui/react-icons";

import TextUpdaterNode from "./TextUpdaterNode.jsx";
import CustomNode from "./CustomNode.jsx";

import "./text-updater-node.css";

const rfStyle = {
  backgroundColor: "#dce1ea",
};

export const UpdateDataContext = createContext(null);

// const globalStatusRef = { current: "initial" };

const COMMAND_PREFIXES = {
  CHANGE_NODE_VALUE: "change_node_value",
  SELECTED_NODES: "selected_nodes",
  ADD_EDGE: "add_edge",
  DELETE_NODE: "delete_node",
  DELETE_EDGE: "delete_edge",
  REFRESH_GRAPH_VIEW: "refreshGraphView",
  SAVE_FLOW: "saveFlow",
  RESTORE_FLOW: "restoreFlow",
  CLEAR_FLOW: "clearFlow",
  GROUP_NODES: "groupSelectedNodes",
  RENAME_WORKFLOW: "renameWorkflow",
  SAVE_WORKFLOW_NAME: "saveWorkflowName",
};

const render = createRender(() => {
  const model = useModel();

  const [nodes, setNodes] = useState([]);
  const [edges, setEdges] = useState([]);
  const [selectedNodes, setSelectedNodes] = useState([]);

  const nodeTypes = {
    textUpdater: TextUpdaterNode,
    customNode: CustomNode,
  };

  const sendCommand = useCallback(
    (command, payload = "__global__") => {
      console.log(`Sending command: ${command} with payload: ${payload}`);
      const timestamp = new Date().getTime();
      model.set("commands", `${command}: ${payload} - ${timestamp}`);
      model.save_changes();
    },
    [model]
  );

  const updateData = useCallback(
    (nodeLabel, handleIndex, newValue) => {
      console.log(
        `Updating node ${nodeLabel}, handle ${handleIndex} with value: ${newValue}`
      );
      const var_data = {
        label: nodeLabel,
        handle: handleIndex,
        value: newValue,
      };
      sendCommand(COMMAND_PREFIXES.CHANGE_NODE_VALUE, JSON.stringify(var_data));
      setNodes((prevNodes) =>
        prevNodes.map((node) => {
          if (node.id !== nodeLabel) return node;

          const updatedTargetValues = [...node.data.target_values];
          updatedTargetValues[handleIndex] = newValue;

          return {
            ...node,
            data: {
              ...node.data,
              target_values: updatedTargetValues,
            },
          };
        })
      );
    },
    [sendCommand]
  );

  // Setup event listeners on model with cleanup
  useEffect(() => {
    const onNodesChange = () => {
      const newNodesRaw = model.get("nodes");
      try {
        const newNodes = JSON.parse(newNodesRaw);
        setNodes(newNodes);
      } catch {
        console.error("Failed to parse nodes from model.");
      }
    };

    const onEdgesChange = () => {
      const newEdgesRaw = model.get("edges");
      try {
        const newEdges = JSON.parse(newEdgesRaw);
        setEdges(newEdges);
      } catch {
        console.error("Failed to parse edges from model.");
      }
    };

    const onMyDataChange = () => {
      // if (globalStatusRef.current === "running") {
      //   console.log("Data change ignored due to running status");
      //   return;
      // }
      // globalStatusRef.current = "running";

      const rawData = model.get("mydata");
      try {
        const data = JSON.parse(rawData);
        const { nodes: newNodes, edges: newEdges, graph: newGraph } = data;

        getLayoutedNodes(newNodes, newEdges, newGraph).then((layoutedNodes) => {
          setNodes(layoutedNodes);
          setEdges(newEdges);
          globalStatusRef.current = "finished";
        });
      } catch {
        console.error("Failed to parse mydata from model.");
        globalStatusRef.current = "finished";
      }
    };

    const onCommandsChange = () => {
      const cmds = model.get("commands");
      console.log("Commands changed:", cmds);
    };

    model.on("change:nodes", onNodesChange);
    model.on("change:edges", onEdgesChange);
    model.on("change:mydata", onMyDataChange);
    model.on("change:commands", onCommandsChange);

    return () => {
      model.off("change:nodes", onNodesChange);
      model.off("change:edges", onEdgesChange);
      model.off("change:mydata", onMyDataChange);
      model.off("change:commands", onCommandsChange);
    };
  }, [model]);

  // Handlers for ReactFlow events
  const onNodesChange = useCallback(
    (changes) => {
      setNodes((nds) => {
        const updatedNodes = applyNodeChanges(changes, nds);

        // After node changes are applied, filter nodes with selected === true to get full current selection
        const currentlySelectedNodes = updatedNodes.filter(
          (node) => node.selected
        );

        setSelectedNodes(currentlySelectedNodes);

        const selectedIdsStr = currentlySelectedNodes
          .map((node) => node.id)
          .join(",");
        sendCommand(COMMAND_PREFIXES.SELECTED_NODES, selectedIdsStr);

        model.set("nodes", JSON.stringify(updatedNodes));
        model.save_changes();

        return updatedNodes;
      });
    },
    [model, sendCommand]
  );

  const onEdgesChange = useCallback(
    (changes) => {
      setEdges((eds) => {
        const updatedEdges = applyEdgeChanges(changes, eds);
        model.set("edges", JSON.stringify(updatedEdges));
        model.save_changes();
        return updatedEdges;
      });
    },
    [model]
  );

  const onConnect = useCallback(
    (params) => {
      setEdges((eds) => {
        const newEdges = addEdge(params, eds);
        model.set("edges", JSON.stringify(newEdges));
        sendCommand(
          COMMAND_PREFIXES.ADD_EDGE,
          `${params.source}/${params.sourceHandle} > ${params.target}/${params.targetHandle}`
        );
        return newEdges;
      });
    },
    [model, sendCommand]
  );

  const deleteNode = useCallback(
    (id) => {
      sendCommand(COMMAND_PREFIXES.DELETE_NODE, id);
    },
    [sendCommand]
  );

  const onNodesDelete = useCallback(
    (deleted) => {
      if (deleted.length > 0) {
        deleteNode(deleted[0].id);
      }
    },
    [deleteNode]
  );

  const deleteEdge = useCallback(
    (params) => {
      sendCommand(
        COMMAND_PREFIXES.DELETE_EDGE,
        `${params.source}/${params.sourceHandle} > ${params.target}/${params.targetHandle}`
      );
    },
    [sendCommand]
  );

  const onEdgesDelete = useCallback(
    (deleted) => {
      if (deleted.length > 0) {
        deleteEdge(deleted[0]);
      }
    },
    [deleteEdge]
  );

  // Commands for controlling the workflow
  const workflowCommands = {
    refreshGraphView: () => sendCommand(COMMAND_PREFIXES.REFRESH_GRAPH_VIEW),
    saveFlow: () => sendCommand(COMMAND_PREFIXES.SAVE_FLOW),
    restoreFlow: () => sendCommand(COMMAND_PREFIXES.RESTORE_FLOW),
    clearFlow: () => sendCommand(COMMAND_PREFIXES.CLEAR_FLOW),
    groupNodes: () => sendCommand(COMMAND_PREFIXES.GROUP_NODES),
    renameWorkflow: () => sendCommand(COMMAND_PREFIXES.RENAME_WORKFLOW),
    saveWorkflowName: (name) => {
      if (name.trim()) {
        sendCommand(COMMAND_PREFIXES.SAVE_WORKFLOW_NAME, name.trim());
      }
    },
  };

  const setPosition = useCallback(
    (pos) =>
      setNodes((nodes) =>
        nodes.map((node) => ({
          ...node,
          data: { ...node.data, toolbarPosition: pos },
        }))
      ),
    []
  );

  const forceToolbarVisible = useCallback(
    (enabled) =>
      setNodes((nodes) =>
        nodes.map((node) => ({
          ...node,
          data: { ...node.data, forceToolbarVisible: enabled },
        }))
      ),
    []
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
              onClick={workflowCommands.refreshGraphView}
              title="Refresh Graph View"
            >
              <SymbolIcon />
            </ControlButton>
            <ControlButton
              onClick={workflowCommands.saveFlow}
              title="Save Flow"
            >
              <UploadIcon />
            </ControlButton>
            <ControlButton
              onClick={workflowCommands.restoreFlow}
              title="Restore Flow"
            >
              <DownloadIcon />
            </ControlButton>
            <ControlButton
              onClick={workflowCommands.clearFlow}
              title="Delete Workflow"
            >
              <Cross1Icon />
            </ControlButton>
            <ControlButton
              onClick={workflowCommands.groupNodes}
              title="Group Nodes"
            >
              <GroupIcon />
            </ControlButton>
            {/* Uncomment if rename workflow will be implemented */}
            {/* <ControlButton onClick={workflowCommands.renameWorkflow} title="Rename Workflow">
              <Pencil1Icon />
            </ControlButton> */}
          </Controls>
          <Panel position="top-right">
            <input
              type="text"
              placeholder="Enter workflow name"
              onKeyDown={(e) => {
                if (e.key === "Enter") {
                  workflowCommands.saveWorkflowName(e.target.value);
                  e.target.value = "";
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
