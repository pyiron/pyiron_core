import React, {
  useCallback,
  useState,
  createContext,
  useEffect,
  useRef
} from "react";
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
  Node,
  Edge,
  OnNodesChange,
  OnEdgesChange,
  OnConnect,
  Connection,
} from "@xyflow/react";

import "@xyflow/react/dist/style.css";
import {
  SymbolIcon,
  UploadIcon,
  DownloadIcon,
  Cross1Icon,
  GroupIcon,
} from "@radix-ui/react-icons";

// import TextUpdaterNode from "./TextUpdaterNode.jsx";
import CustomNode from "./CustomNode.tsx";

import "./text-updater-node.css";

export type UpdateDataFn = (
  nodeLabel: string,
  handleIndex: number,
  newValue: unknown
) => void;
export const UpdateDataContext = createContext<UpdateDataFn | null>(null);

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
  RENAME_NODE: "renameNode",
};

const rfStyle = { backgroundColor: "#dce1ea" };

const render = createRender(() => {
  const model = useModel();

  const [nodes, setNodes] = useState<Node[]>([]);
  const [edges, setEdges] = useState<Edge[]>([]);
  const [selectedNodes, setSelectedNodes] = useState<Node[]>([]);

  const flowWrapperRef = useRef<HTMLDivElement>(null);

  const nodeTypes = { customNode: CustomNode };

  const sendCommand = useCallback(
    (command: string, payload: string = "__global__") => {
      const timestamp = Date.now();
      console.log(`[SEND COMMAND] ${command}: ${payload} - ${timestamp}`);
      model.set("commands", `${command}: ${payload} - ${timestamp}`);
      model.save_changes();
    },
    [model]
  );

  const updateData: UpdateDataFn = useCallback(
    (nodeLabel, handleIndex, newValue) => {
      const var_data = { label: nodeLabel, handle: handleIndex, value: newValue };
      sendCommand(COMMAND_PREFIXES.CHANGE_NODE_VALUE, JSON.stringify(var_data));
      setNodes((prevNodes) =>
        prevNodes.map((node) => {
          if (node.id !== nodeLabel) return node;
          const updatedTargetValues = [...(node.data.target_values as unknown[])];
          updatedTargetValues[handleIndex] = newValue;
          return { ...node, data: { ...node.data, target_values: updatedTargetValues } };
        })
      );
    },
    [sendCommand]
  );

  useEffect(() => {
    const onNodesChangeFromModel = () => {
      try {
        setNodes(JSON.parse(model.get("nodes")) as Node[]);
      } catch {
        console.error("Failed to parse nodes from model.");
      }
    };

    const onEdgesChangeFromModel = () => {
      try {
        setEdges(JSON.parse(model.get("edges")) as Edge[]);
      } catch {
        console.error("Failed to parse edges from model.");
      }
    };

    const onMyDataChange = () => {
      try {
        const data = JSON.parse(model.get("mydata")) as {
          nodes: Node[];
          edges: Edge[];
          graph: any;
        };
        getLayoutedNodes(data.nodes, data.edges, data.graph).then((layoutedNodes) => {
          setNodes(layoutedNodes);
          setEdges(data.edges);
        });
      } catch {
        console.error("Failed to parse mydata from model.");
      }
    };

    model.on("change:nodes", onNodesChangeFromModel);
    model.on("change:edges", onEdgesChangeFromModel);
    model.on("change:mydata", onMyDataChange);

    return () => {
      model.off("change:nodes", onNodesChangeFromModel);
      model.off("change:edges", onEdgesChangeFromModel);
      model.off("change:mydata", onMyDataChange);
    };
  }, [model]);

  const onNodesChangeHandler: OnNodesChange = useCallback(
    (changes) => {
      console.log("[onNodesChange] changes:", changes);
      setNodes((nds) => {
        const updatedNodes = applyNodeChanges(changes, nds);
        const currentlySelectedNodes = updatedNodes.filter((node) => node.selected);
        setSelectedNodes(currentlySelectedNodes);
        sendCommand(
          COMMAND_PREFIXES.SELECTED_NODES,
          currentlySelectedNodes.map((node) => node.id).join(",")
        );
        model.set("nodes", JSON.stringify(updatedNodes));
        model.save_changes();
        return updatedNodes;
      });
    },
    [model, sendCommand]
  );

  const onEdgesChangeHandler: OnEdgesChange = useCallback(
    (changes) => {
      console.log("[onEdgesChange] changes:", changes);
      setEdges((eds) => {
        const updatedEdges = applyEdgeChanges(changes, eds);
        model.set("edges", JSON.stringify(updatedEdges));
        model.save_changes();
        return updatedEdges;
      });
    },
    [model]
  );

  const onConnectHandler: OnConnect = useCallback(
    (params: Connection) => {
      console.log("[onConnect] params:", params);
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

  /**
   * NOTE: Custom node deletion handling
   *
   * We intercept DELETE/BACKSPACE at the keyboard level instead of relying on
   * React Flow's built‑in node deletion events. This is because in React Flow's 
   * default deletion sequence, when a node is deleted it first triggers
   * `onEdgesDelete` for all connected edges, and only afterwards calls
   * `onNodesDelete`. This means that in a controlled ReactFlow setup,
   * the backend would always receive `delete_edge` commands before (or even 
   * without) the intended `delete_node` command.
   *
   * In our application, the backend automatically deletes connected edges when 
   * a node is removed, so sending `delete_edge` for these cases is redundant 
   * and can cause unwanted side effects.
   *
   * By overriding the default deletion flow:
   *  - We detect and handle DELETE/BACKSPACE presses ourselves.
   *  - We send **only** the `delete_node` command to the backend.
   *  - We manually remove the node and any connected edges from local state
   *    without triggering `onEdgesDelete`.
   *  - `onEdgesDelete` remains available only for manual, user‑initiated edge deletions.
   *
   * This ensures consistent backend commands, avoids duplicate deletions,
   * and prevents race‑conditions from React Flow's internal event order.
   */

  // Intercept Delete/Backspace for node deletion
  const handleKeyDown = useCallback((event: KeyboardEvent) => {
    // Ignore key events from input fields, textareas, or other editable elements
    const target = event.target as HTMLElement;
    const isEditable =
      target.tagName === 'INPUT' ||
      target.tagName === 'TEXTAREA' ||
      target.isContentEditable;

    if (isEditable) {
      return; // let the browser handle deletion normally
    }

    // Continue only for Delete/Backspace keys when nodes are selected
    if (!(event.key === "Delete" || event.key === "Backspace")) return;
    if (selectedNodes.length === 0) return;

    event.preventDefault();
    console.log("[handleKeyDown] Deleting selected nodes:", selectedNodes);

    selectedNodes.forEach((node) => {
      sendCommand(COMMAND_PREFIXES.DELETE_NODE, node.id);
    });

    // Remove from local state
    setNodes((nds) => nds.filter((n) => !selectedNodes.some((sel) => sel.id === n.id)));
    // Remove connected edges locally
    setEdges((eds) =>
      eds.filter((e) => !selectedNodes.some((sel) => sel.id === e.source || sel.id === e.target))
    );
  }, [selectedNodes, sendCommand]);

  useEffect(() => {
    const wrapper = flowWrapperRef.current;
    if (wrapper) {
      wrapper.addEventListener("keydown", handleKeyDown);
    }
    return () => {
      if (wrapper) {
        wrapper.removeEventListener("keydown", handleKeyDown);
      }
    };
  }, [handleKeyDown]);

  const onEdgesDelete = useCallback(
    (deleted: Edge[]) => {
      // These are only manual edge deletions now
      console.log("[onEdgesDelete] manual edge deletion:", deleted);
      deleted.forEach((edge) => {
        sendCommand(
          COMMAND_PREFIXES.DELETE_EDGE,
          `${edge.source}/${edge.sourceHandle} > ${edge.target}/${edge.targetHandle}`
        );
      });
    },
    [sendCommand]
  );

  const workflowCommands = {
    refreshGraphView: () => sendCommand(COMMAND_PREFIXES.REFRESH_GRAPH_VIEW),
    saveFlow: () => sendCommand(COMMAND_PREFIXES.SAVE_FLOW),
    restoreFlow: () => sendCommand(COMMAND_PREFIXES.RESTORE_FLOW),
    clearFlow: () => sendCommand(COMMAND_PREFIXES.CLEAR_FLOW),
    groupNodes: () => sendCommand(COMMAND_PREFIXES.GROUP_NODES),
    renameWorkflow: () => sendCommand(COMMAND_PREFIXES.RENAME_WORKFLOW),
    saveWorkflowName: (name: string) => {
      if (name.trim()) {
        sendCommand(COMMAND_PREFIXES.SAVE_WORKFLOW_NAME, name.trim());
      }
    },
  };

  return (
    <div
      ref={flowWrapperRef}
      tabIndex={0} // Ensure div can receive key events
      style={{ position: "relative", height: "800px", width: "100%" }}
    >
      <UpdateDataContext.Provider value={updateData}>
        <ReactFlow
          nodes={nodes}
          edges={edges.map((edge) => ({
            ...edge,
            style: { stroke: "black", strokeWidth: 1 },
          }))}
          onNodesChange={onNodesChangeHandler}
          onEdgesChange={onEdgesChangeHandler}
          onConnect={onConnectHandler}
          onEdgesDelete={onEdgesDelete} // only manual edges now
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
            <ControlButton onClick={workflowCommands.saveFlow} title="Save Flow">
              <UploadIcon />
            </ControlButton>
            <ControlButton onClick={workflowCommands.restoreFlow} title="Restore Flow">
              <DownloadIcon />
            </ControlButton>
            <ControlButton onClick={workflowCommands.clearFlow} title="Delete Workflow">
              <Cross1Icon />
            </ControlButton>
            <ControlButton onClick={workflowCommands.groupNodes} title="Group Nodes">
              <GroupIcon />
            </ControlButton>
          </Controls>
          <Panel position="top-right">
            <input
              type="text"
              placeholder="Enter workflow name"
              onKeyDown={(e) => {
                if (e.key === "Enter") {
                  workflowCommands.saveWorkflowName(
                    (e.target as HTMLInputElement).value
                  );
                  (e.target as HTMLInputElement).value = "";
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