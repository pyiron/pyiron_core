/**
 * Author: Joerg Neugebauer
 * Copyright: Copyright 2024, Max-Planck-Institut for Sustainable Materials GmbH - Computational Materials Design (CM) Department
 * Version: 0.2
 * Maintainer:
 * Email:
 * Status: development
 * Date: Aug 1, 2024
 */

import React, { useCallback, useState, useEffect, createContext } from "react";
import { createRender, useModel } from "@anywidget/react";
import { getLayoutedNodes } from "./useElkLayout";
// import ELK from "elkjs/lib/elk.bundled.js";
import {
  ReactFlow,
  Controls,
  MiniMap,
  Background,
  applyEdgeChanges,
  applyNodeChanges,
  addEdge,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";

import TextUpdaterNode from "./TextUpdaterNode.jsx";
import CustomNode from "./CustomNode.jsx";

import "./text-updater-node.css";

const rfStyle = {
  //backgroundColor: '#B8CEFF',
  backgroundColor: "#dce1ea",
  //backgroundColor: 'white',
};

export const UpdateDataContext = createContext(null);

// const nodeTypes = { textUpdater: TextUpdaterNode, customNode: CustomNode };

const render = createRender(() => {
  const model = useModel();
  // console.log("model: ", model);
  const initialNodes = JSON.parse(model.get("nodes"));
  const initialEdges = JSON.parse(model.get("edges"));

  const [nodes, setNodes] = useState(initialNodes);
  const [edges, setEdges] = useState(initialEdges);

  const nodeTypes = {
    textUpdater: TextUpdaterNode,
    customNode: CustomNode,
  };

  const updateData = (nodeLabel, handleIndex, newValue) => {
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

  // for test only, can be later removed
  useEffect(() => {
    console.log("nodes_test:", nodes);
    model.set("nodes", JSON.stringify(nodes)); // TODO: maybe better do it via command changeValue(nodeID, handleID, value)
    model.save_changes();
  }, [nodes]);

  model.on("change:nodes", () => {
    const new_nodes = model.get("nodes");
    // console.log("load nodes: ", new_nodes);

    setNodes(JSON.parse(new_nodes));
  });

  model.on("change:edges", () => {
    const new_edges = model.get("edges");
    setEdges(JSON.parse(new_edges));
  });

  model.on("change:graph", () => {
    const new_graph = model.get("graph");
    const graph = JSON.parse(new_graph);
    console.log("load graph: ", graph);

    getLayoutedNodes(nodes, edges, graph).then((layoutedNodes) => {
      setNodes(layoutedNodes);
    });
  });

  model.on("change:mydata", () => {
    console.log("data changed");
    const new_data = model.get("mydata");
    // console.log("load data: ", new_data);
    const data = JSON.parse(new_data);

    console.log("load data: ", data);

    const new_nodes = data.nodes;
    const new_edges = data.edges;
    const new_graph = data.graph;
    // console.log("new data: ", new_nodes, new_edges, new_graph);

    setEdges(new_edges);

    // console.log("new graph: ", new_graph);

    // give start and end time for the layouting

    console.log("start: ", new Date().getTime());
    getLayoutedNodes(new_nodes, new_edges, new_graph).then((layoutedNodes) => {
      setNodes(layoutedNodes);
      model.set("commands", `finished: ${new Date().getTime()}`);
      model.save_changes();
      console.log("finished: ", new Date().getTime());
    });
    // since this is async, we need to wait for the layouted nodes
    // it appears that we never get here
  });

  model.on("change:commands", () => {
    const new_commands = model.get("commands");
    console.log("load commands: ", new_commands);
    // getLayoutedNodes(nodes, edges).then((layoutedNodes) => {
    //   setNodes(layoutedNodes);
    // });
  });

  const onNodesChange = useCallback(
    (changes) => {
      setNodes((nds) => {
        const new_nodes = applyNodeChanges(changes, nds);
        console.log("onNodesChange: ", changes, new_nodes);
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
          nodeTypes={nodeTypes}
          fitView
          style={rfStyle}
        >
          <Background variant="dots" gap={12} size={1} />
          <MiniMap />
          <Controls />
        </ReactFlow>
      </UpdateDataContext.Provider>
    </div>
  );
});

export default { render };
