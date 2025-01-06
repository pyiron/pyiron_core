/**
 * Author: Joerg Neugebauer
 * Copyright: Copyright 2025, Max-Planck-Institut for Sustainable Materials GmbH - Computational Materials Design (CM) Department
 * Version: 0.2
 * Maintainer:
 * Email:
 * Status: development
 * Date: Dec 1, 2024
 */

import { useEffect } from "react";
import ELK from "elkjs/lib/elk.bundled.js";
import { useNodesInitialized, useReactFlow } from "@xyflow/react";
// import { saveAs } from 'file-saver';

// elk layouting options can be found here:
// https://www.eclipse.org/elk/reference/algorithms/org-eclipse-elk-layered.html

// uses elkjs to give each node a layouted position
export const getLayoutedNodes = async (nodes, edges, graph) => {
  const layoutOptions = {
    "elk.algorithm": "layered",
    "elk.direction": "RIGHT",
    "elk.layered.spacing.edgeNodeBetweenLayers": "40",
    "elk.spacing.nodeNode": "40",
    "elk.layered.nodePlacement.strategy": "SIMPLE",
  };

//   console.log("nodes layout: ", nodes);
//   console.log("edges layout: ", edges);
//   console.log("graph layout: ", graph);
  const elk = new ELK();

  const layoutedGraph = await elk.layout(graph);
//   console.log("layoutedGraph: ", layoutedGraph);

  // layouted graph is a nested dictionary containing the layouted nodes in the children key
  // we need a flattened list of nodes to update the react flow nodes
  const flattenGraph = (graph) => {
    const nodes = [];
    const traverse = (node) => {
      nodes.push({ ...node, expanded: node.expanded ?? false });
      if (node.children) {
        node.children.forEach(traverse);
      }
    };
    traverse(graph);
    return nodes;
  };

  const flattenedNodes = flattenGraph(layoutedGraph);
  // remove the root node
  flattenedNodes.shift();

//   console.log("flattenedNodes: ", flattenedNodes);

  const layoutedNodes = nodes.map((node) => {
    const layoutedNode = flattenedNodes.find((lgNode) => lgNode.id === node.id);

    // console.log(
    //   "layoutedNode: ",
    //   layoutedNode,
    //   node.id,
    //   node.expanded,
    //   layoutedNode.expanded
    // );
    return {
      ...node,
      // update width and height in node.style
      style: {
        ...node.style,
        width: layoutedNode?.width ?? 0,
        // increase the height by 30 if layoutedNode.expanded is true
        height: node?.expanded
          ? layoutedNode.height + 30
          : layoutedNode.height ?? 0,
      },
      position: {
        x: layoutedNode?.x ?? 0,
        y: node.parentId ? layoutedNode.y + 30 : layoutedNode?.y ?? 0,
      },
    };
  });
  console.log("layoutedGraphNodes: ", layoutedNodes);

  return layoutedNodes;
};
