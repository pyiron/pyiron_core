/**
 * Author: Joerg Neugebauer (with ChatGPT enhancements for isolated/compound nodes + animation)
 * Version: 0.5
 * Status: development
 * Date: Dec 1, 2024
 */

import { useEffect, useRef } from "react";
import ELK from "elkjs/lib/elk.bundled.js";
import { useReactFlow, useNodesInitialized } from "@xyflow/react";

/** ===== Layout Calculation (same as last version) ===== */
export const getLayoutedNodes = async (nodes, edges) => {
  const layoutOptions = {
    "elk.algorithm": "layered",
    "elk.direction": "RIGHT",
    "elk.layered.spacing.edgeNodeBetweenLayers": "40",
    "elk.spacing.nodeNode": "40",
    "elk.layered.nodePlacement.strategy": "SIMPLE",
  };

  const elk = new ELK();

  const nodeById = Object.fromEntries(nodes.map(n => [n.id, n]));

  const childrenMap = {};
  nodes.forEach(n => {
    if (n.parentId) {
      childrenMap[n.parentId] = childrenMap[n.parentId] || [];
      childrenMap[n.parentId].push(n.id);
    }
  });

  const connectedIds = new Set(edges.flatMap(e => [e.source, e.target]));

  const expandConnected = (id) => {
    let current = nodeById[id];
    while (current?.parentId) {
      connectedIds.add(current.parentId);
      current = nodeById[current.parentId];
    }
    const walkChildren = (pid) => {
      (childrenMap[pid] || []).forEach(cid => {
        connectedIds.add(cid);
        walkChildren(cid);
      });
    };
    walkChildren(id);
  };

  [...connectedIds].forEach(expandConnected);

  const connectedNodes = nodes.filter(n => connectedIds.has(n.id));
  const isolatedNodes  = nodes.filter(n => !connectedIds.has(n.id));

  const buildElkChildren = (parentId) => {
    return (childrenMap[parentId] || []).map(cid => {
      const cNode = nodeById[cid];
      return {
        id: cNode.id,
        width: cNode.style?.width || 150,
        height: cNode.style?.height || 50,
        children: buildElkChildren(cNode.id)
      };
    });
  };

  const elkGraph = {
    id: "root",
    layoutOptions,
    children: connectedNodes
      .filter(n => !n.parentId)
      .map(n => ({
        id: n.id,
        width: n.style?.width || 150,
        height: n.style?.height || 50,
        children: buildElkChildren(n.id)
      })),
    edges: edges.map(e => ({
      id: e.id,
      sources: [e.source],
      targets: [e.target],
    })),
  };

  const layoutedGraph = connectedNodes.length > 0
    ? await elk.layout(elkGraph)
    : elkGraph;

  const flattenGraph = (graph) => {
    const out = [];
    const traverse = (node) => {
      out.push(node);
      if (node.children) node.children.forEach(traverse);
    };
    traverse(graph);
    return out;
  };

  let layoutedConnected = [];
  if (connectedNodes.length > 0) {
    layoutedConnected = flattenGraph(layoutedGraph);
    layoutedConnected.shift();
  }

  const maxX = layoutedConnected.length > 0
    ? Math.max(...layoutedConnected.map(n => n.x + (n.width || 0)))
    : 0;
  const extraSpacing = 200;

  const layoutedNodes = nodes.map(node => {
    if (connectedIds.has(node.id)) {
      const layoutedNode = layoutedConnected.find(lg => lg.id === node.id);
      return {
        ...node,
        style: {
          ...node.style,
          width: layoutedNode?.width ?? node.style?.width ?? 150,
          height: node?.expanded
            ? (layoutedNode?.height ?? 50) + 30
            : layoutedNode?.height ?? node.style?.height ?? 50,
        },
        position: {
          x: layoutedNode?.x ?? 0,
          y: node.parentId
            ? (layoutedNode?.y ?? 0) + 30
            : layoutedNode?.y ?? 0
        }
      };
    } else {
      const index = isolatedNodes.findIndex(n => n.id === node.id);
      return {
        ...node,
        position: { x: maxX + extraSpacing, y: index * 100 },
        style: {
          ...node.style,
          width: node.style?.width ?? 150,
          height: node.style?.height ?? 50,
        }
      };
    }
  });

  return layoutedNodes;
};

/** ===== Utility: Animate between positions ===== */
const animatePositions = (oldNodes, newNodes, setNodes, duration = 500) => {
  const startTime = performance.now();

  const nodeStartPositions = {};
  oldNodes.forEach(n => {
    nodeStartPositions[n.id] = { x: n.position.x, y: n.position.y };
  });

  const step = (currentTime) => {
    const t = Math.min((currentTime - startTime) / duration, 1); // 0→1
    const easedT = t * (2 - t); // easeOut interpolation

    const interpolatedNodes = newNodes.map(node => {
      const startPos = nodeStartPositions[node.id] || node.position;
      return {
        ...node,
        position: {
          x: startPos.x + (node.position.x - startPos.x) * easedT,
          y: startPos.y + (node.position.y - startPos.y) * easedT
        }
      };
    });

    setNodes(interpolatedNodes);
    if (t < 1) requestAnimationFrame(step);
  };

  requestAnimationFrame(step);
};

/** ===== Hook for animated ELK layout ===== */
export const useElkLayout = (nodes, edges) => {
  const nodesInitialized = useNodesInitialized();
  const { setNodes } = useReactFlow();
  const prevNodesRef = useRef(nodes);

  useEffect(() => {
    if (nodesInitialized && nodes.length > 0) {
      getLayoutedNodes(nodes, edges).then(layoutedNodes => {
        animatePositions(prevNodesRef.current, layoutedNodes, setNodes);
        prevNodesRef.current = layoutedNodes;
      });
    }
  }, [nodesInitialized, nodes, edges, setNodes]);
};