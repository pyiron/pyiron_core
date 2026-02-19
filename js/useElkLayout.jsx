/* --------------------------------------------------------------
   Layout helper for React‑Flow using ELK (layered, right‑to‑left)
   -------------------------------------------------------------- */

import ELK from "elkjs/lib/elk.bundled.js";

/* --------------------------------------------------------------
   Debug helper – set to true only while you need the JSON file.
   -------------------------------------------------------------- */
const DEBUG_ELK = false;

/**
 * Triggers a download of a JSON object.
 * @param {object} data      The data to be saved.
 * @param {string} filename  Desired file name.
 */
function downloadJSON(data, filename) {
  if (!DEBUG_ELK) return; // keep production quiet

  const blob = new Blob([JSON.stringify(data, null, 2)], {
    type: "application/json",
  });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();

  // clean‑up
  setTimeout(() => {
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }, 0);
}

/**
 * Calculates a new layout for the supplied nodes/edges using ELK.
 *
 * @param {Array} nodes  Array of React‑Flow node objects.
 * @param {Array} edges  Array of React‑Flow edge objects.
 *
 * @returns {Promise<Array>} Same node objects but with updated
 *                           `position` and (optionally) `style`.
 */
export async function getLayoutedNodes(nodes, edges) {
  /* --------------------------------------------------------------
     1️⃣  ELK configuration
     -------------------------------------------------------------- */
  const layoutOptions = {
    "elk.algorithm": "layered",
    "elk.direction": "RIGHT",
    "elk.layered.spacing.edgeNodeBetweenLayers": "40",
    "elk.spacing.nodeNode": "40",
    "elk.layered.nodePlacement.strategy": "SIMPLE",
    "elk.portConstraints": "FIXED_ORDER",
    "hierarchyHandling": "INCLUDE_CHILDREN",
  };

  console.log("nodes in Elk:", nodes, "data:", nodes.map(n => n.data), "style:", nodes.map(n => n.style));
  const elk = new ELK();

  /* --------------------------------------------------------------
     2️⃣  Helper maps (fast look‑ups)
     -------------------------------------------------------------- */
  const nodeById = Object.fromEntries(nodes.map((n) => [n.id, n]));

  // parentId → [childId, …]
  const childrenMap = {};
  nodes.forEach((n) => {
    if (n.parentId) {
      childrenMap[n.parentId] = childrenMap[n.parentId] || [];
      childrenMap[n.parentId].push(n.id);
    }
  });

  /* --------------------------------------------------------------
     3️⃣  Determine which nodes are part of a connected component.
          Isolated nodes will be placed after the layout block.
     -------------------------------------------------------------- */
  const connectedIds = new Set(edges.flatMap((e) => [e.source, e.target]));
  // Ensure parent nodes are considered connected even if they have no edges
  Object.keys(childrenMap).forEach((pid) => connectedIds.add(pid));

  const expandConnected = (id) => {
    // walk up the hierarchy – every ancestor must stay together
    let cur = nodeById[id];
    while (cur?.parentId) {
      connectedIds.add(cur.parentId);
      cur = nodeById[cur.parentId];
    }

    // walk down – all descendants belong to the same component
    const walkChildren = (parentId) => {
      (childrenMap[parentId] || []).forEach((childId) => {
        connectedIds.add(childId);
        walkChildren(childId);
      });
    };
    walkChildren(id);
  };

  [...connectedIds].forEach(expandConnected);

  const connectedNodes = nodes.filter((n) => connectedIds.has(n.id));
  const isolatedNodes = nodes.filter((n) => !connectedIds.has(n.id));

  /* --------------------------------------------------------------
     4️⃣  Recursive builder for ELK children
     -------------------------------------------------------------- */
  const buildElkChildren = (parentId) => {
    const children = childrenMap[parentId] || [];

    if (children.length === 0) return undefined; // omit empty `children`

    return children.map((cid) => {
      const cNode = nodeById[cid];
      return {
        id: cNode.id,
        width: cNode.style?.width ?? 150,
        height: cNode.style?.height ?? 50,
        properties: {
          "org.eclipse.elk.portConstraints": "FIXED_ORDER",
        },
        ports: [
          // target ports – reversed because ELK expects clockwise order
          ...(cNode.data?.target_labels ?? [])
            .slice()
            .reverse()
            .map((p) => ({
              id: `${cNode.id}_t_${p}`,
              layoutOptions: { side: "WEST" },
            })),
          // source ports
          ...(cNode.data?.source_labels ?? []).map((p) => ({
            id: `${cNode.id}_s_${p}`,
            layoutOptions: { side: "EAST" },
          })),
        ],
        // recurse only when there are grandchildren
        ...(buildElkChildren(cNode.id) && {
          children: buildElkChildren(cNode.id),
        }),
      };
    });
  };

  /* --------------------------------------------------------------
     5️⃣  Assemble the ELK graph object
     -------------------------------------------------------------- */
  const elkGraph = {
    id: "root",
    layoutOptions,
    children: (connectedNodes.length > 0 ? connectedNodes : nodes)
      .filter((n) => !n.parentId) // top‑level nodes only
      .map((n) => ({
        id: n.id,
        width: n.style?.width ?? 150,
        height: n.style?.height ?? 50,
        properties: {
          "org.eclipse.elk.portConstraints": "FIXED_ORDER",
        },
        ports: [
          ...(n.data?.target_labels ?? [])
            .slice()
            .reverse()
            .map((p) => ({
              id: `${n.id}_t_${p}`,
              layoutOptions: { side: "WEST" },
            })),
          ...(n.data?.source_labels ?? []).map((p) => ({
            id: `${n.id}_s_${p}`,
            layoutOptions: { side: "EAST" },
          })),
        ],
        ...(buildElkChildren(n.id) && { children: buildElkChildren(n.id) }),
      })),
    edges: edges.map((e) => ({
      id: `e_${e.id}`,
      sources: [
        `${e.source}_s_${e.sourceHandle ?? ""}`.replace(/_s_$/, "_s_"),
      ],
      targets: [
        `${e.target}_t_${e.targetHandle ?? ""}`.replace(/_t_$/, "_t_"),
      ],
    })),
  };

  // -----------------------------------------------------------------
  // 6️⃣  OPTIONAL: dump the raw ELK graph for debugging
  // -----------------------------------------------------------------
  downloadJSON(elkGraph, "elkGraph.json");

  // -----------------------------------------------------------------
  // 7️⃣  Run the layout (or skip it when there is nothing to layout)
  // -----------------------------------------------------------------
  const layoutedGraph =
    connectedNodes.length > 0 ? await elk.layout(elkGraph) : elkGraph;

  /* --------------------------------------------------------------
     8️⃣  Flatten the ELK result (skip the artificial root node)
     -------------------------------------------------------------- */
  const flattenGraph = (graph) => {
    const result = [];

    const walk = (node) => {
      // skip the artificial root node (`id === "root"`)
      if (node.id !== "root") result.push(node);
      if (node.children) node.children.forEach(walk);
    };

    walk(graph);
    return result;
  };

  const layoutedConnected = flattenGraph(layoutedGraph);

  /* --------------------------------------------------------------
     9️⃣  Determine the right‑most X‑coordinate of the laid‑out block.
          Isolated nodes will be placed a bit further to the right.
     -------------------------------------------------------------- */
  const maxX =
    layoutedConnected.length > 0
      ? Math.max(
          ...layoutedConnected.map(
            (n) => (n.x ?? 0) + (n.width ?? 150) // fallback width = 150
          )
        )
      : 0;
  const EXTRA_SPACING = 200; // gap between the layout block & isolated nodes

  /* --------------------------------------------------------------
    🔟  Build the final node array that React‑Flow expects.
     -------------------------------------------------------------- */
  const finalNodes = nodes.map((node) => {
    // -----------------------------------------------------------------
    // Nodes that belong to a connected component → use ELK positions
    // -----------------------------------------------------------------
    if (connectedIds.has(node.id)) {
      const elkNode = layoutedConnected.find((lg) => lg.id === node.id) ?? {};

      const baseWidth = elkNode.width ?? node.style?.width ?? 150;
      const baseHeight = elkNode.height ?? node.style?.height ?? 50;
      console.log('layouted: ', node.id, elkNode.width, node.data?.expanded)

      // If the node is expanded we give it a little extra height so the
      // inner content (e.g. a group body) does not overlap the ports.
      const height = node.data?.expanded ? baseHeight + 30 : baseHeight;

      // When a node lives inside a group we need to offset it by the
      // group's header height (30 px in this example).
      const yOffset = node.parentId ? 30 : 0;

      return {
        ...node,
        style: {
          ...node.style,
          width: baseWidth,
          height,
        },
        position: {
          x: elkNode.x ?? 0,
          y: (elkNode.y ?? 0) + yOffset,
        },
      };
    }

    // -----------------------------------------------------------------
    // Isolated nodes → stack them vertically to the right of the block
    // -----------------------------------------------------------------
    const isoIndex = isolatedNodes.findIndex((n) => n.id === node.id);
    return {
      ...node,
      style: {
        ...node.style,
        width: node.style?.width ?? 150,
        height: node.style?.height ?? 50,
      },
      position: {
        x: maxX + EXTRA_SPACING,
        y: isoIndex * 100, // simple vertical spacing
      },
    };
  });

  return finalNodes;
}

