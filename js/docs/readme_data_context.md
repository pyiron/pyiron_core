# UpdateDataContext Flow in `widget.tsx` / `CustomNode.tsx`

## Overview
`UpdateDataContext` is a **React Context** that allows deeply nested components (like `CustomNode.tsx`) to update node data in the main ReactFlow graph without prop-drilling.

It connects **UI events in node components** to:
1. **Local React state updates** (instant UI feedback in the graph).
2. **Backend / AnyWidget model updates** (persistent state sync).



## Components Involved
- **`widget.tsx`**
  - Defines and exports the `UpdateDataContext`.
  - Implements the `updateData` function.
  - Provides the context to all child components via `<UpdateDataContext.Provider>`.

- **`CustomNode.tsx`**
  - Consumes the context with `useContext(UpdateDataContext)`.
  - Calls it when inputs change.



## Definition

```ts
// widget.tsx
export type UpdateDataFn = (
  nodeLabel: string,
  handleIndex: number,
  newValue: unknown
) => void;

export const UpdateDataContext = createContext<UpdateDataFn | null>(null);
```



## Provider Setup (`widget.tsx`)

```tsx
const updateData: UpdateDataFn = (nodeLabel, handleIndex, newValue) => {
  const var_data = { label: nodeLabel, handle: handleIndex, value: newValue };

  // 1. Send to backend/model
  sendCommand(COMMAND_PREFIXES.CHANGE_NODE_VALUE, JSON.stringify(var_data));

  // 2. Update local state
  setNodes(prevNodes =>
    prevNodes.map(node =>
      node.id === nodeLabel
        ? {
            ...node,
            data: {
              ...node.data,
              target_values: Object.assign(
                [...node.data.target_values],
                { [handleIndex]: newValue }
              )
            }
          }
        : node
    )
  );
};

return (
  <UpdateDataContext.Provider value={updateData}>
    <ReactFlow ... />
  </UpdateDataContext.Provider>
);
```



## Consuming the Context (`CustomNode.tsx`)

```tsx
const updateData = useContext(UpdateDataContext)!;

const pushValue = (raw: any) => {
  const conv = convertInput(raw, type);
  const finalVal = conv.error ? raw : conv.value;
  updateData(data.label, index, finalVal);
};
```

When a user interacts with an input, `pushValue` calls `updateData`, which:
- **Updates local ReactFlow node data** immediately (instant UI update).
- **Sends a command to the backend** to persist the change.



## Flow Diagram

```
User changes value in CustomNode UI
            │
            ▼
 pushValue() in CustomNode.tsx
            │
            ▼
 updateData(...) from UpdateDataContext
            │
  ┌─────────┴─────────┐
  │                   │
Update local React    Send command to
state (setNodes)      backend model
  │                   │
ReactFlow rerenders   Backend state updates
(instantly visible)   (persistent sync)
```



## Benefits
- **No prop drilling**: Any nested component in the graph can update nodes.
- **Single source of truth**: The function in `widget.tsx` owns both UI and backend sync.
- **Immediate feedback**: Local state update means users see changes instantly.



## Common Pitfalls 🚨

1. **Forgetting to export the context**  
   - If `widget.tsx` does **not** export:
     ```ts
     export const UpdateDataContext = createContext<...>(null);
     ```
     Then:
     ```ts
     import { UpdateDataContext } from "./widget";
     ```
     will fail with:
     ```
     No matching export for import "UpdateDataContext"
     ```

2. **Forgetting the Provider**  
   - If you don’t wrap child components in:
     ```tsx
     <UpdateDataContext.Provider value={updateData}>
       {...children}
     </UpdateDataContext.Provider>
     ```
     then `useContext(UpdateDataContext)` in `CustomNode.tsx` will return `null`, leading to:
     ```
     TypeError: context is not a function
     ```

3. **Not memoizing `updateData`**  
   - Without `useCallback`, ReactFlow nodes may re-render more than necessary, hurting performance.

4. **Updating model state but not local state**  
   - If `sendCommand()` is called but `setNodes()` isn’t, the UI won’t update until the backend pushes changes back — slower user feedback.

5. **Updating local state but not sending to the backend**  
   - If `setNodes()` is called but not `sendCommand()`, the backend stays out-of-sync and changes will be lost.



By following this pattern — **export, provide, consume, and sync** — you ensure a smooth, immediate, and persistent node update workflow.

