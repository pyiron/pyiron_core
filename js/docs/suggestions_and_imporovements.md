## **🛠 Architectural Improvements**

### **1. Stronger Type Safety & Contracts**
- **Current Concern:** Many props (esp. in `CustomNode`'s `data` object) are `any`, and type parsing logic (`parseLiteralType`, `convertInput`) deals with untyped values.
- **Improvement:**  
  - Define shared **TypeScript interfaces** for:
    - Node `data` structure (`source_labels`, `target_labels`, `target_types`, …)
    - Command payloads (`change_node_value`, `add_edge`, …)
  - Replace `any` where possible with these interfaces.
- **Benefit:** Catches errors at compile-time, improves autocomplete for props and command building.



### **2. Decouple UI from Backend Command Formatting**
- **Current Concern:** `widget.tsx` builds backend command strings inline (e.g., `"delete_node: " + payload`).
- **Improvement:**  
  - Introduce a **CommandBuilder** utility module with functions:
    ```ts
    buildChangeNodeValue(label: string, handle: number, value: unknown): string
    buildAddEdgeCommand(source, sHandle, target, tHandle): string
    // …
    ```
  - The widget calls these instead of concatenating strings.
- **Benefit:**  
  - Consistent formatting  
  - Easier to update protocol format without touching UI logic



### **3. Extract `InputHandle` & `OutputHandle` into Separate Files**
- **Current Concern:** `CustomNode.tsx` is quite large, embedding multiple subcomponents.
- **Improvement:**  
  - Move `InputHandle`, `OutputHandle`, and `NodeLabel` to their own files.
    ```
    components/nodes/InputHandle.tsx
    components/nodes/OutputHandle.tsx
    components/nodes/NodeLabel.tsx
    ```
  - Keep `CustomNode.tsx` lean — only compose these.
- **Benefit:** Better maintainability, easier targeted testing.



### **4. Context usage optimization**
- **Current Concern:** `UpdateDataContext` only passes a single function but triggers a full re-render of all child consumers on any change.
- **Improvement:**  
  - Use `React.useCallback` in Provider to memoize the function and prevent unnecessary recomputes.
- **Benefit:** Avoids unnecessary re-renders in nodes that haven’t changed.



### **5. Improved Styling Consistency**
- **Current Concern:** Many inline styles (`style={{ ... }}`) spread across files; `STYLE_VARS` only covers dimensions, not all colors or spacing.
- **Improvement:**  
  - Extract **all styles** into a central theme/constants file or CSS Modules / Styled-components.
- **Benefit:**  
  - Easier theme changes  
  - Consistent look & feel



### **6. Improve OptionalLiteralSelect State Sync**
- **Current Concern:** Double state (`useValue` and `selected`) can be a source of state-sync bugs.
- **Improvement:**  
  - Consider making it a **controlled component** where `value` alone drives whether it's enabled, instead of duplicating in `useValue`.
- **Benefit:** Fewer moving parts, simpler update paths.



### **7. More Granular Testing**
- **Current Concern:** A lot of logic sits inside components.
- **Improvement:**  
  - Extract pure logic (e.g., determining input type from `parseLiteralType`) into pure functions, and unit-test them separately.
- **Benefit:** Easier, faster tests that don’t require rendering the DOM.



## **🆕 Feature Suggestions**

### **A. Node-Level Validation**
- Allow `parseLiteralType`/`convertInput` to also return **validation status** and display inline errors.
- Example: Reject `"abc"` in an `int` type input, highlight input in red.



### **B. Undo / Redo Support**
- Maintain a local stack of changes `(nodes, edges)` in `widget.tsx`.
- Toolbar buttons to undo/redo changes.
- Would integrate well with `sendCommand`.



### **C. Enhanced Toolbar**
- Add buttons for **Auto-layout** (re-run ELK.js), **Zoom to fit**, **Export diagram**.
- Quick “Duplicate Node” function.



### **D. Customizable Node Templates**
- Let users define reusable node templates with pre-filled settings.
- Store template in backend and re-use in multiple places.



### **E. Better Boolean Editing UI**
- Instead of `Checkbox` + label, consider a toggle switch UI or “True”/“False” selector.



### **F. Support for Default Values per Type**
- When `onToggle` in `OptionalLiteralSelect` enables a value, instead of **remembering last value**, pull a default for that type from a config (`defaultValues["int"] = 0`, `"string" = ""`).



### **G. Multi-Select Editing**
- Allow selecting multiple nodes and editing shared properties at once.



### **H. Node Color/Tagging**
- Add ability to assign a custom color or tag to nodes to visually group them.



## **Potential Revised Component Structure**

```
/components
  /nodes
    CustomNode.tsx           <-- just composes subcomponents
    NodeLabel.tsx
    InputHandle.tsx
    OutputHandle.tsx
  Checkbox.tsx
  OptionalLiteralSelect.tsx

/context
  UpdateDataContext.ts

/hooks
  useNodeValueUpdate.ts      <-- encapsulates UpdateDataContext + sendCommand
  useGraphModelSync.ts       <-- handles model.on/.../off

/utils
  commands.ts                <-- build* functions for backend commands
  typeParsing.ts             <-- parseLiteralType, unwrapOptionalType, convertInput
  styleVars.ts               <-- STYLE_VARS, colors, spacing
```
