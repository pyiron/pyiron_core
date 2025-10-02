
# **Project Javascript Documentation**


## 1️⃣ **`Checkbox.tsx`** — Simple Custom Checkbox Component

### **Purpose**
A light-weight, styled checkbox UI element used inside other components (including `OptionalLiteralSelect` and `CustomNode`).  
It doesn’t use native `<input type="checkbox">` but emulates it with HTML/CSS and an SVG checkmark.

### **Exports**
- **`Checkbox`** *(React.FC)*



### **Props**

| Prop         | Type       | Required | Default  | Description |
|--------------|-----------|----------|----------|-------------|
| **`checked`** | `boolean` | yes      | —        | Whether the checkmark is shown (true) or hidden (false). |
| **`inputHeight`** | `number` | yes | — | Outer clickable height in px (matches UI constants like `STYLE_VARS.inputHeight`). Determines sizing of border box and SVG. |
| **`onToggle`** | `() => void` | no | — | Click handler toggling the state (optional). |
| **`borderColor`** | `string` | no | `#333` | Border color of the checkbox square. |



### **Behavior**
- Calculates `innerSize = inputHeight - 6` (inner box)  
- Calculates `checkSize = innerSize - 4` (SVG checkmark size)
- When `checked` is true, renders an inline SVG checkmark `<polyline>` shaped like ✓.  
- Click:
  - Stops propagation (`e.stopPropagation()`)
  - Calls `onToggle` if provided  
- Styles:
  - Outer wrapper: flex-centered, `cursor:pointer`  
  - Inner box: `border: 1px solid {borderColor}`  



### **Usage Example**

```tsx
import { Checkbox } from './Checkbox';

<Checkbox
  checked={isSelected}
  inputHeight={20}
  onToggle={() => setIsSelected((v) => !v)}
  borderColor="blue"
/>
```



## 2️⃣ **`OptionalLiteralSelect.tsx`** — Select/Input with Optional Checkbox

### **Purpose**
A generic UI component for selecting or inputting a literal value with optional enable/disable checkbox logic.  
Supports two modes:
- **Dropdown select**  
- **Free-text/number input**  

Used to represent optional parameters in a node’s UI.



### **Exports**
- **`OptionalLiteralSelect`** *(React.FC)*  
- **`STYLE_VARS`**: numeric and color constants for sizing/highlighting



### **Props (interface `OptionalLiteralSelectProps`)**
| Prop | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `value` | `string \| null` | yes | — | Current value (null = "default"/disabled) |
| `options` | `readonly string[]` | yes | — | List of selectable options (empty for free input) |
| `onChange` | `(v: string \| null, commit?: boolean) => void` | yes | — | Called on user input/change; `commit` indicates if change is final |
| `highlighted` | boolean | no | false | If true, highlight background |
| `onHoverChange` | `(state: boolean) => void` | no | — | Hover state callback |
| `showCheckbox` | boolean | no | true | Show/hide the optional-enabler checkbox |
| `mode` | `"select" \| "input"` | no | `"select"` | UI mode |
| `inputType` | string | no | `"text"` | HTML input type (e.g., `"number"`) |



### **Behavior**
- Manages internal `useValue` (whether the value is active) and `selected` states.
- Syncs `value` prop → internal states in `useEffect`.
- **Checkbox toggle**:
  - If turned off → commits `null`
  - If turned on → commits current selected value
- If `mode === "select"`: shows `<select>` dropdown with optional initial `"default"` option.
- If `mode === "input"`: shows `<input>` text/number box.
- Input changes fire `onChange` with `commit=false` for intermediate edits, and commit on Enter/Blur.
- Applies styles from `STYLE_VARS`.



### **Usage Example**
```tsx
<OptionalLiteralSelect
  value={currentVal}
  options={["low", "medium", "high"]}
  onChange={(val, commit) => console.log(val, commit)}
  highlighted={hovered}
  showCheckbox={true}
  mode="select"
/>
```



## 3️⃣ **`CustomNode.tsx`** — Rich React Flow Node UI

### **Purpose**
A complex custom node for use in a React Flow graph that supports:
- Input handles with literal selects, booleans, numeric/text inputs
- Output handles
- Node label editing
- Node status color indicators
- Integrated toolbar buttons



### **Key Exports**
- Default: `NodeComponent` *(memoized React.FC for `data: any`)*
- Named constants: `SELF_OFFSET` (positioning), `STYLE_VARS`



### **Composition**
- **`CustomHandle`**: wraps React Flow `<Handle>` with styling for custom node types
- **`NodeLabel`**: editable label (double-click → changes, commit on Enter/Blur)
- **`InputHandle`**:  
  - Parses type string → optional display (`parseLiteralType` from `utils.ts`)
  - Based on type: shows `OptionalLiteralSelect`, boolean checkbox, or `<input>`
  - Manages commit logic and passes updates to context `UpdateDataContext`
- **`OutputHandle`**: basic label + handle



### **Main NodeComponent Behavior**
- Receives `data` with labels, types, values
- Sets up `hoverState` for handle highlighting
- Renders:
  - Self handle if present ('self' output label) with color from `stateColors`
  - Column of `<InputHandle>`s for target labels/types
  - Column of `<OutputHandle>`s for source labels/types
- Toolbar with buttons → calls `sendNodeCommand` to `model.set("commands", "...")`



### **Usage in Graph**
```tsx
<ReactFlow nodeTypes={{ customNode: CustomNode }}>
  ...
</ReactFlow>
```

Where `data` must contain `source_labels`, `target_labels`, etc.



## 4️⃣ **`widget.tsx`** — The Entire Flow Editor Widget

### **Purpose**
Implements the actual visual editor:
- **Uses React Flow** for rendering the graph
- Loads/saves state via `@anywidget/react` model APIs (`useModel`)
- Syncs with backend data via model change events
- Provides context `UpdateDataContext` to child nodes



### **Exports**
- Default: `{ render }` — A `createRender` wrapper exposing a `.render` component for embedding as anywidget.
- Named: `UpdateDataContext`: React context for `(nodeLabel, handleIndex, newValue) => void`



### **Key Internal Functions:**
- `sendCommand(command, payload?)`: push to `model.set("commands", "...")` and save
- `updateData(nodeLabel, handleIndex, newValue)`: sends `"change_node_value"` command and updates local `nodes` state
- `onNodesChangeHandler`: updates nodes, selected nodes list, sends `"selected_nodes"` command
- `onEdgesChangeHandler`: updates edges in state/model
- `onConnectHandler`: adds edge in state/model, sends `"add_edge"`
- `handleKeyDown`: intercepts Delete/Backspace → sends `"delete_node"` only, filters edges locally
- `onEdgesDelete`: sends `"delete_edge"` for manual deletions
- `workflowCommands`: functions bound to toolbar controls (refresh, saveFlow, restoreFlow, clearFlow, groupNodes, saveWorkflowName)



### **Rendered Structure**
- `<UpdateDataContext.Provider value={updateData}>`
- `<ReactFlow>` configured with node/edge data and handlers
  - `<Background>` / `<MiniMap>` / `<Controls>` / `<Panel>` with control buttons



## 5️⃣ **`utils.ts`** — Helper Functions

From our earlier references & tests:

### **`parseLiteralType(type: string)`**
Given a type string like:
- `"Optional[int]"`
- `"Literal['A','B']"`
- `"bool"`
Returns:
```ts
{
  optional: boolean,
  options: string[],
  mode: 'select' | 'input',
  baseType: string
}
```
Used by `InputHandle` to decide how to render the editor for a value.



### **`unwrapOptionalType(typeStr: string): string`**
- Removes quotes, trims, and if the string starts with `"optional["`, strips that wrapper and lowercases the inner type.



### **`convertInput(value: any, inpType: string)`**
- Unwraps optional, then tries to parse based on type:
  - `"int"` → parseInt, check NaN
  - `"float"` → parseFloat, check NaN
  - Default: return as string
- Returns `{ value, error?: true }`.



## 📌 **Module Interactions**
- `widget.tsx` uses `CustomNode` as a React Flow node type.
- `CustomNode` uses `OptionalLiteralSelect` and `Checkbox` for value editing.
- `OptionalLiteralSelect` uses `Checkbox` for enabling/disabling the value.
- `utils.ts` is used in `CustomNode`’s InputHandle for type parsing & value conversion.



## ✅ **Summary**
The js project is a **React Flow graph editor widget** with:
- Highly customizable **node inputs** (literal select, free input, booleans)
- **Optional/enable-able values** with `Checkbox`
- Integration with a backend model via `@anywidget/react`
- Complete keyboard/mouse/toolbar controls for manipulating the graph
- Utilities for **parsing node input types** and converting user text to typed values

Perfect ✅ — let’s make an **architecture diagram** that shows exactly how the modules you’ve provided interact with each other in your React Flow-based widget.



## Architecture Diagram – Module Relationships

                      ┌───────────────────────────┐
                      │       utils.ts            │
                      │ ───────────────────────── │
                      │ • parseLiteralType()      │
                      │ • unwrapOptionalType()    │
                      │ • convertInput()          │
                      └───────────▲───────────────┘
                                  │ used by
                                  │
               ┌──────────────────┴──────────────────┐
               │                                     │
┌───────────────────────────┐              ┌───────────────────────────┐
│     OptionalLiteralSelect │              │        CustomNode         │
│ ───────────────────────── │              │ ───────────────────────── │
│ • Select or input field   │              │ • React Flow custom node  │
│   with optional enable     │              │ • Renders InputHandles /  │
│   checkbox                 │              │   OutputHandles           │
│ • Props: value, options,   │              │ • Uses parseLiteralType    │
│   onChange, showCheckbox…  │              │   & convertInput           │
│ • Uses Checkbox component  │              │ • Renders OptionalLiteral  │
│   for enabling/disabling   │              │   Select or Checkbox or    │
└───────────▲────────────────┘              │   HTML <input>            │
            │ uses                           │ • Label editing           │
            │                                └───────────▲───────────────┘
   ┌────────┴───────┐                            used by │
   │    Checkbox    │                              as    │ nodeTypes
   │ ───────────────│                                   │
   │ • Styled div    │                                   │
   │   checkbox      │                                   │
   │ • Shows SVG ✓   │                                   │
   │   when checked  │                                   │
   │ • onToggle()    │                                   │
   └────────────────┘                                   │
                                                        │
                                   ┌────────────────────┴─────────────────────┐
                                   │                  widget.tsx              │
                                   │ ───────────────────────────────────────  │
                                   │ • Main anywidget React Flow component    │
                                   │ • Sets up ReactFlow with nodeTypes:      │
                                   │     { customNode: CustomNode }           │
                                   │ • Provides UpdateDataContext for nodes   │
                                   │ • Handles nodes/edges data, connect,     │
                                   │   delete, toolbar commands               │
                                   │ • Talks to backend via useModel()        │
                                   └──────────────────────────────────────────┘



### **Flow Explanation**
1. **`widget.tsx`**  
   - Entry point for the flow editor widget.  
   - Creates a React Flow editor (`<ReactFlow>`) and supplies `nodeTypes` containing `CustomNode`.
   - Provides a context `UpdateDataContext` to allow child nodes (`CustomNode`) to send value updates back to the widget’s model.
   - Handles all graph events: node/edge changes, connects, deletes, toolbar actions.
   - Talks to backend via `@anywidget/react` model API (`useModel`).

2. **`CustomNode.tsx`**  
   - Defines how custom nodes look and behave.
   - Uses `<InputHandle>` for each input handle:
     - Calls `parseLiteralType` (from `utils.ts`) to decide which UI to render.
     - Uses `OptionalLiteralSelect` for literal/optional values.
     - Uses `Checkbox` for booleans, or HTML `<input>` for numbers/text.
     - Commits changes via `UpdateDataContext`.
   - Uses `<OutputHandle>` for outputs.
   - Renders an editable `<NodeLabel>` header.
   - Applies hover and status styles.

3. **`OptionalLiteralSelect.tsx`**  
   - Shows a dropdown `<select>` or `<input>` field for a value.
   - Can include an enabling/disabling `<Checkbox>` on the side.
   - Manages internal state to support partially committed edits.
   - Calls its `onChange` prop when the user updates the value, indicating whether the change is a commit.

4. **`Checkbox.tsx`**  
   - The basic building block for optional toggles.
   - Styled clickable box with SVG checkmark when checked.
   - Calls `onToggle` when clicked.
   - Used by `OptionalLiteralSelect` and `CustomNode` boolean inputs.

5. **`utils.ts`**  
   - Pure functions for parsing type information and converting input values:
     - `parseLiteralType(type)` – returns `optional`, `options`, `mode`, `baseType`.
     - `unwrapOptionalType(typeStr)` – strips `Optional[...]` wrapper.
     - `convertInput(value, type)` – converts string input to int, float, etc.
   - No React; used inside `CustomNode`.



## **Interaction Summary**

- The **UI tree**:  
  `widget.tsx` (top-level editor) → React Flow → `CustomNode` → `InputHandle` → One of:
    - `OptionalLiteralSelect` → `Checkbox`  
    - `Checkbox` (boolean)  
    - Plain `<input>`

- **Data flow**:  
  `Checkbox` / `OptionalLiteralSelect` → Calls `onChange` / `onToggle` → `InputHandle` → Calls `UpdateDataContext` → `widget.tsx` → Updates `nodes` state and sends backend command

- **Type parsing**:  
  When rendering inputs, `CustomNode` calls `parseLiteralType` (utils) to decide:
  - Optional? → show `Checkbox`
  - Options array non-empty? → show select
  - Base type numeric? → input type number



Do you want me to also prepare a **sequence diagram** that shows what happens when a user changes a node value (e.g., toggles a checkbox or edits a select field) so you can document the event flow end-to-end? That would make the documentation even clearer for onboarding developers.

## Unit tests

```
> npm install --save-dev jest ts-jest @testing-library/react @testing-library/jest-dom @testing-library/user-event
> npm run test -- --coverage

> export PYTHONPATH=$PYTHONPATH:/Users/jorgneugebauer/git_libs/pyiron_core
> pytest --cov=pyiron_database --cov=pyiron_nodes --cov=pyiron_workflow --cov-report=term-missing tests/

# or for html output
> pytest --cov=pyiron_database --cov=pyiron_nodes --cov=pyiron_workflow --cov-report=html tests/

```