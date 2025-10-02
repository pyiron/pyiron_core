import React, { memo, useEffect, useState, useContext } from "react";
import {
  Handle,
  useUpdateNodeInternals,
  NodeToolbar,
  Position,
} from "@xyflow/react";
import { useModel } from "@anywidget/react";
import { UpdateDataContext } from "./widget";
import {
  OptionalLiteralSelect,
  STYLE_VARS as OLS_STYLE_VARS,
} from "./OptionalLiteralSelect";
import { parseLiteralType } from "./utils";
import { Checkbox } from "./Checkbox";

import "./CustomNode.css";

const STYLE_VARS = {
  inputHeight: OLS_STYLE_VARS.inputHeight,
  inputFontSize: OLS_STYLE_VARS.inputFontSize,
  inputWidth: OLS_STYLE_VARS.inputWidth,
  labelFontSize: 10,
  labelNodeFontSize: "bold" as const,
  labelMarginBottom: "0.3em",
  labelTextAlign: "center" as const,
  inputMarginLeft: 5,
  verticalRowSpacing: 20,
  customHandleTop: 30,
  highlightColor: OLS_STYLE_VARS.highlightColor,
};

export const SELF_OFFSET = { top: 9, right: 9 };

const stateColors: Record<string, string> = {
  NotRun: "lightgray",
  Success: "green",
  Failed: "red",
  LoadedFromCache: "yellow",
};

const CustomHandle = ({
  position,
  type,
  index,
  label,
  highlighted,
  isNodeType,
  customStyle,
}: any) => {
  const offsetStyle = position === Position.Left ? { left: -6 } : { right: -6 };
  const className = isNodeType ? "node-type-handle" : undefined;
  const dataAttrs = isNodeType ? { "data-nodetype": "true" } : {};

  const inlineStyle: React.CSSProperties = {
    top:
      index >= 0
        ? STYLE_VARS.customHandleTop +
          STYLE_VARS.verticalRowSpacing * index
        : undefined,
    backgroundColor: highlighted ? STYLE_VARS.highlightColor : undefined,
    ...offsetStyle,
    ...customStyle,
  };

  return (
    <Handle
      key={`${position}-${index}-${label}`}
      type={type}
      position={position}
      id={label}
      className={className}
      {...dataAttrs}
      style={inlineStyle}
    />
  );
};

const NodeLabel = ({
  text,
  onRename,
}: {
  text: string;
  onRename: (name: string) => void;
}) => {
  const [editing, setEditing] = useState(false);
  const [value, setValue] = useState(text);
  return (
    <div
      style={{
        fontWeight: STYLE_VARS.labelNodeFontSize,
        marginBottom: STYLE_VARS.labelMarginBottom,
        textAlign: STYLE_VARS.labelTextAlign,
        cursor: "pointer",
      }}
      onDoubleClick={() => setEditing(true)}
    >
      {editing ? (
        <input
          type="text"
          value={value}
          autoFocus
          onChange={(e) => setValue(e.target.value)}
          onBlur={() => {
            setEditing(false);
            if (value.trim() && value.trim() !== text) {
              onRename(value.trim());
            }
          }}
          onKeyDown={(e) => {
            if (e.key === "Enter") {
              setEditing(false);
              if (value.trim() && value.trim() !== text) {
                onRename(value.trim());
              }
            }
          }}
          style={{
            width: "100%",
            fontSize: STYLE_VARS.inputFontSize,
            boxSizing: "border-box",
          }}
        />
      ) : (
        text
      )}
    </div>
  );
};

/**
 * Renders an input handle for a custom node, supporting various input types and connection states.
 *
 * @param label - The label to display for the input.
 * @param type - The type of the input (e.g., "int", "float", "bool", "Node", "NonPrimitive", etc.).
 * @param value - The current value of the input.
 * @param index - The index of the input within the node.
 * @param context - Callback function to update the input value.
 * @param data - Additional data for the node, including connection state.
 * @param onHoverChange - Callback for hover state changes.
 * @param highlighted - Whether the input is currently highlighted.
 *
 * The component adapts its rendering based on the input type:
 * - For literal and optional types, renders a select or input with optional checkbox.
 * - For boolean types, renders a toggleable checkbox.
 * - For primitive types, renders a text or number input.
 * - For non-primitive types, disables editing and displays a gray label.
 *
 * Visual cues:
 * - Red border and asterisk indicate mandatory fields.
 * - Green text indicates a connected input.
 * - Gray color (`color: "gray"` or `color: "#555"`) is used for non-editable, non-primitive types.
 *   - `#555` is a darker gray for better contrast and accessibility.
 */
const InputHandle = ({
  label,
  type,
  value,
  index,
  context,
  data,
  onHoverChange,
  highlighted = false,
}: any) => {
  const { optional, options, mode, baseType } = parseLiteralType(type);
  const isLiteral = options.length > 0;
  const isNodeType = type === "Node";
  const isConnected = data.target_connected && data.target_connected[index];

  // NEW: detect non-primitive types
  const isNonPrimitive =
    baseType === "nonprimitive" || type === "NonPrimitive";

  const isMandatory = !isConnected && value === "NotData";
  const displayValue = isMandatory ? "" : value ?? "";

  const [editValue, setEditValue] = useState(displayValue || "");
  useEffect(() => {
    setEditValue(displayValue || "");
  }, [displayValue]);

  const pushValue = (raw: any) => {
    const conv = convertInput(raw, type);
    const finalVal = conv.error ? raw : conv.value;
    context(data.label, index, finalVal);
  };

  return (
    <div
      style={{
        height: STYLE_VARS.inputHeight,
        fontSize: STYLE_VARS.inputFontSize,
        display: "flex",
        alignItems: "center",
        flexDirection: "row-reverse",
        justifyContent: "flex-end",
        marginBottom:
          STYLE_VARS.verticalRowSpacing - STYLE_VARS.inputHeight,
      }}
      onMouseEnter={() => onHoverChange(index, true, "left")}
      onMouseLeave={() => onHoverChange(index, false, "left")}
    >
      <span
        style={{
          marginLeft: STYLE_VARS.inputMarginLeft,
          opacity: isConnected || isNonPrimitive ? 0.6 : 1,
          fontStyle:
            isConnected || isNonPrimitive ? "italic" : "normal",
          color: isNonPrimitive ? "gray" : "inherit",
          cursor: isNonPrimitive ? "not-allowed" : "default",
        }}
        title={
          isNonPrimitive
            ? "This value is not editable. Connect it to a node that provides a value with matching type."
            : ""
        }
      >
        {label}
        {isMandatory && (
          <span style={{ color: "red", marginLeft: 2 }}> *</span>
        )}
        {isConnected && (
          <span style={{ color: "green" }}>(connected)</span>
        )}
        {isNonPrimitive && (
            <span style={{ color: "#555", marginLeft: 4 }}>
            {/* (non-editable) */}
            </span>
        )}
      </span>

      <CustomHandle
        position={Position.Left}
        type="target"
        index={index}
        label={label}
        highlighted={highlighted}
        isNodeType={isNodeType}
      />

      {!isConnected && !isNonPrimitive && (
        optional || isLiteral ? (
          <OptionalLiteralSelect
            value={editValue || null}
            options={isLiteral ? (options as string[]) : []}
            onChange={(newVal, commit = false) => {
              setEditValue(newVal ?? "");
              if (commit) {
                pushValue(newVal ?? "");
              }
            }}
            highlighted={highlighted}
            showCheckbox={optional}
            mode={mode || (isLiteral ? "select" : "input")}
            inputType={
              baseType && (baseType === "int" || baseType === "float")
                ? "number"
                : "text"
            }
            style={{
              width: STYLE_VARS.inputWidth,
              boxSizing: "border-box",
              border: isMandatory
                ? "1px solid red"
                : "1px solid #ccc",
              color: isMandatory ? "gray" : "black",
              backgroundColor: isMandatory
                ? "#fff8c4"
                : "white",
            }}
            placeholder={isMandatory ? "required" : ""}
          />
        ) : type === "bool" ? (
          <div
            className="nodrag"
            draggable={false}
            style={{
              width: STYLE_VARS.inputWidth,
              height: STYLE_VARS.inputHeight,
              display: "flex",
              alignItems: "center",
              justifyContent: "space-between",
              border: isMandatory
                ? "1px solid red"
                : "1px solid #ccc",
              backgroundColor: isMandatory
                ? "#fffacd"
                : "white",
              cursor: "pointer",
              paddingLeft: "4px",
              boxSizing: "border-box",
            }}
            onClick={() => pushValue(!value)}
          >
            <span
              style={{
                color: isMandatory ? "gray" : "#888",
                fontSize: STYLE_VARS.inputFontSize,
              }}
            >
              {isMandatory ? "mandatory" : value ? "True" : "False"}
            </span>

            <Checkbox
              checked={!!value}
              inputHeight={STYLE_VARS.inputHeight}
              onToggle={() => pushValue(!value)}
            />
          </div>
        ) : (
          <input
            type={
              baseType === "int" || baseType === "float"
                ? "number"
                : "text"
            }
            step={baseType === "float" ? "any" : undefined}
            value={displayValue}
            placeholder={isMandatory ? "mandatory" : ""}
            onChange={(e) =>
              context(data.label, index, e.target.value)
            }
            onBlur={(e) => pushValue(e.target.value.trim())}
            style={{
              width: STYLE_VARS.inputWidth,
              height: STYLE_VARS.inputHeight,
              fontSize: STYLE_VARS.inputFontSize,
              border: isMandatory
                ? "1px solid red"
                : "1px solid #ccc",
              boxSizing: "border-box",
              padding: "0 4px",
              color: isMandatory ? "gray" : "black",
              backgroundColor: isMandatory
                ? "#fffacd"
                : "white",
            }}
          />
        )
      )}
    </div>
  );
};

const OutputHandle = ({
  label,
  index,
  onHoverChange,
  highlighted,
  type,
}: any) => {
  const isNodeType = type === "Node";
  return (
    <div
      style={{
        height: STYLE_VARS.inputHeight,
        fontSize: STYLE_VARS.inputFontSize,
        textAlign: "right",
        marginBottom:
          STYLE_VARS.verticalRowSpacing - STYLE_VARS.inputHeight,
      }}
      onMouseEnter={() => onHoverChange(index, true, "right")}
      onMouseLeave={() => onHoverChange(index, false, "right")}
    >
      {label}
      <CustomHandle
        position={Position.Right}
        type="source"
        index={index}
        label={label}
        highlighted={highlighted}
        isNodeType={isNodeType}
      />
    </div>
  );
};

const NodeComponent: React.FC<{ data: any }> = memo(({ data }) => {
  const updateNodeInternals = useUpdateNodeInternals();
  const model = useModel();
  const context = useContext(UpdateDataContext)!;

  const sendRenameCommand = (newName: string) => {
    model.set(
      "commands",
      `renameNode: ${JSON.stringify({
        oldLabel: data.label,
        newLabel: newName,
      })} - ${Date.now()}`
    );
    model.save_changes();
  };

  const sendNodeCommand = (cmd: string) => {
    model.set(
      "commands",
      `${cmd}: ${data.label} - ${Date.now()}`
    );
    model.save_changes();
  };

  const numHandles = Math.max(
    data.source_labels.length,
    data.target_labels.length
  );
  const [hoverState, setHoverState] = useState<
    Record<string, boolean>
  >({});

  const handleHoverChange = (
    idx: number,
    state: boolean,
    side: string
  ) => {
    setHoverState((prev) => ({
      ...prev,
      [`${side}-${idx}`]: state,
    }));
  };

  useEffect(() => {
    for (let idx = 0; idx < numHandles; idx++) {
      updateNodeInternals(`handle-${idx}`);
    }
  }, [numHandles, updateNodeInternals]);

  const selfIndex = data.source_labels.findIndex(
    (lbl: string) => lbl === "self"
  );

  return (
    <div
      style={{
        height:
          STYLE_VARS.customHandleTop +
          numHandles * STYLE_VARS.verticalRowSpacing,
        position: "relative",
      }}
    >
      <NodeLabel text={data.label} onRename={sendRenameCommand} />

      {selfIndex !== -1 && (
        <Handle
          type="source"
          position={Position.Right}
          id="self"
          style={{
            position: "absolute",
            top: SELF_OFFSET.top,
            right: SELF_OFFSET.right,
            width: 16,
            height: 16,
            borderRadius: "50%",
            border: "2px solid black",
            backgroundColor: "white",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
          }}
        >
          <div
            style={{
              width: 8,
              height: 8,
              borderRadius: "50%",
              backgroundColor:
                stateColors[data.source_values[selfIndex]] ||
                "white",
            }}
          ></div>
        </Handle>
      )}

      <div>
        {Array.from({ length: numHandles }).map((_, index) => (
          <div
            key={index}
            style={{
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
            }}
          >
            {index < data.target_labels.length ? (
              <InputHandle
                label={data.target_labels[index]}
                type={data.target_types[index]}
                value={data.target_values[index]}
                index={index}
                context={context}
                data={data}
                onHoverChange={handleHoverChange}
                highlighted={hoverState[`left-${index}`]}
              />
            ) : (
              <div></div>
            )}
            {index < data.source_labels.length &&
            data.source_labels[index] !== "self" ? (
              <OutputHandle
                label={data.source_labels[index]}
                index={index}
                onHoverChange={handleHoverChange}
                highlighted={hoverState[`right-${index}`]}
                type={data.source_types[index]}
              />
            ) : (
              <div></div>
            )}
          </div>
        ))}
      </div>

      <NodeToolbar
        isVisible={data.forceToolbarVisible || undefined}
        position={data.toolbarPosition}
      >
        <button onClick={() => sendNodeCommand("run")}>Run</button>
        <button onClick={() => sendNodeCommand("source")}>
          Source
        </button>
        <button
          onClick={() =>
            sendNodeCommand(
              data.expanded ? "collapse" : "expand"
            )
          }
        >
          {data.expanded ? "Collapse" : "Expand"}
        </button>
      </NodeToolbar>
    </div>
  );
});

export default NodeComponent;

export function unwrapOptionalType(typeStr: string): string {
  let cleaned = typeStr.replace(/['"]/g, "").trim();
  if (cleaned.toLowerCase().startsWith("optional[")) {
    cleaned = cleaned.substring(
      "optional[".length,
      cleaned.length - 1
    ).trim();
  }
  return cleaned.toLowerCase();
}

export function convertInput(value: any, inpType: string) {
  const base = unwrapOptionalType(inpType);
  switch (base) {
    case "int":
      const i = parseInt(value, 10);
      return isNaN(i) ? { value, error: true } : { value: i };
    case "float":
      const f = parseFloat(value);
      return isNaN(f) ? { value, error: true } : { value: f };
    default:
      return { value };
  }
}