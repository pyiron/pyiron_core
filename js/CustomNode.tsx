// CustomNode.tsx
import React, {
  memo,
  useEffect,
  useState,
  useContext,
  useCallback,
  useRef,
} from "react";
import { Handle, useUpdateNodeInternals, NodeToolbar } from "@xyflow/react";
import { useModel } from "@anywidget/react";
import { UpdateDataContext } from "./widget.jsx";
import {
  OptionalLiteralSelect,
  STYLE_VARS as OLS_STYLE_VARS,
} from "./OptionalLiteralSelect.tsx";
import { parseLiteralType } from "./utils.ts";

/* Styling constants */
const STYLE_VARS = {
  inputHeight: OLS_STYLE_VARS.inputHeight,
  inputFontSize: OLS_STYLE_VARS.inputFontSize,
  inputWidth: OLS_STYLE_VARS.inputWidth,
  labelFontSize: 10,
  labelNodeFontSize: "bold",
  labelMarginBottom: "0.3em",
  labelTextAlign: "center",
  inputMarginLeft: 5,
  verticalRowSpacing: 20,
  customHandleTop: 30,
  highlightColor: OLS_STYLE_VARS.highlightColor,
};

const inputTypeMap: Record<string, string> = {
  str: "text",
  int: "text",
  float: "text",
  bool: "checkbox",
};

const getBackgroundColor = (value: any, highlighted: boolean) => {
  if (highlighted) return STYLE_VARS.highlightColor;
  if (value === null) return "grey";
  if (value === "NotData") return "yellow";
  return "white";
};

const convertInput = (value: any, inpType: string) => {
  const trimmedValue = typeof value === "string" ? value.trim() : value;
  switch (inpType) {
    case "int": {
      const i = parseInt(trimmedValue, 10);
      return isNaN(i) ? { value: trimmedValue, error: true } : { value: i };
    }
    case "float": {
      const f = parseFloat(trimmedValue);
      return isNaN(f) ? { value: trimmedValue, error: true } : { value: f };
    }
    case "bool":
      return { value: trimmedValue };
    default:
      return { value: trimmedValue };
  }
};

const CustomHandle = ({ position, type, index, label, highlighted }) => (
  <Handle
    key={`${position}-${index}`}
    type={type}
    position={position}
    id={label}
    style={{
      top: STYLE_VARS.customHandleTop + STYLE_VARS.verticalRowSpacing * index,
      fontSize: STYLE_VARS.inputFontSize,
      backgroundColor: highlighted ? STYLE_VARS.highlightColor : undefined,
      transition: "background-color 0.2s ease",
    }}
  />
);

const Label = ({ text, isMandatory }) => (
  <div
    style={{
      fontSize: STYLE_VARS.inputFontSize,
      marginBottom: STYLE_VARS.labelMarginBottom,
      textAlign: STYLE_VARS.labelTextAlign,
    }}
  >
    {text}
    {isMandatory && <span style={{ color: "red" }}> *</span>}
  </div>
);

const NodeLabel = ({ text }) => (
  <div
    style={{
      fontWeight: STYLE_VARS.labelNodeFontSize,
      marginBottom: STYLE_VARS.labelMarginBottom,
      textAlign: STYLE_VARS.labelTextAlign,
    }}
  >
    {text}
  </div>
);

const InputHandle = ({
  label,
  type,
  value,
  editValue,
  index,
  context,
  data,
  isMandatory,
  onHoverChange,
  highlighted = false,
}) => {
  const { optional, options, mode, baseType } = parseLiteralType(type);
  const isLiteral = options.length > 0;

  const [inputValue, setInputValue] = useState(
    value === "NotData" || value === null ? "" : value
  );
  const [hasError, setHasError] = useState(false);

  const debouncedContext = useCallback(
    debounce((nodeLabel, idx, newVal) => {
      context(nodeLabel, idx, newVal);
    }, 300),
    [context]
  );

  const debounceRef = useRef(debouncedContext);
  useEffect(() => {
    debounceRef.current = debouncedContext;
    return () => {
      debounceRef.current?.cancel?.();
    };
  }, [debouncedContext]);

  const pushValue = (raw) => {
    const conv = convertInput(raw, type);
    setHasError(conv.error ?? false);
    const finalVal = conv.error ? raw : conv.value;
    debouncedContext(data.label, index, finalVal);
  };

  const baseInputStyle = {
    width: STYLE_VARS.inputWidth,
    height: STYLE_VARS.inputHeight,
    fontSize: STYLE_VARS.inputFontSize,
    backgroundColor: getBackgroundColor(value, highlighted),
    border: hasError ? "1px solid red" : "1px solid #ccc",
    boxSizing: "border-box",
    transition: "background-color 0.2s ease",
  };

  const htmlInputType =
    baseType && (baseType === "int" || baseType === "float") ? "number" : "text";

  return (
    <div
      style={{
        height: STYLE_VARS.inputHeight,
        fontSize: STYLE_VARS.inputFontSize,
        display: "flex",
        alignItems: "center",
        flexDirection: "row-reverse",
        justifyContent: "flex-end",
        marginBottom: STYLE_VARS.verticalRowSpacing - STYLE_VARS.inputHeight,
      }}
      onMouseEnter={() => onHoverChange(index, true, "left")}
      onMouseLeave={() => onHoverChange(index, false, "left")}
    >
      <span style={{ marginLeft: STYLE_VARS.inputMarginLeft }}>
        {label}
        {isMandatory && <span style={{ color: "red" }}> *</span>}
      </span>

      {editValue &&
        (optional || isLiteral ? (
          <OptionalLiteralSelect
            value={value ?? null}
            options={isLiteral ? (options as readonly string[]) : []}
            onChange={(newVal) => pushValue(newVal)}
            highlighted={highlighted}
            showCheckbox={optional}
            mode={mode || (isLiteral ? "select" : "input")}
            inputType={htmlInputType}
          />
        ) : type === "bool" ? (
          <div
            className="nodrag"
            draggable={false}
            onClick={() => {
              const newVal = !(inputValue as boolean);
              setInputValue(newVal);
              pushValue(newVal);
            }}
            style={{
              width: STYLE_VARS.inputWidth,
              height: STYLE_VARS.inputHeight,
              display: "flex",
              alignItems: "center",
              backgroundColor: getBackgroundColor(value, highlighted),
              border: "1px solid #ccc",
              boxSizing: "border-box",
              transition: "background-color 0.2s ease",
              cursor: "pointer",
              fontSize: STYLE_VARS.inputFontSize,
            }}
          >
            <div style={{ flexGrow: 1, textAlign: "center" }}>
              {inputValue ? "True" : "False"}
            </div>
            <div style={{ width: "1px", height: "100%", backgroundColor: "#ccc" }}></div>
            <div
              style={{
                width: STYLE_VARS.inputHeight,
                height: "100%",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
              }}
            >
              <div
                style={{
                  width: STYLE_VARS.inputHeight - 6,
                  height: STYLE_VARS.inputHeight - 6,
                  border: "1px solid #333",
                  backgroundColor: "transparent",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                }}
              >
                {inputValue && (
                  <svg
                    xmlns="http://www.w3.org/2000/svg"
                    width={STYLE_VARS.inputHeight - 10}
                    height={STYLE_VARS.inputHeight - 10}
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="black"
                    strokeWidth="3"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  >
                    <polyline points="20 6 9 17 4 12" />
                  </svg>
                )}
              </div>
            </div>
          </div>
        ) : (
          <input
            type={htmlInputType}
            value={inputValue as string | number}
            placeholder={value === "NotData" ? "enter data" : ""}
            className="nodrag"
            onChange={(e) => {
              const newValue = e.target.value;
              setInputValue(newValue);
              debounceRef.current(data.label, index, newValue);
            }}
            onKeyDown={(e) => {
              if (e.key === "Enter") pushValue(inputValue);
            }}
            onBlur={() => pushValue(inputValue)}
            style={baseInputStyle}
          />
        ))}
    </div>
  );
};

const OutputHandle = ({ label, index, onHoverChange, highlighted }) => (
  <div
    style={{
      height: STYLE_VARS.inputHeight,
      fontSize: STYLE_VARS.inputFontSize,
      textAlign: "right",
      marginBottom: STYLE_VARS.verticalRowSpacing - STYLE_VARS.inputHeight,
    }}
    onMouseEnter={() => onHoverChange(index, true, "right")}
    onMouseLeave={() => onHoverChange(index, false, "right")}
  >
    {label}
    <CustomHandle
      position="right"
      type="source"
      index={index}
      label={label}
      highlighted={highlighted}
    />
  </div>
);

type NodeData = {
  label: string;
  source_labels: string[];
  source_types: string[];
  source_values: any[];
  target_labels: string[];
  target_types: string[];
  target_values: any[];
  forceToolbarVisible?: boolean;
  toolbarPosition?: any;
  expanded?: boolean;
};

const NodeComponent: React.FC<{ data: NodeData }> = memo(({ data }) => {
  const updateNodeInternals = useUpdateNodeInternals();
  const model = useModel();
  const context = useContext(UpdateDataContext);

  const numHandles = Math.max(
    data.source_labels.length,
    data.target_labels.length
  );
  const [handles] = useState(Array(numHandles).fill({}));

  const [hoverState, setHoverState] = useState({});
  const handleHoverChange = (idx, state, side) => {
    setHoverState((prev) => ({ ...prev, [`${side}-${idx}`]: state }));
  };

  useEffect(() => {
    handles.forEach((_, index) => {
      updateNodeInternals(`handle-${index}`);
    });
  }, [handles, updateNodeInternals]);

  const handleCommand = (action: string) => {
    if (action === "collapse") {
      action = "expand";
    }
    model.set(
      "commands",
      `${action}: ${data.label} - ${new Date().getTime()}`
    );
    model.save_changes();
  };

  const nodeHeight =
    STYLE_VARS.customHandleTop + numHandles * STYLE_VARS.verticalRowSpacing;

  return (
    <div style={{ height: nodeHeight }}>
      <NodeLabel text={data.label} />
      <div>
        {handles.map((_, index) => (
          <div
            key={index}
            style={{
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
            }}
          >
            {index < data.target_labels.length ? (
              <>
                {data.target_types[index] !== "None" &&
                data.target_types[index] !== "NonPrimitive" ? (
                  <InputHandle
                    label={data.target_labels[index]}
                    type={data.target_types[index]}
                    value={data.target_values[index]}
                    editValue={true}
                    index={index}
                    context={context}
                    data={data}
                    isMandatory={data.target_values[index] === "NotData"}
                    onHoverChange={handleHoverChange}
                    highlighted={hoverState[`left-${index}`]}
                  />
                ) : (
                  <Label
                    text={data.target_labels[index]}
                    isMandatory={false}
                  />
                )}
                <CustomHandle
                  position="left"
                  type="target"
                  index={index}
                  label={data.target_labels[index]}
                  highlighted={hoverState[`left-${index}`]}
                />
              </>
            ) : (
              <div></div>
            )}

            {index < data.source_labels.length ? (
              <OutputHandle
                label={data.source_labels[index]}
                index={index}
                onHoverChange={handleHoverChange}
                highlighted={hoverState[`right-${index}`]}
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
        <button onClick={() => handleCommand("run")}>Run</button>
        <button onClick={() => handleCommand("source")}>Source</button>
        <button
          onClick={() =>
            handleCommand(data.expanded ? "collapse" : "expand")
          }
        >
          {data.expanded ? "Collapse" : "Expand"}
        </button>
      </NodeToolbar>
    </div>
  );
});

export default NodeComponent;

function debounce(func, wait) {
  let timeout;
  const debounced = (...args) => {
    clearTimeout(timeout);
    timeout = setTimeout(() => func(...args), wait);
  };
  debounced.cancel = () => clearTimeout(timeout);
  return debounced;
}