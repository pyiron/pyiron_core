import React, { memo, useEffect, useState, useContext } from "react";
import { Handle, useUpdateNodeInternals, NodeToolbar } from "@xyflow/react";
import { useModel } from "@anywidget/react";
import { UpdateDataContext } from "./widget.jsx";

const STYLE_VARS = {
  inputHeight: 16,
  inputFontSize: 10,
  labelFontSize: 10, //"normal",
  labelNodeFontSize: "bold",
  labelMarginBottom: "0.3em",
  labelTextAlign: "center",
  inputWidth: 30,
  inputHeightSmall: 10,
  inputFontSizeSmall: 6,
  inputMarginLeft: 5,
  customHandleTop: 30,
  customHandleSpacing: 16,
};

const inputTypeMap = {
  str: "text",
  int: "text",
  float: "text",
  bool: "checkbox",
};

const getBackgroundColor = (value) => {
  if (value === null) return "grey";
  if (value === "NotData") return "yellow";
  return "white";
};

const convertInput = (value, inpType) => {
  const trimmedValue = typeof value === "string" ? value.trim() : value;
  switch (inpType) {
    case "int":
      return isNaN(parseInt(trimmedValue, 10))
        ? trimmedValue
        : parseInt(trimmedValue, 10);
    case "float":
      return isNaN(parseFloat(trimmedValue))
        ? trimmedValue
        : parseFloat(trimmedValue);
    case "bool":
      return trimmedValue;
    default:
      return trimmedValue;
  }
};

const CustomHandle = ({ position, type, index, label }) => (
  <Handle
    key={`${position}-${index}`}
    type={type}
    position={position}
    id={label}
    style={{
      top: STYLE_VARS.customHandleTop + STYLE_VARS.customHandleSpacing * index,
      fontSize: STYLE_VARS.inputFontSizeSmall,
    }}
  />
);

const Label = ({ text, isMandatory }) => (
  <div
    style={{
      // fontWeight: STYLE_VARS.labelFontSize,
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
      // fontSize: STYLE_VARS.labelFontSize,
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
}) => {
  const [inputValue, setInputValue] = useState(
    value === "NotData" ? "" : value === null ? "" : value
  );

  return (
    <>
      <div
        style={{
          height: STYLE_VARS.inputHeight,
          fontSize: STYLE_VARS.inputFontSize,
          display: "flex",
          alignItems: "center",
          flexDirection: "row-reverse",
          justifyContent: "flex-end",
        }}
      >
        <span style={{ marginLeft: STYLE_VARS.inputMarginLeft }}>
          {label}
          {isMandatory && <span style={{ color: "red" }}> *</span>}
        </span>
        {editValue && (
          <input
            type={inputTypeMap[type] || "text"}
            checked={type === "bool" ? inputValue : undefined}
            value={type !== "bool" ? inputValue : undefined}
            placeholder={value === "NotData" ? "enter data" : ""}
            className="nodrag"
            onChange={(e) => {
              const newValue =
                type === "bool" ? e.target.checked : e.target.value;
              setInputValue(newValue);
              context(data.label, index, newValue);
            }}
            onKeyDown={(e) => {
              if (e.key === "Enter") {
                const convertedValue = convertInput(inputValue, type);
                context(data.label, index, convertedValue);
              }
            }}
            onBlur={() => {
              const convertedValue = convertInput(inputValue, type);
              context(data.label, index, convertedValue);
            }}
            style={{
              width: STYLE_VARS.inputWidth,
              height: STYLE_VARS.inputHeightSmall,
              fontSize: STYLE_VARS.inputFontSizeSmall,
              backgroundColor: getBackgroundColor(value),
            }}
          />
        )}
      </div>
    </>
  );
};

const OutputHandle = ({ label, index }) => (
  <>
    <div
      style={{
        height: STYLE_VARS.inputHeight,
        fontSize: STYLE_VARS.inputFontSize,
        textAlign: "right",
      }}
    >
      {label}
    </div>
    <CustomHandle position="right" type="source" index={index} label={label} />
  </>
);

const NodeComponent = memo(({ data }) => {
  const updateNodeInternals = useUpdateNodeInternals();
  const model = useModel();
  const context = useContext(UpdateDataContext);
  const numHandles = Math.max(
    data.source_labels.length,
    data.target_labels.length
  );
  const [handles] = useState(Array(numHandles).fill({}));

  useEffect(() => {
    handles.forEach((_, index) => {
      updateNodeInternals(`handle-${index}`);
    });
  }, [handles, updateNodeInternals]);

  const handleCommand = (action) => {
    console.log(action, data.label);

    if (action === "collapse") {
      action = "expand";
    }

    model.set("commands", `${action}: ${data.label} - ${new Date().getTime()}`);
    model.save_changes();
  };

  return (
    <div>
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
                  />
                ) : (
                  <Label text={data.target_labels[index]} />
                )}
                <CustomHandle
                  position="left"
                  type="target"
                  index={index}
                  label={data.target_labels[index]}
                />
              </>
            ) : (
              <div></div>
            )}

            {index < data.source_labels.length ? (
              <OutputHandle label={data.source_labels[index]} index={index} />
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
          onClick={() => handleCommand(data.expanded ? "collapse" : "expand")}
        >
          {data.expanded ? "Collapse" : "Expand"}
        </button>
      </NodeToolbar>
    </div>
  );
});

export default NodeComponent;
