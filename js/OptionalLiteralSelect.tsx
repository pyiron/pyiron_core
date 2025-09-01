// OptionalLiteralSelect.tsx
import React, { useState, useEffect } from "react";
import { Checkbox } from "./Checkbox"; // at top

export interface OptionalLiteralSelectProps {
  value: string | null;
  options: readonly string[];
  onChange: (v: string | null) => void;
  highlighted?: boolean;
  onHoverChange?: (state: boolean) => void;
  showCheckbox?: boolean;
  mode?: "select" | "input";
  inputType?: string;
}

export const STYLE_VARS = {
  inputWidth: 80,
  inputHeight: 20,
  inputFontSize: 10,
  highlightColor: "#d0ebff",
};

export const OptionalLiteralSelect: React.FC<OptionalLiteralSelectProps> = ({
  value,
  options,
  onChange,
  highlighted = false,
  onHoverChange,
  showCheckbox = true,
  mode = "select",
  inputType = "text",
}) => {
  const [useValue, setUseValue] = useState(value !== null);
  const [selected, setSelected] = useState(value ?? options[0] ?? "");

  useEffect(() => {
    setUseValue(value !== null);
    if (value !== null) setSelected(value);
  }, [value]);

  const handleToggle = () => {
    const newChecked = !useValue;
    setUseValue(newChecked);
    if (!newChecked) {
      onChange(null);
    } else {
      onChange(selected);
    }
  };

  const handleSelect = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const newVal = e.target.value;
    setSelected(newVal);
    if (!showCheckbox || useValue) onChange(newVal);
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const newVal = e.target.value;
    setSelected(newVal);
    if (!showCheckbox || useValue) onChange(newVal);
  };

  const selectValue = showCheckbox && !useValue ? "__default__" : selected;

  return (
    <div
      onMouseEnter={() => onHoverChange?.(true)}
      onMouseLeave={() => onHoverChange?.(false)}
      style={{
        width: STYLE_VARS.inputWidth,
        height: STYLE_VARS.inputHeight,
        display: "flex",
        alignItems: "center",
        backgroundColor: highlighted ? STYLE_VARS.highlightColor : "white",
        border: "1px solid #ccc",
        boxSizing: "border-box",
        transition: "background-color 0.2s ease",
        fontSize: STYLE_VARS.inputFontSize,
      }}
    >
      {mode === "select" ? (
        <div style={{ position: "relative", flexGrow: 1, height: "100%" }}>
          <select
            className="nodrag"
            draggable={false}
            value={selectValue}
            onChange={handleSelect}
            style={{
              width: "100%",
              height: "100%",
              border: "none",
              background: "none",
              outline: "none",
              padding: "0 16px 0 4px",
              margin: 0,
              appearance: "none",
              WebkitAppearance: "none",
              MozAppearance: "none",
              color: showCheckbox && !useValue ? "#888" : "inherit",
              fontSize: "inherit",
              cursor: showCheckbox && !useValue ? "default" : "pointer",
            }}
          >
            {showCheckbox && (
              <option value="__default__" disabled>
                default
              </option>
            )}
            {options.map((opt) => (
              <option key={opt} value={opt}>
                {opt}
              </option>
            ))}
          </select>
          <svg
            xmlns="http://www.w3.org/2000/svg"
            viewBox="0 0 10 6"
            width="10"
            height="6"
            style={{
              position: "absolute",
              right: 4,
              top: "50%",
              transform: "translateY(-50%)",
              pointerEvents: "none",
              fill: "black",
              opacity: showCheckbox && !useValue ? 0.3 : 0.7,
            }}
          >
            <path d="M0 0h10L5 6z" />
          </svg>
        </div>
      ) : (
        <input
          type={inputType}
          className="nodrag"
          draggable={false}
          value={useValue ? (selected as string) : ""}
          placeholder={showCheckbox && !useValue ? "default" : ""}
          onChange={handleInputChange}
          style={{
            width: "100%",
            height: "100%",
            border: "none",
            outline: "none",
            padding: "0 4px",
            margin: 0,
            background: "none",
            color: showCheckbox && !useValue ? "#888" : "inherit",
            fontSize: "inherit",
          }}
          disabled={showCheckbox && !useValue}
        />
      )}

      {showCheckbox && (
        <>
        <div style={{ width: "1px", height: "100%", backgroundColor: "#ccc" }}></div>
        <Checkbox
          checked={useValue}
          inputHeight={STYLE_VARS.inputHeight}
          onToggle={handleToggle}
        />
        </>
      )}
    </div>
  );
};