import React, { useState, useEffect } from "react";
import { Checkbox } from "./Checkbox";

export interface OptionalLiteralSelectProps {
  value: string | null;
  options: readonly string[];
  onChange: (v: string | null, commit?: boolean) => void; // commit flag
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

  // Sync with incoming props
  useEffect(() => {
    setUseValue(value !== null);
    if (value !== null) {
      setSelected(value);
    }
  }, [value]);

  const handleToggle = () => {
    const newChecked = !useValue;
    setUseValue(newChecked);
    if (!newChecked) {
      onChange(null, true); // commit null
    } else {
      onChange(selected, true); // commit existing
    }
  };

  const handleSelectChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const newVal = e.target.value;
    setSelected(newVal);
    if (!showCheckbox || useValue) {
      onChange(newVal, true); // select = instant commit
    }
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const newVal = e.target.value;
    setSelected(newVal);
    if (!showCheckbox || useValue) {
      onChange(newVal, false); // typing (no commit)
    }
  };

  const handleInputCommit = (val: string) => {
    const trimmed = val.trim();
    setSelected(trimmed);
    if (!showCheckbox || useValue) {
      onChange(trimmed, true); // commit
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter") {
      handleInputCommit((e.target as HTMLInputElement).value);
    }
  };

  const handleBlur = (e: React.FocusEvent<HTMLInputElement>) => {
    handleInputCommit(e.target.value);
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
        fontSize: STYLE_VARS.inputFontSize,
      }}
    >
      {mode === "select" ? (
        <div style={{ position: "relative", flexGrow: 1, height: "100%" }}>
          <select
            className="nodrag"
            draggable={false}
            value={selectValue}
            onChange={handleSelectChange}
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
        </div>
      ) : (
        <input
          type={inputType}
          step={inputType === "number" ? "any" : undefined}
          className="nodrag"
          draggable={false}
          value={useValue ? (selected as string) : ""}
          placeholder={showCheckbox && !useValue ? "default" : ""}
          onChange={handleInputChange}
          onKeyDown={handleKeyDown}   // commit on Enter
          onBlur={handleBlur}         // commit on Blur
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