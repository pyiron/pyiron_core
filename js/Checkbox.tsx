// Checkbox.tsx
import React from "react";

interface CheckboxProps {
  checked: boolean;
  inputHeight: number; // pass STYLE_VARS.inputHeight
  onToggle?: () => void;
  borderColor?: string;
}

export const Checkbox: React.FC<CheckboxProps> = ({
  checked,
  inputHeight,
  onToggle,
  borderColor = "#333",
}) => {
  const innerSize = inputHeight - 6; // matches OptionalLiteralSelect
  const checkSize = innerSize - 4;   // matches the SVG size from OptionalLiteralSelect

  return (
    <div
      className="nodrag"
      draggable={false}
      onClick={(e) => {
        e.stopPropagation();
        onToggle?.();
      }}
      style={{
        width: inputHeight, // identical outer clickable width
        height: "100%",     // height from parent = STYLE_VARS.inputHeight
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        cursor: "pointer",
        userSelect: "none",
      }}
    >
      <div
        style={{
          width: innerSize,
          height: innerSize,
          border: `1px solid ${borderColor}`,
          backgroundColor: "transparent",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
        }}
      >
        {checked && (
          <svg
            xmlns="http://www.w3.org/2000/svg"
            width={checkSize}
            height={checkSize}
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
  );
};