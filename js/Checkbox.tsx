import React from "react";

interface CheckboxProps {
  checked: boolean;
  inputHeight: number;
  onToggle?: () => void;
  borderColor?: string;
}

export const Checkbox: React.FC<CheckboxProps> = ({
  checked,
  inputHeight,
  onToggle,
  borderColor = "#333",
}) => {
  const innerSize = inputHeight - 6;
  const checkSize = innerSize - 4;

  return (
    <div
      data-testid="checkbox-wrapper"
      className="nodrag"
      draggable={false}
      onClick={(e) => {
        e.stopPropagation();
        onToggle?.();
      }}
      style={{
        width: inputHeight,
        height: "100%",
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
            data-testid="checkbox-svg"
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