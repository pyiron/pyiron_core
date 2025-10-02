import React from "react";
import { render, screen, fireEvent } from "@testing-library/react";
import { Checkbox } from "../Checkbox";

describe("Checkbox Component", () => {
  it("renders without SVG when unchecked", () => {
    render(<Checkbox checked={false} inputHeight={20} />);
    expect(screen.queryByTestId("checkbox-svg")).not.toBeInTheDocument();
  });

  it("renders SVG checkmark when checked", () => {
    render(<Checkbox checked={true} inputHeight={20} />);
    expect(screen.getByTestId("checkbox-svg").tagName.toUpperCase()).toBe("SVG");
  });

  it("uses provided borderColor", () => {
    render(
      <Checkbox checked={false} inputHeight={20} borderColor="red" />
    );
    const wrapper = screen.getByTestId("checkbox-wrapper");
    const innerDiv = wrapper.querySelector("div");
    expect(innerDiv).toHaveStyle("border: 1px solid red");
  });

  it("calls onToggle when clicked", () => {
    const onToggleMock = jest.fn();
    render(<Checkbox checked={false} inputHeight={20} onToggle={onToggleMock} />);
    fireEvent.click(screen.getByTestId("checkbox-wrapper"));
    expect(onToggleMock).toHaveBeenCalledTimes(1);
  });

  it("does not throw if onToggle is not provided", () => {
    render(<Checkbox checked={false} inputHeight={20} />);
    fireEvent.click(screen.getByTestId("checkbox-wrapper"));
    // Should not throw
  });

  it("has correct innerSize and checkSize", () => {
    const inputHeight = 30;
    const innerSize = inputHeight - 6;
    const checkSize = innerSize - 4;
    render(<Checkbox checked={true} inputHeight={inputHeight} />);
    const wrapper = screen.getByTestId("checkbox-wrapper");
    const innerDiv = wrapper.querySelector("div");
    expect(innerDiv).toHaveStyle(`width: ${innerSize}px`);
    const svg = screen.getByTestId("checkbox-svg");
    expect(svg).toHaveAttribute("width", checkSize.toString());
  });

  it("stops click event propagation", () => {
    const onToggleMock = jest.fn();
    const parentClick = jest.fn();
    render(
      <div onClick={parentClick}>
        <Checkbox checked={true} inputHeight={20} onToggle={onToggleMock} />
      </div>
    );
    const wrapper = screen.getByTestId("checkbox-wrapper");
    fireEvent.click(wrapper);
    expect(parentClick).not.toHaveBeenCalled();
    expect(onToggleMock).toHaveBeenCalledTimes(1);
  });
});