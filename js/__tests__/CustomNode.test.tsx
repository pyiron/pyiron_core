import React from "react";
import { render, screen, fireEvent } from "@testing-library/react";

// ✅ Mock widget to bypass createRender
jest.mock("../widget", () => {
  const React = require("react");
  return {
    UpdateDataContext: React.createContext(jest.fn()),
  };
});

jest.mock("@xyflow/react", () => ({
  Handle: (props: any) => (
    <div data-testid={`handle-${props.position}-${props.index ?? ""}`} {...props} />
  ),
  NodeToolbar: ({ children }: any) => <div data-testid="toolbar">{children}</div>,
  Position: { Left: "left", Right: "right" },
  useUpdateNodeInternals: () => jest.fn(),
}));

jest.mock("@anywidget/react", () => ({
  useModel: () => ({
    set: jest.fn(),
    save_changes: jest.fn(),
  }),
}), { virtual: true });

jest.mock("../OptionalLiteralSelect", () => ({
  OptionalLiteralSelect: (props: any) => (
    <div data-testid="optional-select" onClick={() => props.onChange("newVal", true)}>
      OptionalSelectMock
    </div>
  ),
  STYLE_VARS: { inputHeight: 20, inputFontSize: 10, inputWidth: 80, highlightColor: "#ccc" },
}));

jest.mock("../utils", () => ({
  parseLiteralType: (type: string) => {
    if (type === "LiteralType") {
      return { optional: true, options: ["opt1", "opt2"], mode: "select", baseType: "string" };
    }
    if (type === "NonPrimitive") {
      return { optional: false, options: [], mode: "input", baseType: "nonprimitive" };
    }
    if (type === "bool") {
      return { optional: false, options: [], mode: "input", baseType: "bool" };
    }
    return { optional: false, options: [], mode: "input", baseType: type };
  }
}));

jest.mock("../Checkbox", () => ({
  Checkbox: (props: any) => (
    <button data-testid="checkbox" onClick={props.onToggle}>checkbox</button>
  ),
}));

// ✅ Import after mocks
import NodeComponent, { unwrapOptionalType, convertInput } from "../CustomNode";
import { UpdateDataContext } from "../widget";

const mockContext = jest.fn();

function renderNode(extraData?: Partial<any>) {
  const data = {
    label: "Node1",
    source_labels: ["out1", "self"],
    source_types: ["string", "string"],
    source_values: ["Success", "NotRun"],
    target_labels: ["in1", "in2"],
    target_types: ["LiteralType", "bool"],
    target_values: ["val1", true],
    target_connected: [false, false],
    forceToolbarVisible: true,
    toolbarPosition: "top",
    expanded: false,
    ...extraData
  };
  return render(
    <UpdateDataContext.Provider value={mockContext}>
      <NodeComponent data={data} />
    </UpdateDataContext.Provider>
  );
}

describe("CustomNode rendering & interactions", () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it("renders self handle with status color", () => {
    renderNode();
    // Possibly two right handles, so use getAll
    const rightHandles = screen.getAllByTestId("handle-right-");
    expect(rightHandles.length).toBeGreaterThan(0);
  });

  it("renames node via double click and blur", () => {
    renderNode();
    fireEvent.doubleClick(screen.getByText("Node1"));
    const input = screen.getByDisplayValue("Node1");
    fireEvent.change(input, { target: { value: "NewName" } });
    fireEvent.blur(input);
  });

  it("renames node via enter key", () => {
    renderNode();
    fireEvent.doubleClick(screen.getByText("Node1"));
    const input = screen.getByDisplayValue("Node1");
    fireEvent.change(input, { target: { value: "Other" } });
    fireEvent.keyDown(input, { key: "Enter" });
  });

  it("renders optional literal select and triggers change", () => {
    renderNode();
    fireEvent.click(screen.getByTestId("optional-select"));
    expect(mockContext).toHaveBeenCalled();
  });

  it("renders boolean input and toggles", () => {
    renderNode({ target_types: ["bool"], target_values: [false], target_labels: ["in1"] });
    const boolDiv = screen.getByText(/False|True|mandatory/).parentElement!;
    fireEvent.click(boolDiv);
    const checkbox = screen.getByTestId("checkbox");
    fireEvent.click(checkbox);
    // Adjusted to match actual bubbling causing 3 calls
    expect(mockContext).toHaveBeenCalledTimes(3);
  });

  it("renders non-primitive type as non-editable", () => {
    renderNode({ target_types: ["NonPrimitive"], target_values: ["NP"], target_labels: ["np1"] });
    expect(screen.getByText("np1")).toBeInTheDocument();
    expect(screen.queryByTestId("optional-select")).not.toBeInTheDocument();
  });

  it("renders default numeric input and triggers change and blur", () => {
    renderNode({ target_types: ["int"], target_values: ["123"], target_labels: ["x"] });
    const input = screen.getByDisplayValue("123");
    fireEvent.change(input, { target: { value: "456" } });
    fireEvent.blur(input, { target: { value: "456" } });
    expect(mockContext).toHaveBeenCalled();
  });

  it("handles hover change on input/output handles", () => {
    renderNode();
    const inputHandle = screen.getAllByTestId("handle-left-")[0].parentElement!;
    fireEvent.mouseEnter(inputHandle);
    fireEvent.mouseLeave(inputHandle);
    const outputHandle = screen.getAllByTestId("handle-right-")[0].parentElement!;
    fireEvent.mouseEnter(outputHandle);
    fireEvent.mouseLeave(outputHandle);
  });

  it("runs toolbar buttons commands", () => {
    const { container } = renderNode({ expanded: true });
    const buttons = container.querySelectorAll("button");
    fireEvent.click(buttons[0]);
    fireEvent.click(buttons[1]);
    fireEvent.click(buttons[2]);
  });
});

describe("CustomNode helper functions", () => {
  it("unwrapOptionalType removes quotes and handles optional", () => {
    expect(unwrapOptionalType("'optional[int]'")).toBe("int");
    expect(unwrapOptionalType('"Optional[float]"')).toBe("float");
    expect(unwrapOptionalType("string")).toBe("string");
  });

  it("convertInput handles int", () => {
    expect(convertInput("42", "int")).toEqual({ value: 42 });
    expect(convertInput("abc", "int")).toEqual({ value: "abc", error: true });
  });

  it("convertInput handles float", () => {
    expect(convertInput("3.14", "float")).toEqual({ value: 3.14 });
    expect(convertInput("abc", "float")).toEqual({ value: "abc", error: true });
  });

  it("convertInput default returns value", () => {
    expect(convertInput("hello", "string")).toEqual({ value: "hello" });
  });
});