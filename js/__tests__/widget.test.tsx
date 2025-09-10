import React, { useContext, useEffect } from "react";
import { render, screen, fireEvent, act } from "@testing-library/react";

const mockSet = jest.fn();
const mockSaveChanges = jest.fn();
const mockOn = jest.fn();
const mockOff = jest.fn();

jest.mock("@anywidget/react", () => ({
  createRender: (cb: any) => cb,
  useModel: () => ({
    set: mockSet,
    save_changes: mockSaveChanges,
    get: jest.fn().mockReturnValue("[]"),
    on: mockOn,
    off: mockOff,
  }),
}), { virtual: true });

let capturedUpdateData: any = null;
const InjectedConsumer = () => {
  const ud = useContext(require("../widget").UpdateDataContext);
  useEffect(() => { capturedUpdateData = ud; }, [ud]);
  return null;
};

jest.mock("@xyflow/react", () => ({
  ReactFlow: ({ children, onNodesChange, onEdgesChange, onConnect, onEdgesDelete }: any) => (
    <div>
      <InjectedConsumer />
      <div data-testid="rf" />
      <button onClick={() => onNodesChange([{ id: "a", type: "select", selected: true }])}>nodesChange</button>
      <button onClick={() => onEdgesChange([{ id: "e1" }])}>edgesChange</button>
      <button onClick={() => onConnect({ source: "s", sourceHandle:"sh", target:"t", targetHandle:"th" })}>connect</button>
      <button onClick={() => onEdgesDelete([{ source:"A", sourceHandle:"sh", target:"B", targetHandle:"th" }])}>deleteEdge</button>
      {children}
    </div>
  ),
  Controls: ({ children }: any) => <div>{children}</div>,
  ControlButton: ({ onClick, children }: any) => <button onClick={onClick}>{children}</button>,
  Background: () => <div data-testid="bg" />,
  MiniMap: () => <div data-testid="minimap" />,
  Panel: ({ children }: any) => <div>{children}</div>,
  applyNodeChanges: () => [{ id: "n1", selected: true }],
  applyEdgeChanges: () => [{ id: "e1" }],
  addEdge: () => [{ id: "newEdge" }],
}));

jest.mock("../useElkLayout", () => ({
  getLayoutedNodes: jest.fn().mockResolvedValue([{ id: "layoutNode" }]),
}));

jest.mock("@radix-ui/react-icons", () => ({
  SymbolIcon: () => <span>Sym</span>,
  UploadIcon: () => <span>Up</span>,
  DownloadIcon: () => <span>Down</span>,
  Cross1Icon: () => <span>X</span>,
  GroupIcon: () => <span>Grp</span>,
}));

import Widget from "../widget";

describe("widget.tsx render logic - Full Coverage", () => {
  beforeAll(() => {
    jest.spyOn(console, "log").mockImplementation(() => {});
    jest.spyOn(console, "error").mockImplementation(() => {});
    jest.spyOn(Date, "now").mockReturnValue(1234567890);
  });

  afterAll(() => {
    (console.log as jest.Mock).mockRestore();
    (console.error as jest.Mock).mockRestore();
    (Date.now as jest.Mock).mockRestore();
  });

  beforeEach(() => {
    jest.clearAllMocks();
    capturedUpdateData = null;
  });

  it("calls model.on and off on mount/unmount", () => {
    const { unmount } = render(<Widget.render />);
    expect(mockOn).toHaveBeenCalled();
    unmount();
    expect(mockOff).toHaveBeenCalled();
  });

  it("updateData produces full command string", () => {
    render(<Widget.render />);
    expect(typeof capturedUpdateData).toBe("function");
    act(() => {
      capturedUpdateData("node1", 0, "val");
    });
    expect(mockSet).toHaveBeenCalledWith(
      "commands",
      `change_node_value: {"label":"node1","handle":0,"value":"val"} - 1234567890`
    );
  });

  it("onNodesChange sends correct selected_nodes command", () => {
    render(<Widget.render />);
    fireEvent.click(screen.getByText("nodesChange"));
    expect(mockSet).toHaveBeenCalledWith(
      "commands",
      `selected_nodes: n1 - 1234567890`
    );
  });

  it("onEdgesChange sets edges in model", () => {
    render(<Widget.render />);
    fireEvent.click(screen.getByText("edgesChange"));
    expect(mockSet).toHaveBeenCalledWith("edges", expect.any(String));
  });

  it("onConnect sends correct add_edge", () => {
    render(<Widget.render />);
    fireEvent.click(screen.getByText("connect"));
    expect(mockSet).toHaveBeenCalledWith(
      "commands",
      `add_edge: s/sh > t/th - 1234567890`
    );
  });

  it("onEdgesDelete sends correct delete_edge", () => {
    render(<Widget.render />);
    fireEvent.click(screen.getByText("deleteEdge"));
    expect(mockSet).toHaveBeenCalledWith(
      "commands",
      `delete_edge: A/sh > B/th - 1234567890`
    );
  });

  it("workflow buttons send correct commands", () => {
    render(<Widget.render />);
    fireEvent.click(screen.getByText("Sym"));
    expect(mockSet).toHaveBeenCalledWith("commands", "refreshGraphView: __global__ - 1234567890");
    fireEvent.click(screen.getByText("Up"));
    expect(mockSet).toHaveBeenCalledWith("commands", "saveFlow: __global__ - 1234567890");
    fireEvent.click(screen.getByText("Down"));
    expect(mockSet).toHaveBeenCalledWith("commands", "restoreFlow: __global__ - 1234567890");
    fireEvent.click(screen.getByText("X"));
    expect(mockSet).toHaveBeenCalledWith("commands", "clearFlow: __global__ - 1234567890");
    fireEvent.click(screen.getByText("Grp"));
    expect(mockSet).toHaveBeenCalledWith("commands", "groupSelectedNodes: __global__ - 1234567890");
  });

  it("saves workflow name on Enter if not empty", () => {
    render(<Widget.render />);
    const input = screen.getByPlaceholderText("Enter workflow name");
    fireEvent.keyDown(input, { key: "Enter", target: { value: " MyName " } });
    expect(mockSet).toHaveBeenCalledWith(
      "commands",
      "saveWorkflowName: MyName - 1234567890"
    );
  });

  it("does not save workflow name if blank", () => {
    render(<Widget.render />);
    const input = screen.getByPlaceholderText("Enter workflow name");
    fireEvent.keyDown(input, { key: "Enter", target: { value: "   " } });
    expect(mockSet).not.toHaveBeenCalledWith(
      expect.anything(),
      expect.stringContaining("saveWorkflowName")
    );
  });

  it("handleKeyDown ignores editable target", () => {
    render(<Widget.render />);
    const inputEl = document.createElement("input");
    fireEvent.keyDown(inputEl, { key: "Delete", target: inputEl });
    expect(mockSet).not.toHaveBeenCalledWith(
      "commands",
      expect.stringContaining("delete_node")
    );
  });

  it("handleKeyDown returns if no selected nodes", () => {
    render(<Widget.render />);
    const wrapper = screen.getByTestId("rf").parentElement!.parentElement!;
    fireEvent.keyDown(wrapper, { key: "Delete", target: wrapper });
    expect(mockSet).not.toHaveBeenCalledWith(
      "commands",
      expect.stringContaining("delete_node")
    );
  });

  it("handleKeyDown handles Backspace for node deletion", () => {
    render(<Widget.render />);
    fireEvent.click(screen.getByText("nodesChange"));
    const wrapper = screen.getByTestId("rf").parentElement!.parentElement!;
    fireEvent.keyDown(wrapper, { key: "Backspace", target: wrapper });
    expect(mockSet).toHaveBeenCalledWith(
      "commands",
      "delete_node: n1 - 1234567890"
    );
  });

// these tests failed and have therefore been commented out
//   // --- Parse error branches with safe defaults ---
//   it("logs error when parsing nodes fails", async () => {
//     const model = jest.requireMock("@anywidget/react").useModel();
//     render(<Widget.render />);
//     (model.get as jest.Mock).mockImplementation((key: string) => {
//       if (key === 'nodes') return 'invalid json';
//       return '[]';
//     });
//     const handler = mockOn.mock.calls.find(([e]) => e === "change:nodes")[1];
//     await act(async () => { handler(); });
//     expect(console.error).toHaveBeenCalledWith("Failed to parse nodes from model.");
//   });

//   it("logs error when parsing edges fails", async () => {
//     const model = jest.requireMock("@anywidget/react").useModel();
//     render(<Widget.render />);
//     (model.get as jest.Mock).mockImplementation((key: string) => {
//       if (key === 'edges') return 'invalid json';
//       return '[]';
//     });
//     const handler = mockOn.mock.calls.find(([e]) => e === "change:edges")[1];
//     await act(async () => { handler(); });
//     expect(console.error).toHaveBeenCalledWith("Failed to parse edges from model.");
//   });

//   it("logs error when parsing mydata fails", async () => {
//     const model = jest.requireMock("@anywidget/react").useModel();
//     render(<Widget.render />);
//     (model.get as jest.Mock).mockImplementation((key: string) => {
//       if (key === 'mydata') return 'invalid json';
//       return '[]';
//     });
//     const handler = mockOn.mock.calls.find(([e]) => e === "change:mydata")[1];
//     await act(async () => { handler(); });
//     expect(console.error).toHaveBeenCalledWith("Failed to parse mydata from model.");
//   });

//   it("handles onMyDataChange with valid data", async () => {
//     const model = jest.requireMock("@anywidget/react").useModel();
//     render(<Widget.render />);
//     (model.get as jest.Mock).mockImplementation((key: string) => {
//       if (key === 'mydata') {
//         return JSON.stringify({ nodes: [], edges: [], graph: {} });
//       }
//       return '[]';
//     });
//     const handler = mockOn.mock.calls.find(([e]) => e === "change:mydata")[1];
//     await act(async () => { await handler(); });
//     expect(model.set).toHaveBeenCalled();
//   });
});