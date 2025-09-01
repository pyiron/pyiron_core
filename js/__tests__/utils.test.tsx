import { parseLiteralType } from "../utils";

describe("parseLiteralType", () => {
  test("plain Literal with several items", () => {
    expect(parseLiteralType("Literal[fcc, bcc, hcp]")).toEqual({
      optional: false,
      options: ["fcc", "bcc", "hcp"],
      mode: "select"
    });
  });

  test("Optional[Literal] with irregular whitespace", () => {
    expect(parseLiteralType("  Optional[ Literal[ a , b ] ] ")).toEqual({
      optional: true,
      options: ["a", "b"],
      mode: "select"
    });
  });

  test("Literal with a single item", () => {
    expect(parseLiteralType("Literal[one]")).toEqual({
      optional: false,
      options: ["one"],
      mode: "select"
    });
  });

  test("non‑Literal string returns empty options", () => {
    expect(parseLiteralType("Foo[bar]")).toEqual({
      optional: false,
      options: [],
      mode: "input",
      baseType: "foo[bar]"
    });
  });

  test("malformed Literal (missing closing bracket) → empty options", () => {
    expect(parseLiteralType("Literal[fcc, bcc")).toEqual({
      optional: false,
      options: [],
      mode: "select"
    });
  });

  test("empty Literal list → empty options", () => {
    expect(parseLiteralType("Literal[   ]")).toEqual({
      optional: false,
      options: [],
      mode: "select"
    });
  });

  // Numeric optionals
  test("Optional[float] is optional input mode number", () => {
    expect(parseLiteralType("Optional[float]")).toEqual({
      optional: true,
      options: [],
      mode: "input",
      baseType: "float"
    });
  });

  test("Optional[int] is optional input mode number", () => {
    expect(parseLiteralType("Optional[int]")).toEqual({
      optional: true,
      options: [],
      mode: "input",
      baseType: "int"
    });
  });

  // Numeric non-optionals
  test("float is non-optional input mode number", () => {
    expect(parseLiteralType("float")).toEqual({
      optional: false,
      options: [],
      mode: "input",
      baseType: "float"
    });
  });

  test("int is non-optional input mode number", () => {
    expect(parseLiteralType("int")).toEqual({
      optional: false,
      options: [],
      mode: "input",
      baseType: "int"
    });
  });

  // Boolean non-optionals
  test("bool is non-optional input mode", () => {
    expect(parseLiteralType("bool")).toEqual({
      optional: false,
      options: [],
      mode: "input",
      baseType: "bool"
    });
  });

  // Boolean optionals
  test("Optional[bool] is optional input mode", () => {
    expect(parseLiteralType("Optional[bool]")).toEqual({
      optional: true,
      options: [],
      mode: "input",
      baseType: "bool"
    });
  });
});