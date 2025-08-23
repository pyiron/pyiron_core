import { parseLiteralType } from "../utils";

describe("parseLiteralType", () => {
  test("plain Literal with several items", () => {
    const result = parseLiteralType("Literal[fcc, bcc, hcp]");
    expect(result).toEqual({
      optional: false,
      options: ["fcc", "bcc", "hcp"],
    });
  });

  test("Optional[Literal] with irregular whitespace", () => {
    const result = parseLiteralType("  Optional[ Literal[ a , b ] ] ");
    expect(result).toEqual({
      optional: true,
      options: ["a", "b"],
    });
  });

  test("Literal with a single item", () => {
    const result = parseLiteralType("Literal[one]");
    expect(result).toEqual({
      optional: false,
      options: ["one"],
    });
  });

  test("non‑Literal string returns empty options", () => {
    const result = parseLiteralType("Foo[bar]");
    expect(result).toEqual({
      optional: false,
      options: [],
    });
  });

  test("malformed Literal (missing closing bracket) → empty options", () => {
    const result = parseLiteralType("Literal[fcc, bcc");
    expect(result).toEqual({
      optional: false,
      options: [],
    });
  });

  test("empty Literal list → empty options", () => {
    const result = parseLiteralType("Literal[   ]");
    expect(result).toEqual({
      optional: false,
      options: [],
    });
  });

  test("Optional[Literal] with extra trailing content (ignored)", () => {
    const result = parseLiteralType("Optional[Literal[a, b], int]");
    expect(result).toEqual({
      optional: true,
      options: ["a", "b"],
    });
  });
});