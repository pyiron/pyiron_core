// utils.ts
export function parseLiteralType(typeString: string): {
  optional: boolean;
  options: string[];
  mode?: "select" | "input";
  baseType?: string;
} {
  const trimmed = typeString.trim();

  // Remove all whitespace for detection
  const compact = trimmed.replace(/\s+/g, "");

  // Optional[Literal[…]]
  if (compact.startsWith("Optional[Literal[")) {
    const inside = compact.substring("Optional[Literal[".length);
    const closeIndex = inside.indexOf("]]");
    if (closeIndex >= 0) {
      const options = inside.substring(0, closeIndex)
        .split(",")
        .map(o => o.trim())
        .filter(o => o.length > 0);
      return {
        optional: true,
        options,
        mode: "select",
      };
    }
    return { optional: true, options: [], mode: "select" };
  }

  // Literal[…]
  if (compact.startsWith("Literal[")) {
    const inside = compact.substring("Literal[".length);
    const closeIndex = inside.indexOf("]");
    if (closeIndex >= 0) {
      const options = inside.substring(0, closeIndex)
        .split(",")
        .map(o => o.trim())
        .filter(o => o.length > 0);
      return {
        optional: false,
        options,
        mode: "select",
      };
    }
    return { optional: false, options: [], mode: "select" };
  }

  // Optional[…primitive…]
  if (compact.startsWith("Optional[") && compact.endsWith("]")) {
    const inner = compact.slice(9, -1).trim().toLowerCase();
    return {
      optional: true,
      options: [],
      mode: "input",
      baseType: inner,
    };
  }

  // Non-optional primitive
  return {
    optional: false,
    options: [],
    mode: "input",
    baseType: compact.toLowerCase(),
  };
}