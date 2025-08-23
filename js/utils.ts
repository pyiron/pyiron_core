// utils.ts
export function parseLiteralType(typeString: string): {
  optional: boolean;
  options: string[];
  mode?: "select" | "input";
  baseType?: string;
} {
  const trimmed = typeString.trim();

  // Optional[Literal[…]]
  if (trimmed.startsWith("Optional[Literal[")) {
    const optionsString = trimmed.split("Optional[Literal[")[1].split("]")[0];
    const options = optionsString.split(",").map(o => o.trim());
    return {
      optional: true,
      options,
      mode: "select",
    };
  }

  // Literal[…]
  if (trimmed.startsWith("Literal[")) {
    const optionsString = trimmed.split("Literal[")[1].split("]")[0];
    const options = optionsString.split(",").map(o => o.trim());
    return {
      optional: false,
      options,
      mode: "select",
    };
  }

  // Optional[…primitive…]
  if (trimmed.startsWith("Optional[") && trimmed.endsWith("]")) {
    const inner = trimmed.slice(9, -1).trim().toLowerCase();
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
    baseType: trimmed.toLowerCase(),
  };
}