type JsonPrimitive = boolean | null | number | string;
type JsonValue = JsonPrimitive | JsonValue[] | { [key: string]: JsonValue };

const INDENT = "  ";
const INLINE_ARRAY_LIMIT = 100;

function isPrimitive(value: JsonValue): value is JsonPrimitive {
  return value === null || typeof value !== "object";
}

function formatValue(value: JsonValue, depth: number): string {
  if (isPrimitive(value)) return JSON.stringify(value);

  if (Array.isArray(value)) {
    if (value.length === 0) return "[]";

    if (value.every(isPrimitive)) {
      const inline = `[${value.map((item) => JSON.stringify(item)).join(", ")}]`;
      if (inline.length + depth * INDENT.length <= INLINE_ARRAY_LIMIT) return inline;
    }

    return `[
${value.map((item) => `${INDENT.repeat(depth + 1)}${formatValue(item, depth + 1)}`).join(",\n")}
${INDENT.repeat(depth)}]`;
  }

  const entries = Object.entries(value);
  if (entries.length === 0) return "{}";

  return `{
${entries
  .map(
    ([key, item]) =>
      `${INDENT.repeat(depth + 1)}${JSON.stringify(key)}: ${formatValue(item, depth + 1)}`,
  )
  .join(",\n")}
${INDENT.repeat(depth)}}`;
}

export function formatSnapshotJson(value: unknown): string {
  const serialized = JSON.stringify(value);
  const normalized = serialized === undefined ? null : (JSON.parse(serialized) as JsonValue);
  return formatValue(normalized, 0);
}
