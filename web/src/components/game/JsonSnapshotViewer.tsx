import { useMemo, useState, type ReactNode } from "react";

import { Button } from "@/components/ui/button";
import { formatSnapshotJson } from "./jsonSnapshot";

function renderJsonLine(line: string, lineIndex: number): ReactNode[] {
  const tokenPattern =
    /("(?:\\.|[^"\\])*")(?=\s*:)|("(?:\\.|[^"\\])*")|(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)|\b(true|false|null)\b/g;
  const nodes: ReactNode[] = [];
  let cursor = 0;

  for (const match of line.matchAll(tokenPattern)) {
    const start = match.index;
    if (start > cursor) nodes.push(line.slice(cursor, start));

    const className = match[1]
      ? "snapshot-token-key"
      : match[2]
        ? "snapshot-token-string"
        : match[3]
          ? "snapshot-token-number"
          : "snapshot-token-literal";
    nodes.push(
      <span className={className} key={`${lineIndex}-${start}`}>
        {match[0]}
      </span>,
    );
    cursor = start + match[0].length;
  }

  if (cursor < line.length) nodes.push(line.slice(cursor));
  return nodes;
}

export function JsonSnapshotViewer({
  value,
  label = "Full validated snapshot",
}: {
  value: unknown;
  label?: string;
}) {
  const [copyStatus, setCopyStatus] = useState("");
  const json = useMemo(() => formatSnapshotJson(value), [value]);
  const lines = json.split("\n");
  const copyJson = async () => {
    try {
      await navigator.clipboard.writeText(json);
      setCopyStatus("JSON copied.");
    } catch {
      setCopyStatus("Copy failed. Select the JSON manually.");
    }
  };

  return (
    <details className="validated-snapshot-details">
      <summary className="validated-snapshot-summary">{label}</summary>
      <div className="snapshot-code-viewer">
        <div className="snapshot-code-toolbar">
          <span aria-hidden="true">JSON</span>
          <Button
            className="snapshot-copy-button"
            type="button"
            variant="ghost"
            size="sm"
            onClick={() => void copyJson()}
          >
            Copy JSON
          </Button>
        </div>
        <div
          className="snapshot-code-scroll"
          role="region"
          tabIndex={0}
          aria-label={`${label} JSON`}
        >
          <ol className="snapshot-code-lines">
            {lines.map((line, index) => (
              <li className="snapshot-code-line" key={index}>
                <code>{renderJsonLine(line, index)}</code>
              </li>
            ))}
          </ol>
        </div>
        <p className="snapshot-copy-status" aria-live="polite">
          {copyStatus}
        </p>
      </div>
    </details>
  );
}
