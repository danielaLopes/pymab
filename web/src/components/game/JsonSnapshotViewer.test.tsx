import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { JsonSnapshotViewer } from "./JsonSnapshotViewer";
import { formatSnapshotJson } from "./jsonSnapshot";

const snapshot = {
  contextMatrix: [
    [1, -1, 0.19999999999999996, -1],
    [1, -1, 0.19999999999999996, -1],
  ],
  decision: {
    label: "Predicted click probability",
    selectionBranch: "exploit",
    values: [0.5, 0.5, 0.5],
  },
  outcomeLabel: "Click",
};

describe("formatSnapshotJson", () => {
  it("keeps primitive arrays compact without changing their values", () => {
    const formatted = formatSnapshotJson(snapshot);

    expect(formatted).toContain("[1, -1, 0.19999999999999996, -1]");
    expect(formatted).toContain('"values": [0.5, 0.5, 0.5]');
    expect(JSON.parse(formatted)).toEqual(snapshot);
  });
});

describe("JsonSnapshotViewer", () => {
  const writeText = vi.fn<() => Promise<void>>();

  beforeEach(() => {
    writeText.mockReset();
    writeText.mockResolvedValue();
    Object.defineProperty(navigator, "clipboard", {
      configurable: true,
      value: { writeText },
    });
  });

  it("renders an accessible, line-numbered JSON code region", () => {
    render(<JsonSnapshotViewer value={snapshot} />);

    fireEvent.click(screen.getByText("Full validated snapshot"));

    expect(screen.getByRole("region", { name: "Full validated snapshot JSON" })).toBeVisible();
    expect(screen.getAllByRole("listitem")).toHaveLength(
      formatSnapshotJson(snapshot).split("\n").length,
    );
  });

  it("copies the complete formatted JSON", async () => {
    render(<JsonSnapshotViewer value={snapshot} />);

    fireEvent.click(screen.getByText("Full validated snapshot"));
    fireEvent.click(screen.getByRole("button", { name: "Copy JSON" }));

    await waitFor(() => expect(writeText).toHaveBeenCalledWith(formatSnapshotJson(snapshot)));
    expect(await screen.findByText("JSON copied.")).toBeVisible();
  });
});
