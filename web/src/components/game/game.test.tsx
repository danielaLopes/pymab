import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";

import type { LessonSnapshot } from "../../engine/protocol";
import { DecisionHistoryBoard } from "./DecisionHistoryBoard";
import { CampaignMap, InspectPanel, PolicyBars } from ".";

describe("Infinite Crossroads components", () => {
  it("gives every path a persistent non-colour identity", () => {
    render(<DecisionHistoryBoard snapshot={null} />);
    expect(screen.getByRole("columnheader", { name: /Moon Path/ })).toBeVisible();
    expect(screen.getByRole("columnheader", { name: /Sun Path/ })).toBeVisible();
    expect(screen.getByRole("columnheader", { name: /Star Path/ })).toBeVisible();
    expect(screen.getByRole("row", { name: /awaiting the first decision/i })).toBeVisible();
  });

  it("provides one route for every concrete policy", () => {
    render(<CampaignMap />, { wrapper: MemoryRouter });
    expect(screen.getAllByRole("link")).toHaveLength(27);
    expect(screen.getByRole("status")).toHaveTextContent("27 policies");
  });

  it("uses recommendation candidate symbols in the predicted probability state", () => {
    const snapshot = {
      policyId: "logistic-contextual-bandit",
      presentation: {
        arms: [
          { name: "Article", shortName: "Article", symbolKind: "article" },
          { name: "Product", shortName: "Product", symbolKind: "product" },
          { name: "Tutorial", shortName: "Tutorial", symbolKind: "tutorial" },
        ],
      },
      diagnostic: {
        decision: {
          label: "Predicted click probability",
          values: [0.2, 0.5, 0.8],
        },
      },
    } as unknown as LessonSnapshot;

    const { container } = render(<PolicyBars snapshot={snapshot} />);

    expect(container.querySelector('[data-symbol-kind="article"]')).toHaveTextContent("▤");
    expect(container.querySelector('[data-symbol-kind="product"]')).toHaveTextContent("▣");
    expect(container.querySelector('[data-symbol-kind="tutorial"]')).toHaveTextContent("▶");
    expect(container.querySelector('[data-symbol-kind="moon"]')).not.toBeInTheDocument();
  });

  it("explains the recommendation context matrix and its encodings", async () => {
    const user = userEvent.setup();
    const snapshot = {
      policyId: "logistic-contextual-bandit",
      scenarioId: "recommendations",
      family: "contextual",
      horizon: 12,
      packageVersion: "2.0.0",
      seed: 2401,
      sourceCommit: "1234567890",
      parameters: { epsilon: 0.08, learning_rate: 0.18, l2: 0.01 },
      generatedCode: "print('demo')",
      hiddenTruth: null,
      presentation: {
        arms: [
          { name: "Article", shortName: "Article", symbolKind: "article" },
          { name: "Product", shortName: "Product", symbolKind: "product" },
        ],
        contextFeatures: [
          { id: "base", name: "Base", type: "base" },
          { id: "visitor", name: "Visitor type", type: "binary" },
          { id: "engagement", name: "Engagement", type: "numeric" },
        ],
      },
      diagnostic: {
        contextMatrix: [
          [1, 1, 0.4],
          [1, 1, 0.4],
        ],
        thetaBefore: [
          [0, 0, 0],
          [0, 0, 0],
        ],
        decision: { label: "Predicted click probability", values: [0.5, 0.5] },
      },
    } as unknown as LessonSnapshot;

    render(
      <InspectPanel
        snapshot={snapshot}
        open
        onToggle={() => undefined}
        onOpenLab={() => undefined}
      />,
    );

    await user.hover(screen.getByRole("button", { name: "About Current context matrix" }));
    expect(await screen.findByRole("tooltip")).toHaveTextContent(
      "Every candidate is evaluated for the same visitor, so the rows repeat",
    );

    await user.keyboard("{Escape}");
    await user.hover(screen.getByRole("button", { name: "About Visitor type values" }));
    expect(await screen.findByRole("tooltip")).toHaveTextContent(
      "New is stored as -1. Returning is stored as +1.",
    );
  });

  it("does not add recommendation help to other contextual matrices", () => {
    const snapshot = {
      policyId: "linucb",
      scenarioId: null,
      family: "contextual",
      horizon: 12,
      packageVersion: "2.0.0",
      seed: 2401,
      sourceCommit: "1234567890",
      parameters: { alpha: 1 },
      generatedCode: "print('demo')",
      hiddenTruth: null,
      presentation: {
        arms: [
          { name: "Moon", shortName: "Moon", symbolKind: "moon" },
          { name: "Sun", shortName: "Sun", symbolKind: "sun" },
        ],
        contextFeatures: [
          { id: "base", name: "Base", type: "base" },
          { id: "light", name: "Light", type: "binary" },
        ],
      },
      diagnostic: {
        contextMatrix: [
          [1, 1],
          [1, 1],
        ],
        thetaBefore: [
          [0, 0],
          [0, 0],
        ],
        predictedMeans: [0, 0],
        bonuses: [1, 1],
        ucbScores: [1, 1],
      },
    } as unknown as LessonSnapshot;

    render(
      <InspectPanel
        snapshot={snapshot}
        open
        onToggle={() => undefined}
        onOpenLab={() => undefined}
      />,
    );

    expect(
      screen.queryByRole("button", { name: "About Current context matrix" }),
    ).not.toBeInTheDocument();
  });
});
