import { render, screen } from "@testing-library/react";
import { vi } from "vitest";

import {
  cloneRecommendationCandidates,
  defaultRecommendationFeatureIds,
} from "@/state/scenarioCandidates";
import { ScenarioSetupPanel, type ScenarioConfiguration } from "./ScenarioSetupPanel";

function configuration(mode: ScenarioConfiguration["mode"]): ScenarioConfiguration {
  return {
    scenarioId: "recommendations",
    mode,
    seed: 2401,
    parameters: { epsilon: 0.08, learning_rate: 0.18, l2: 0.01 },
    environment: null,
    candidates: cloneRecommendationCandidates(),
    featureIds: [...defaultRecommendationFeatureIds],
    nextCandidateOrdinal: 4,
  };
}

function renderPanel(mode: ScenarioConfiguration["mode"]) {
  return render(
    <ScenarioSetupPanel
      configuration={configuration(mode)}
      pending={false}
      expanded
      onChange={vi.fn()}
      onScenarioChange={vi.fn()}
      onExpandedChange={vi.fn()}
      onApply={vi.fn()}
    />,
  );
}

describe("ScenarioSetupPanel", () => {
  it("presents the current settings as a terminal status line", () => {
    renderPanel("freePlay");

    const status = screen.getByRole("group", {
      name: /Current settings: Mode Free play, 3 candidates, 3 signals, Exploration rate 0.08/,
    });

    expect(status).toHaveTextContent(">_mode: free_play");
    expect(status).toHaveTextContent("candidates: 3");
    expect(status).toHaveTextContent("signals: 3");
    expect(status).toHaveTextContent("exploration: 0.08");
    expect(status).toHaveTextContent("learning: 0.18");
    expect(status).toHaveTextContent("l2: 0.01");
  });

  it("explains how Free Play applies its settings", () => {
    renderPanel("freePlay");

    expect(
      screen.getByText(/Free play unlocks the random seed and simulation coefficients/),
    ).toBeVisible();
    expect(screen.getByRole("button", { name: "Start free play run" })).toBeVisible();
  });

  it("keeps the standard apply label outside Free Play", () => {
    renderPanel("guided");

    expect(
      screen.queryByText(/Free play unlocks the random seed and simulation coefficients/),
    ).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Start configured run" })).toBeVisible();
  });
});
