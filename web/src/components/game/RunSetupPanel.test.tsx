import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { useState } from "react";

import type { LessonId, LessonMode } from "../../engine/protocol";
import {
  draftFromConfiguration,
  parameterDefinitions,
  type RunConfiguration,
  type RunDraftConfiguration,
} from "../../state/runConfiguration";
import { RunSetupPanel } from "./RunSetupPanel";

const activeConfiguration: RunConfiguration = {
  lessonId: "epsilon-greedy",
  mode: "guided",
  parameter: 0.2,
  seed: 42,
};

function PanelHarness() {
  const [draft, setDraft] = useState<RunDraftConfiguration>(() =>
    draftFromConfiguration(activeConfiguration),
  );

  const changeAlgorithm = (lessonId: LessonId) => {
    setDraft((current) => ({
      ...current,
      lessonId,
      parameter: String(parameterDefinitions[lessonId].defaultValue),
    }));
  };

  const changeMode = (mode: LessonMode) => {
    setDraft((current) => ({ ...current, mode }));
  };

  return (
    <RunSetupPanel
      activeConfiguration={activeConfiguration}
      draftConfiguration={draft}
      challengeTarget="Collect 12 relics while keeping expected regret at or below 3.25."
      pending={false}
      onAlgorithmChange={changeAlgorithm}
      onModeChange={changeMode}
      onParameterChange={(parameter) => setDraft((current) => ({ ...current, parameter }))}
      onSeedChange={(seed) => setDraft((current) => ({ ...current, seed }))}
      onApply={() => undefined}
    />
  );
}

describe("RunSetupPanel", () => {
  it("keeps the slider and number field synchronized", async () => {
    const user = userEvent.setup();
    render(<PanelHarness />);

    const slider = screen.getByRole("slider", { name: "Exploration chance slider" });
    const number = screen.getByLabelText("Exploration chance");
    slider.focus();
    await user.keyboard("{ArrowRight}");

    expect(number).toHaveValue(0.21);
  });

  it("allows keyboard selection of the algorithm and mode", async () => {
    const user = userEvent.setup();
    render(<PanelHarness />);

    const linucb = screen.getByRole("radio", { name: "LinUCB" });
    linucb.focus();
    await user.keyboard(" ");
    expect(linucb).toHaveAttribute("aria-checked", "true");
    expect(screen.getByLabelText("Confidence width")).toHaveValue(1);

    await user.click(screen.getByRole("radio", { name: "Challenge" }));
    expect(screen.getByRole("radio", { name: "Challenge" })).toHaveAttribute(
      "aria-checked",
      "true",
    );
    expect(
      screen.getByText("Collect 12 relics while keeping expected regret at or below 3.25."),
    ).toBeVisible();
  });

  it("shows errors for invalid parameter and seed values", async () => {
    const user = userEvent.setup();
    render(<PanelHarness />);

    const parameter = screen.getByLabelText("Exploration chance");
    await user.clear(parameter);
    expect(screen.getByText("Enter a value from 0 to 1 in steps of 0.01.")).toBeVisible();

    await user.click(screen.getByRole("radio", { name: "Free play" }));
    const seed = screen.getByLabelText("Random seed");
    await user.clear(seed);
    await user.type(seed, "1.5");
    expect(screen.getByText("Enter a safe whole number.")).toBeVisible();
    expect(screen.getByRole("button", { name: "Restart with these settings" })).toBeDisabled();
  });

  it("summarizes the active run and marks draft changes", async () => {
    const user = userEvent.setup();
    render(<PanelHarness />);

    expect(screen.getByText("ε-greedy · Guided · ε 0.2 · seed 42")).toBeVisible();
    expect(screen.getByText("These settings match the current run.")).toBeVisible();

    await user.click(screen.getByRole("radio", { name: "Challenge" }));
    expect(screen.getByText("Changes have not been applied.")).toBeVisible();
    expect(screen.getByText("ε-greedy · Guided · ε 0.2 · seed 42")).toBeVisible();
  });
});
