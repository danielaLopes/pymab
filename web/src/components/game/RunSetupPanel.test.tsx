import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { useState } from "react";

import { policyCatalog, type PolicyId } from "@/catalog/policies";
import type { LessonMode } from "@/engine/protocol";
import { generateEnvironment, type EnvironmentDraft } from "@/state/environments";
import {
  defaultConfiguration,
  draftFromConfiguration,
  type RunDraftConfiguration,
} from "@/state/runConfiguration";
import { RunSetupPanel } from "./RunSetupPanel";

const activeConfiguration = defaultConfiguration("epsilon-greedy");

function PanelHarness() {
  const [draft, setDraft] = useState<RunDraftConfiguration>(() =>
    draftFromConfiguration(activeConfiguration),
  );
  const changePolicy = (policyId: PolicyId) => {
    const definition = policyCatalog[policyId];
    setDraft((current) =>
      draftFromConfiguration({
        ...defaultConfiguration(policyId, current.mode),
        seed: Number(current.seed),
        parameters: definition.defaults,
      }),
    );
  };
  const changeMode = (mode: LessonMode) => setDraft((current) => ({ ...current, mode }));
  const changeEnvironment = (environment: EnvironmentDraft) =>
    setDraft((current) => ({ ...current, environment, probabilitySource: "custom" }));

  return (
    <RunSetupPanel
      activeConfiguration={activeConfiguration}
      draftConfiguration={draft}
      challengeTarget="Keep expected regret within the target."
      pending={false}
      onPolicyChange={changePolicy}
      onModeChange={changeMode}
      onParameterChange={(key, value) =>
        setDraft((current) => ({ ...current, parameters: { ...current.parameters, [key]: value } }))
      }
      onSeedChange={(seed) => setDraft((current) => ({ ...current, seed }))}
      onEnvironmentChange={changeEnvironment}
      onRegenerateEnvironment={() =>
        setDraft((current) => ({
          ...current,
          environment: generateEnvironment(current.policyId, Number(current.seed)),
          probabilitySource: "generated",
        }))
      }
      onRestorePolicyDefaults={() => undefined}
      onApply={() => undefined}
    />
  );
}

describe("RunSetupPanel", () => {
  it("keeps a parameter slider and number field synchronized", async () => {
    const user = userEvent.setup();
    render(<PanelHarness />);
    const slider = screen.getByRole("slider", { name: "Exploration chance slider" });
    slider.focus();
    await user.keyboard("{ArrowRight}");
    expect(screen.getByLabelText("Exploration chance")).toHaveValue(0.21);
  });

  it("changes policy and mode with accessible controls", async () => {
    const user = userEvent.setup();
    render(<PanelHarness />);
    await user.click(screen.getByRole("combobox", { name: "Policy" }));
    await user.click(screen.getByRole("option", { name: "LinUCB" }));
    expect(screen.getByLabelText("Confidence width")).toHaveValue(1);
    await user.click(screen.getByRole("radio", { name: "Challenge" }));
    expect(screen.getByText("Keep expected regret within the target.")).toBeVisible();
  });

  it("renders family-specific free-play environments", async () => {
    const user = userEvent.setup();
    render(<PanelHarness />);
    await user.click(screen.getByRole("radio", { name: "Free play" }));
    expect(screen.getByText("Portal reward chances")).toBeVisible();
    const moon = screen.getByLabelText("Moon");
    await user.clear(moon);
    await user.type(moon, "31.7");
    expect(screen.getByText("Custom")).toBeVisible();
    await user.click(screen.getByRole("combobox", { name: "Policy" }));
    await user.click(screen.getByRole("option", { name: "Gaussian Thompson sampling" }));
    expect(screen.getByText("Numeric reward model")).toBeVisible();
    expect(screen.getByLabelText("Shared standard deviation")).toBeVisible();
  });
});
