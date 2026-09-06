import type { LessonSnapshot } from "../../engine/protocol";
import { buildDecisionHistory, publicPathDetail } from "./decisionHistory";

const baseSnapshot: LessonSnapshot = {
  policyId: "epsilon-greedy",
  scenarioId: null,
  family: "foundations",
  objective: "cumulative-reward",
  mode: "guided",
  seed: 42,
  packageVersion: "2.0.0",
  sourceCommit: "test",
  sessionId: "history-test",
  step: 0,
  horizon: 20,
  parameters: { epsilon: 0.2, initial_value: 0 },
  environment: null,
  gateIds: ["moon", "sun", "star"],
  presentation: {
    experienceKind: "policy",
    experienceId: "epsilon-greedy",
    arms: [
      { name: "Moon Path", shortName: "Moon", symbolKind: "moon" },
      { name: "Sun Path", shortName: "Sun", symbolKind: "sun" },
      { name: "Star Path", shortName: "Star", symbolKind: "star" },
    ],
    rewardPresentation: "binary",
    positiveOutcomeLabel: "Relic found",
    zeroOutcomeLabel: "No relic",
  },
  selectedArm: null,
  reward: null,
  totalReward: 0,
  instantaneousExpectedRegret: null,
  cumulativeExpectedRegret: 0,
  completed: false,
  passed: false,
  visibleCues: [],
  publicContext: null,
  explanationKey: "ready",
  diagnostic: null,
  recommendation: null,
  history: [],
  hiddenTruth: null,
  generatedCode: "",
};

describe("decision history presentation", () => {
  it("preserves contextual cues and shows only the selected reward", () => {
    const snapshot: LessonSnapshot = {
      ...baseSnapshot,
      policyId: "logistic-contextual-bandit",
      family: "contextual",
      step: 1,
      selectedArm: 1,
      reward: 1,
      visibleCues: [
        { name: "light", value: -1, label: "red light" },
        { name: "echo", value: -1, label: "low echo" },
        { name: "tide", value: 1, label: "high tide" },
      ],
      history: [
        {
          selectedArm: 1,
          reward: 1,
          instantaneousExpectedRegret: 0.1,
          visibleCues: [
            { name: "light", value: -1, label: "red light" },
            { name: "echo", value: -1, label: "low echo" },
            { name: "tide", value: 1, label: "high tide" },
          ],
          publicContext: [
            [1, -1, -1, 1],
            [1, -1, -1, 1],
            [1, -1, -1, 1],
          ],
          explanationKey: "repeat",
          diagnostic: {
            before: { theta: [[0, 0, 0, 0]] },
            after: { theta: [[0, 0, 0, 0]] },
            decision: {
              label: "Predicted reward chance",
              values: [0.28, 0.71, 0.46],
              selectionBranch: "exploit",
            },
          },
        },
      ],
    };

    const [row] = buildDecisionHistory(snapshot);
    expect(row?.cues.map((cue) => cue.value)).toEqual(["red", "low", "high"]);
    expect(row?.reason).toBe("Exploit");
    expect(row?.cells.map((cell) => cell.primary)).toEqual(["28%", "71%", "46%"]);
    expect(row?.cells[1]?.reward).toMatchObject({ kind: "relic", label: "Relic found" });
    expect(row?.cells[0]?.reward).toBeNull();
    expect(row?.cells[2]?.reward).toBeNull();
  });

  it("uses numeric reward tokens outside binary environments", () => {
    const snapshot: LessonSnapshot = {
      ...baseSnapshot,
      policyId: "gaussian-thompson-sampling",
      family: "bayesian",
      presentation: { ...baseSnapshot.presentation, rewardPresentation: "numeric" },
      step: 1,
      selectedArm: 2,
      reward: -0.125,
      history: [
        {
          selectedArm: 2,
          reward: -0.125,
          instantaneousExpectedRegret: 0.3,
          visibleCues: [],
          publicContext: null,
          explanationKey: "repeat",
          diagnostic: {
            before: {},
            after: {},
            decision: { label: "Posterior sample", values: [-0.2, 0.1, 0.4] },
          },
        },
      ],
    };

    const [row] = buildDecisionHistory(snapshot);
    expect(row?.cells[2]?.reward).toMatchObject({
      kind: "numeric",
      label: "Reward -0.13",
    });
  });

  it("shows configured header values only in Free Play", () => {
    const freePlay: LessonSnapshot = {
      ...baseSnapshot,
      mode: "freePlay",
      environment: { probabilities: [0.2, 0.5, 0.8] },
    };
    expect(publicPathDetail(freePlay, 2)).toBe("80% reward chance");
    expect(publicPathDetail({ ...freePlay, mode: "guided" }, 2)).toBeNull();
  });
});
