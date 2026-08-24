import { initialLessonState, lessonReducer } from "./lessonReducer";
import type { LessonSnapshot } from "../engine/protocol";

const snapshot = {
  lessonId: "epsilon-greedy",
  mode: "guided",
  seed: 42,
  packageVersion: "2.0.0",
  sourceCommit: "abc",
  sessionId: "s",
  step: 0,
  horizon: 12,
  parameters: { epsilon: 0.2 },
  gateIds: ["moon", "sun", "star"],
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
  history: [],
  hiddenTruth: null,
  generatedCode: "print('hello')",
} satisfies LessonSnapshot;

describe("lessonReducer", () => {
  it("moves through guided, pending, and debrief states", () => {
    const guided = lessonReducer(initialLessonState, { type: "started", mode: "guided", snapshot });
    expect(guided.phase).toBe("guided");
    expect(lessonReducer(guided, { type: "pending", value: true }).pending).toBe(true);
    const complete = { ...snapshot, step: 12, completed: true };
    expect(lessonReducer(guided, { type: "snapshot", snapshot: complete }).phase).toBe("debrief");
  });

  it("represents challenge setup, errors, and unsupported browsers explicitly", () => {
    expect(lessonReducer(initialLessonState, { type: "challengeSetup" }).phase).toBe(
      "challengeSetup",
    );
    expect(lessonReducer(initialLessonState, { type: "error", message: "boom" })).toMatchObject({
      phase: "error",
      error: "boom",
    });
    expect(
      lessonReducer(initialLessonState, { type: "unsupported", message: "no wasm" }).phase,
    ).toBe("unsupported");
  });
});
