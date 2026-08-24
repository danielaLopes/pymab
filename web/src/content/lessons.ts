import type { LessonId } from "../engine/protocol";

export const lessonContent: Record<
  LessonId,
  {
    eyebrow: string;
    title: string;
    intro: string;
    parameter: "epsilon" | "alpha";
    parameterLabel: string;
    choices: number[];
    guidedSeed: number;
    challengeSeed: number;
    target: string;
  }
> = {
  "epsilon-greedy": {
    eyebrow: "Mission 01 · No context",
    title: "The Three Ancient Gates",
    intro:
      "Each chamber offers the same three gates. PyMAB must decide when to trust what it knows—and when to test another path.",
    parameter: "epsilon",
    parameterLabel: "Exploration chance ε",
    choices: [0, 0.05, 0.1, 0.2, 0.4, 0.8],
    guidedSeed: 42,
    challengeSeed: 7,
    target: "Collect 12 relics with expected regret ≤ 3.25.",
  },
  linucb: {
    eyebrow: "Mission 02 · Contextual",
    title: "The Labyrinth of Signals",
    intro:
      "Light, echo, and tide change every chamber. LinUCB learns which gate works best for the signals it can see now.",
    parameter: "alpha",
    parameterLabel: "Confidence width α",
    choices: [0.1, 0.25, 0.5, 1, 2, 4],
    guidedSeed: 31415,
    challengeSeed: 20260824,
    target: "Collect 10 relics with expected regret ≤ 3.25.",
  },
};

export const explanationCopy: Record<string, string> = {
  ready: "The policy is ready. Advance to let PyMAB choose a gate.",
  "epsilon.explore": "Exploration: ε opened the door to a deliberate experiment.",
  "epsilon.exploit": "Exploitation: PyMAB chose among the gates with the highest current estimate.",
  "epsilon.firstObservation":
    "First observation: one reward is evidence, not certainty. The selected gate's estimate now moves toward what happened.",
  "epsilon.firstExploration":
    "First definite exploration: the ε draw deliberately tested a gate instead of following the current favourite.",
  "epsilon.estimateUpdate":
    "Estimate update: only the opened gate learns from this reward; the other estimates stay unchanged.",
  "epsilon.cumulativeRegret":
    "Cumulative expected regret totals the reward probability forgone by every choice in this expedition.",
  "linucb.decision":
    "LinUCB combined its predicted reward with an uncertainty bonus for these signals.",
  "linucb.initialUncertainty":
    "Initial uncertainty: with no evidence yet, every gate receives the same optimism bonus.",
  "linucb.contextPrediction":
    "Context-dependent prediction: the same learned coefficients produce new scores when light, echo, and tide change.",
  "linucb.confidenceBonus":
    "Confidence bonus: α scales how strongly LinUCB values evidence it has not gathered yet.",
  "linucb.update":
    "Learning update: only the chosen gate's coefficient vector and confidence matrix change.",
  "linucb.changedContext":
    "A different signal pattern can recommend a different gate without any path or navigation state.",
};
