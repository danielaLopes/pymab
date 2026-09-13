import type { LessonMode, ScenarioId } from "@/engine/protocol";

export interface ScenarioDefinition {
  id: ScenarioId;
  title: string;
  eyebrow: string;
  summary: string;
  intro: string;
  policyId: "logistic-contextual-bandit" | "linucb";
  className: "LogisticContextualBanditPolicy" | "LinUCBPolicy";
  guidedSeed: number;
  challengeSeed: number;
  parameters: Record<string, number>;
  parameterDefinitions: Array<{
    key: string;
    label: string;
    help: string;
    minimum: number;
    maximum: number;
    step: number;
  }>;
}

export const scenarioCatalog: Record<ScenarioId, ScenarioDefinition> = {
  recommendations: {
    id: "recommendations",
    title: "Recommendation systems",
    eyebrow: "Applied scenario · Logistic contextual bandit",
    summary: "Choose one item for one slot, then learn from an immediate click or no-click result.",
    intro:
      "See how a contextual bandit can personalize a single recommendation slot when feedback arrives immediately.",
    policyId: "logistic-contextual-bandit",
    className: "LogisticContextualBanditPolicy",
    guidedSeed: 2401,
    challengeSeed: 3401,
    parameters: { epsilon: 0.08, learning_rate: 0.18, l2: 0.01 },
    parameterDefinitions: [
      {
        key: "epsilon",
        label: "Exploration rate",
        help: "Chance of trying a random recommendation.",
        minimum: 0,
        maximum: 1,
        step: 0.01,
      },
      {
        key: "learning_rate",
        label: "Learning rate",
        help: "How strongly one click changes the selected model.",
        minimum: 0.01,
        maximum: 1,
        step: 0.01,
      },
      {
        key: "l2",
        label: "L2 penalty",
        help: "Regularization that limits coefficient growth.",
        minimum: 0,
        maximum: 1,
        step: 0.01,
      },
    ],
  },
  "defensive-verification": {
    id: "defensive-verification",
    title: "Defensive verification",
    eyebrow: "Applied scenario · Linear contextual bandit",
    summary:
      "Choose proportionate verification while balancing protection and legitimate-user friction.",
    intro:
      "Use LinUCB only inside an approved range of reversible actions. Clear low-risk and hard-block decisions remain outside the bandit.",
    policyId: "linucb",
    className: "LinUCBPolicy",
    guidedSeed: 2402,
    challengeSeed: 3402,
    parameters: { alpha: 0.75, l2: 1 },
    parameterDefinitions: [
      {
        key: "alpha",
        label: "Exploration strength",
        help: "How much uncertainty increases an action's score.",
        minimum: 0.05,
        maximum: 3,
        step: 0.05,
      },
      {
        key: "l2",
        label: "L2 regularization",
        help: "How strongly estimates begin anchored toward zero.",
        minimum: 0.1,
        maximum: 5,
        step: 0.1,
      },
    ],
  },
};

export function scenarioSeed(id: ScenarioId, mode: LessonMode): number {
  const scenario = scenarioCatalog[id];
  return mode === "challenge" ? scenario.challengeSeed : scenario.guidedSeed;
}

export function isScenarioId(value: string): value is ScenarioId {
  return value in scenarioCatalog;
}
