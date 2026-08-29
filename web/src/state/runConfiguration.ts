import type { LessonId, LessonMode, LessonSnapshot } from "../engine/protocol";
import {
  generatePortalProbabilities,
  probabilityDraft,
  type PortalProbabilities,
  type ProbabilitySource,
} from "./portalProbabilities";

export interface RunConfiguration {
  lessonId: LessonId;
  mode: LessonMode;
  parameter: number;
  seed: number;
  portalProbabilities: PortalProbabilities | null;
  probabilitySource: ProbabilitySource;
}

export interface RunDraftConfiguration {
  lessonId: LessonId;
  mode: LessonMode;
  parameter: string;
  seed: string;
  portalProbabilities: [string, string, string];
  probabilitySource: ProbabilitySource;
}

export interface ParameterDefinition {
  label: string;
  shortLabel: string;
  minimum: number;
  maximum: number;
  step: number;
  defaultValue: number;
}

export interface RunConfigurationErrors {
  parameter?: string;
  seed?: string;
  portalProbabilities?: [string | undefined, string | undefined, string | undefined];
}

export const parameterDefinitions: Record<LessonId, ParameterDefinition> = {
  "epsilon-greedy": {
    label: "Exploration chance",
    shortLabel: "ε",
    minimum: 0,
    maximum: 1,
    step: 0.01,
    defaultValue: 0.2,
  },
  linucb: {
    label: "Confidence width",
    shortLabel: "α",
    minimum: 0.05,
    maximum: 4,
    step: 0.05,
    defaultValue: 1,
  },
};

export const algorithmLabels: Record<LessonId, string> = {
  "epsilon-greedy": "ε-greedy",
  linucb: "LinUCB",
};

export const modeLabels: Record<LessonMode, string> = {
  guided: "Guided",
  challenge: "Challenge",
  freePlay: "Free play",
};

function isStepAligned(value: number, definition: ParameterDefinition): boolean {
  const steps = (value - definition.minimum) / definition.step;
  return Math.abs(steps - Math.round(steps)) < 1e-8;
}

export function formatParameter(value: number): string {
  return String(Number(value.toFixed(10)));
}

export function draftFromConfiguration(config: RunConfiguration): RunDraftConfiguration {
  const probabilities = config.portalProbabilities ?? generatePortalProbabilities(config.seed);
  return {
    lessonId: config.lessonId,
    mode: config.mode,
    parameter: formatParameter(config.parameter),
    seed: String(config.seed),
    portalProbabilities: probabilityDraft(probabilities),
    probabilitySource: config.probabilitySource,
  };
}

export function configurationFromSnapshot(
  snapshot: LessonSnapshot,
  probabilitySource: ProbabilitySource = "generated",
): RunConfiguration {
  const parameter =
    snapshot.lessonId === "epsilon-greedy"
      ? snapshot.parameters.epsilon
      : snapshot.parameters.alpha;
  if (typeof parameter !== "number") {
    throw new Error("The worker returned a snapshot without the selected parameter.");
  }
  return {
    lessonId: snapshot.lessonId,
    mode: snapshot.mode,
    parameter,
    seed: snapshot.seed,
    portalProbabilities:
      snapshot.environment?.probabilities && snapshot.lessonId === "epsilon-greedy"
        ? [...snapshot.environment.probabilities]
        : null,
    probabilitySource,
  };
}

export function validateRunDraft(draft: RunDraftConfiguration): {
  configuration: RunConfiguration | null;
  errors: RunConfigurationErrors;
} {
  const definition = parameterDefinitions[draft.lessonId];
  const errors: RunConfigurationErrors = {};
  const parameterText = draft.parameter.trim();
  const parameter = Number(parameterText);

  if (
    parameterText === "" ||
    !Number.isFinite(parameter) ||
    parameter < definition.minimum ||
    parameter > definition.maximum ||
    !isStepAligned(parameter, definition)
  ) {
    errors.parameter = `Enter a value from ${definition.minimum} to ${definition.maximum} in steps of ${definition.step}.`;
  }

  const seedText = draft.seed.trim();
  const seed = Number(seedText);
  if (seedText === "" || !Number.isSafeInteger(seed)) {
    errors.seed = "Enter a safe whole number.";
  }

  let portalProbabilities: PortalProbabilities | null = null;
  if (draft.lessonId === "epsilon-greedy" && draft.mode === "freePlay") {
    const probabilityErrors: [string | undefined, string | undefined, string | undefined] = [
      undefined,
      undefined,
      undefined,
    ];
    const normalized = draft.portalProbabilities.map((text, index) => {
      const trimmed = text.trim();
      const percentage = Number(trimmed);
      if (
        trimmed === "" ||
        !Number.isFinite(percentage) ||
        percentage < 0 ||
        percentage > 100 ||
        Math.abs(percentage * 10 - Math.round(percentage * 10)) > 1e-8
      ) {
        probabilityErrors[index] = "Enter 0 to 100 in steps of 0.1.";
      }
      return Math.round(percentage * 10) / 1000;
    });
    if (probabilityErrors.some(Boolean)) errors.portalProbabilities = probabilityErrors;
    else portalProbabilities = normalized as PortalProbabilities;
  }

  if (errors.parameter || errors.seed || errors.portalProbabilities) {
    return { configuration: null, errors };
  }

  return {
    configuration: {
      lessonId: draft.lessonId,
      mode: draft.mode,
      parameter,
      seed,
      portalProbabilities,
      probabilitySource: draft.probabilitySource,
    },
    errors,
  };
}

export function configurationsMatch(
  draft: RunDraftConfiguration,
  active: RunConfiguration | null,
): boolean {
  if (!active) return false;
  const { configuration } = validateRunDraft(draft);
  return (
    configuration !== null &&
    configuration.lessonId === active.lessonId &&
    configuration.mode === active.mode &&
    configuration.parameter === active.parameter &&
    configuration.seed === active.seed &&
    configuration.probabilitySource === active.probabilitySource &&
    ((configuration.portalProbabilities === null && active.portalProbabilities === null) ||
      (configuration.portalProbabilities !== null &&
        active.portalProbabilities !== null &&
        configuration.portalProbabilities.every(
          (value, index) => value === active.portalProbabilities?.[index],
        )))
  );
}
