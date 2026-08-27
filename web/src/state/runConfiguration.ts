import type { LessonId, LessonMode, LessonSnapshot } from "../engine/protocol";

export interface RunConfiguration {
  lessonId: LessonId;
  mode: LessonMode;
  parameter: number;
  seed: number;
}

export interface RunDraftConfiguration {
  lessonId: LessonId;
  mode: LessonMode;
  parameter: string;
  seed: string;
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
  return {
    lessonId: config.lessonId,
    mode: config.mode,
    parameter: formatParameter(config.parameter),
    seed: String(config.seed),
  };
}

export function configurationFromSnapshot(snapshot: LessonSnapshot): RunConfiguration {
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

  if (errors.parameter || errors.seed) {
    return { configuration: null, errors };
  }

  return {
    configuration: {
      lessonId: draft.lessonId,
      mode: draft.mode,
      parameter,
      seed,
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
    configuration.seed === active.seed
  );
}
