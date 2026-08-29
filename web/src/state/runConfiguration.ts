import {
  policyCatalog,
  type ParameterDefinition,
  type ParameterValue,
  type ParameterValues,
  type PolicyId,
} from "@/catalog/policies";
import type { LessonMode, LessonSnapshot } from "@/engine/protocol";
import type { ProbabilitySource } from "./portalProbabilities";
import {
  environmentFromConfiguration,
  generateEnvironment,
  validateEnvironmentDraft,
  type EnvironmentDraft,
} from "./environments";

export type DraftParameterValue = string | boolean;

export interface RunConfiguration {
  policyId: PolicyId;
  mode: LessonMode;
  parameters: ParameterValues;
  seed: number;
  environment: Record<string, unknown> | null;
  probabilitySource: ProbabilitySource;
}

export interface RunDraftConfiguration {
  policyId: PolicyId;
  mode: LessonMode;
  parameters: Record<string, DraftParameterValue>;
  seed: string;
  environment: EnvironmentDraft;
  probabilitySource: ProbabilitySource;
}

export interface RunConfigurationErrors {
  parameters: Record<string, string>;
  seed?: string;
  environment: Record<string, string>;
}

export const algorithmLabels = Object.fromEntries(
  Object.values(policyCatalog).map((item) => [item.id, item.label]),
) as Record<PolicyId, string>;

export const modeLabels: Record<LessonMode, string> = {
  guided: "Guided",
  challenge: "Challenge",
  freePlay: "Free play",
};

function isStepAligned(value: number, definition: ParameterDefinition): boolean {
  if (definition.minimum === undefined || definition.step === undefined) return true;
  const steps = (value - definition.minimum) / definition.step;
  return Math.abs(steps - Math.round(steps)) < 1e-8;
}

export function formatParameter(value: number): string {
  return String(Number(value.toFixed(10)));
}

function draftParameters(values: ParameterValues): Record<string, DraftParameterValue> {
  return Object.fromEntries(
    Object.entries(values).map(([key, value]) => [
      key,
      typeof value === "boolean" ? value : value === null ? "" : String(value),
    ]),
  );
}

export function defaultConfiguration(
  policyId: PolicyId,
  mode: LessonMode = "guided",
): RunConfiguration {
  const definition = policyCatalog[policyId];
  const seed = mode === "challenge" ? definition.challengeSeed : definition.guidedSeed;
  const parameters = mode === "freePlay" ? definition.defaults : definition.guidedParameters;
  return {
    policyId,
    mode,
    parameters: { ...parameters },
    seed,
    environment: null,
    probabilitySource: "generated",
  };
}

export function draftFromConfiguration(config: RunConfiguration): RunDraftConfiguration {
  return {
    policyId: config.policyId,
    mode: config.mode,
    parameters: draftParameters(config.parameters),
    seed: String(config.seed),
    environment: environmentFromConfiguration(config.policyId, config.seed, config.environment),
    probabilitySource: config.probabilitySource,
  };
}

export function configurationFromSnapshot(
  snapshot: LessonSnapshot,
  probabilitySource: ProbabilitySource = "generated",
): RunConfiguration {
  return {
    policyId: snapshot.policyId,
    mode: snapshot.mode,
    parameters: { ...snapshot.parameters },
    seed: snapshot.seed,
    environment: snapshot.environment ? { ...snapshot.environment } : null,
    probabilitySource,
  };
}

function validateParameter(
  definition: ParameterDefinition,
  draftValue: DraftParameterValue | undefined,
): { value: ParameterValue | null; error?: string } {
  if (definition.kind === "boolean") {
    if (typeof draftValue !== "boolean") return { value: null, error: "Choose on or off." };
    return { value: draftValue };
  }
  if (definition.kind === "select") {
    if (
      typeof draftValue !== "string" ||
      !definition.options?.some((option) => option.value === draftValue)
    ) {
      return { value: null, error: "Choose one of the available values." };
    }
    return { value: draftValue };
  }

  const text = typeof draftValue === "string" ? draftValue.trim() : "";
  if (definition.kind === "optional-number" && text === "") return { value: null };
  const numeric = Number(text);
  const invalid =
    text === "" ||
    !Number.isFinite(numeric) ||
    (definition.minimum !== undefined && numeric < definition.minimum) ||
    (definition.maximum !== undefined && numeric > definition.maximum) ||
    (definition.kind === "integer" && !Number.isInteger(numeric)) ||
    !isStepAligned(numeric, definition);
  if (invalid) {
    const range =
      definition.minimum !== undefined && definition.maximum !== undefined
        ? ` from ${definition.minimum} to ${definition.maximum}`
        : "";
    const step = definition.step !== undefined ? ` in steps of ${definition.step}` : "";
    return { value: null, error: `Enter a valid value${range}${step}.` };
  }
  return { value: numeric };
}

export function validateRunDraft(draft: RunDraftConfiguration): {
  configuration: RunConfiguration | null;
  errors: RunConfigurationErrors;
} {
  const definition = policyCatalog[draft.policyId];
  const errors: RunConfigurationErrors = { parameters: {}, environment: {} };
  const parameters: ParameterValues = {};
  for (const parameter of definition.parameters) {
    const result = validateParameter(parameter, draft.parameters[parameter.key]);
    if (result.error) errors.parameters[parameter.key] = result.error;
    else parameters[parameter.key] = result.value;
  }

  const initialEpsilon = parameters.initial_epsilon;
  const minimumEpsilon = parameters.min_epsilon;
  if (
    typeof initialEpsilon === "number" &&
    typeof minimumEpsilon === "number" &&
    minimumEpsilon > initialEpsilon
  ) {
    errors.parameters.min_epsilon = "Minimum exploration cannot exceed the initial value.";
  }

  const seedText = draft.seed.trim();
  const seed = Number(seedText);
  if (seedText === "" || !Number.isSafeInteger(seed)) {
    errors.seed = "Enter a safe whole number.";
  }

  let environment: Record<string, unknown> | null = null;
  if (draft.mode === "freePlay") {
    const validation = validateEnvironmentDraft(draft.environment);
    errors.environment = validation.errors;
    environment = validation.value;
  }

  if (
    Object.keys(errors.parameters).length ||
    errors.seed ||
    Object.keys(errors.environment).length
  ) {
    return { configuration: null, errors };
  }

  return {
    configuration: {
      policyId: draft.policyId,
      mode: draft.mode,
      parameters,
      seed,
      environment,
      probabilitySource: draft.probabilitySource,
    },
    errors,
  };
}

export function regenerateDraftEnvironment(policyId: PolicyId, seed: number): EnvironmentDraft {
  return generateEnvironment(policyId, seed);
}

export function configurationsMatch(
  draft: RunDraftConfiguration,
  active: RunConfiguration | null,
): boolean {
  if (!active) return false;
  const { configuration } = validateRunDraft(draft);
  return configuration !== null && JSON.stringify(configuration) === JSON.stringify(active);
}
