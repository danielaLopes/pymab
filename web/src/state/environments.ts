import { policyCatalog, type EnvironmentKind, type PolicyId } from "@/catalog/policies";
import { generatePortalProbabilities, type PortalProbabilities } from "./portalProbabilities";

export type Triple<T> = [T, T, T];
export type Quad<T> = [T, T, T, T];

export type EnvironmentDraft =
  | { kind: "stationary-bernoulli" | "best-arm"; probabilities: Triple<string> }
  | { kind: "stationary-gaussian"; means: Triple<string>; standardDeviation: string }
  | {
      kind: "changing-bernoulli";
      phases: Array<{ start: string; probabilities: Triple<string> }>;
    }
  | { kind: "adversarial"; rewards: Array<Triple<string>> }
  | {
      kind: "contextual-linear" | "contextual-logistic";
      theta: Triple<Quad<string>>;
      standardDeviation?: string;
    };

export interface EnvironmentValidation {
  value: Record<string, unknown> | null;
  errors: Record<string, string>;
}

const defaultTheta = [
  [0.1, -1.2, 0.2, -0.8],
  [0, 1, 0.3, 1],
  [0.2, 0, -1.1, 0.2],
] as const;

function text(value: number): string {
  return String(Number(value.toFixed(3)));
}

function seededUnit(seed: number, index: number): number {
  let value = (seed ^ (index * 0x9e3779b9)) >>> 0;
  value = Math.imul(value ^ (value >>> 16), 0x21f0aaad);
  value = Math.imul(value ^ (value >>> 15), 0x735a2d97);
  return ((value ^ (value >>> 15)) >>> 0) / 0x1_0000_0000;
}

export function generateEnvironment(policyId: PolicyId, seed: number): EnvironmentDraft {
  const definition = policyCatalog[policyId];
  const kind = definition.environment;
  if (kind === "stationary-bernoulli" || kind === "best-arm") {
    return {
      kind,
      probabilities: generatePortalProbabilities(seed).map((value) =>
        text(value * 100),
      ) as Triple<string>,
    };
  }
  if (kind === "stationary-gaussian") {
    const means = [0, 1, 2].map((index) => text(-0.5 + seededUnit(seed, index) * 1.5));
    means.sort((left, right) => Number(left) - Number(right));
    return { kind, means: means as Triple<string>, standardDeviation: "0.5" };
  }
  if (kind === "changing-bernoulli") {
    const horizon = definition.challengeHorizon;
    const first = generatePortalProbabilities(seed).map((value) =>
      text(value * 100),
    ) as Triple<string>;
    const second: Triple<string> = [first[2], first[0], first[1]];
    const third: Triple<string> = [first[1], first[2], first[0]];
    return {
      kind,
      phases: [
        { start: "0", probabilities: first },
        { start: String(Math.max(1, Math.floor(horizon / 3))), probabilities: second },
        { start: String(Math.max(2, Math.floor((2 * horizon) / 3))), probabilities: third },
      ],
    };
  }
  if (kind === "adversarial") {
    return {
      kind,
      rewards: Array.from({ length: definition.challengeHorizon }, (_, round) => {
        const leader = (round + Math.abs(seed)) % 3;
        const row: Triple<string> = ["0.1", "0.1", "0.1"];
        row[leader] = "1";
        row[(leader + 1) % 3] = "0.4";
        return row;
      }),
    };
  }
  const theta = defaultTheta.map(
    (row, rowIndex) =>
      row.map((value, columnIndex) =>
        text(value + (seededUnit(seed, rowIndex * 4 + columnIndex) - 0.5) * 0.2),
      ) as Quad<string>,
  ) as unknown as Triple<Quad<string>>;
  return kind === "contextual-linear" ? { kind, theta, standardDeviation: "0.2" } : { kind, theta };
}

export function environmentFromConfiguration(
  policyId: PolicyId,
  seed: number,
  environment: Record<string, unknown> | null,
): EnvironmentDraft {
  const generated = generateEnvironment(policyId, seed);
  if (!environment) return generated;
  const kind = policyCatalog[policyId].environment;
  try {
    if (kind === "stationary-bernoulli" || kind === "best-arm") {
      const values = environment.probabilities as number[];
      if (values.length !== 3) return generated;
      return { kind, probabilities: values.map((value) => text(value * 100)) as Triple<string> };
    }
    if (kind === "stationary-gaussian") {
      const values = environment.means as number[];
      if (values.length !== 3) return generated;
      return {
        kind,
        means: values.map(text) as Triple<string>,
        standardDeviation: text(Number(environment.standardDeviation)),
      };
    }
    if (kind === "changing-bernoulli") {
      const phases = environment.phases as Array<{ start: number; probabilities: number[] }>;
      return {
        kind,
        phases: phases.map((phase) => ({
          start: String(phase.start),
          probabilities: phase.probabilities.map((value) => text(value * 100)) as Triple<string>,
        })),
      };
    }
    if (kind === "adversarial") {
      const rewards = environment.rewards as number[][];
      return { kind, rewards: rewards.map((row) => row.map(text) as Triple<string>) };
    }
    const theta = (environment.theta as number[][]).map(
      (row) => row.map(text) as Quad<string>,
    ) as Triple<Quad<string>>;
    return kind === "contextual-linear"
      ? { kind, theta, standardDeviation: text(Number(environment.standardDeviation)) }
      : { kind, theta };
  } catch {
    return generated;
  }
}

function parseNumber(
  raw: string,
  path: string,
  errors: Record<string, string>,
  options: { minimum?: number; maximum?: number; integer?: boolean } = {},
): number {
  const value = Number(raw.trim());
  if (
    raw.trim() === "" ||
    !Number.isFinite(value) ||
    (options.minimum !== undefined && value < options.minimum) ||
    (options.maximum !== undefined && value > options.maximum) ||
    (options.integer && !Number.isInteger(value))
  ) {
    errors[path] =
      options.minimum === 0 && options.maximum === 100
        ? "Enter a value from 0 to 100."
        : "Enter a valid finite value.";
  }
  return value;
}

export function validateEnvironmentDraft(draft: EnvironmentDraft): EnvironmentValidation {
  const errors: Record<string, string> = {};
  if (draft.kind === "stationary-bernoulli" || draft.kind === "best-arm") {
    const probabilities = draft.probabilities.map(
      (value, index) =>
        Math.round(
          parseNumber(value, `probabilities.${index}`, errors, { minimum: 0, maximum: 100 }) * 10,
        ) / 1000,
    ) as PortalProbabilities;
    return { value: Object.keys(errors).length ? null : { probabilities }, errors };
  }
  if (draft.kind === "stationary-gaussian") {
    const means = draft.means.map((value, index) => parseNumber(value, `means.${index}`, errors));
    const standardDeviation = parseNumber(draft.standardDeviation, "standardDeviation", errors, {
      minimum: Number.MIN_VALUE,
    });
    return { value: Object.keys(errors).length ? null : { means, standardDeviation }, errors };
  }
  if (draft.kind === "changing-bernoulli") {
    const phases = draft.phases.map((phase, phaseIndex) => ({
      start: parseNumber(phase.start, `phases.${phaseIndex}.start`, errors, {
        minimum: 0,
        integer: true,
      }),
      probabilities: phase.probabilities.map(
        (value, armIndex) =>
          Math.round(
            parseNumber(value, `phases.${phaseIndex}.probabilities.${armIndex}`, errors, {
              minimum: 0,
              maximum: 100,
            }) * 10,
          ) / 1000,
      ),
    }));
    const starts = phases.map((phase) => phase.start);
    if (
      starts[0] !== 0 ||
      starts.some((start, index) => index > 0 && start <= starts[index - 1]!)
    ) {
      errors.phases = "Phase starts must begin at round 0 and increase.";
    }
    return { value: Object.keys(errors).length ? null : { phases }, errors };
  }
  if (draft.kind === "adversarial") {
    const rewards = draft.rewards.map((row, round) =>
      row.map((value, arm) =>
        parseNumber(value, `rewards.${round}.${arm}`, errors, { minimum: 0, maximum: 1 }),
      ),
    );
    return { value: Object.keys(errors).length ? null : { rewards }, errors };
  }
  const contextual = draft as Extract<
    EnvironmentDraft,
    { kind: "contextual-linear" | "contextual-logistic" }
  >;
  const theta = contextual.theta.map((row, arm) =>
    row.map((value, coefficient) => parseNumber(value, `theta.${arm}.${coefficient}`, errors)),
  );
  const value: Record<string, unknown> = { theta };
  if (contextual.kind === "contextual-linear") {
    value.standardDeviation = parseNumber(
      contextual.standardDeviation ?? "",
      "standardDeviation",
      errors,
      { minimum: Number.MIN_VALUE },
    );
  }
  return { value: Object.keys(errors).length ? null : value, errors };
}

export function environmentLabel(kind: EnvironmentKind): string {
  return {
    "stationary-bernoulli": "Portal reward chances",
    "stationary-gaussian": "Numeric reward model",
    "changing-bernoulli": "Reward phase schedule",
    "best-arm": "Portal reward chances",
    adversarial: "Reward table",
    "contextual-linear": "Signal coefficients",
    "contextual-logistic": "Signal coefficients",
  }[kind];
}
