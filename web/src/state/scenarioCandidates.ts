export const recommendationCandidateKinds = ["article", "product", "tutorial"] as const;

export type RecommendationCandidateKind = (typeof recommendationCandidateKinds)[number];
export type RecommendationCoefficients = [number, number, number, number];

export interface RecommendationCandidate {
  id: string;
  name: string;
  symbolKind: RecommendationCandidateKind;
  coefficients: RecommendationCoefficients;
}

export const defaultRecommendationCandidates: RecommendationCandidate[] = [
  {
    id: "candidate-1",
    name: "Article",
    symbolKind: "article",
    coefficients: [0.1, -0.65, 0.2, 0.35],
  },
  {
    id: "candidate-2",
    name: "Product",
    symbolKind: "product",
    coefficients: [-0.35, 0.85, 0.7, -0.15],
  },
  {
    id: "candidate-3",
    name: "Tutorial",
    symbolKind: "tutorial",
    coefficients: [0.05, -0.75, -0.55, 0.45],
  },
];

export function cloneRecommendationCandidates(
  candidates: RecommendationCandidate[] = defaultRecommendationCandidates,
): RecommendationCandidate[] {
  return candidates.map((candidate) => ({
    ...candidate,
    coefficients: [...candidate.coefficients],
  }));
}

function textHash(value: string): number {
  let hash = 2166136261;
  for (let index = 0; index < value.length; index += 1) {
    hash ^= value.charCodeAt(index);
    hash = Math.imul(hash, 16777619);
  }
  return hash >>> 0;
}

function roundCoefficient(value: number): number {
  return Number(value.toFixed(2));
}

export function generateCandidateCoefficients(
  seed: number,
  candidateId: string,
): RecommendationCoefficients {
  let state = (seed ^ textHash(candidateId)) >>> 0;
  const random = () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  };
  return [
    roundCoefficient(random() * 0.8 - 0.4),
    roundCoefficient(random() * 1.7 - 0.85),
    roundCoefficient(random() * 1.7 - 0.85),
    roundCoefficient(random() * 1.7 - 0.85),
  ];
}

function nextCandidateName(
  kind: RecommendationCandidateKind,
  candidates: RecommendationCandidate[],
): string {
  const base = `${kind[0]!.toUpperCase()}${kind.slice(1)}`;
  const used = new Set(candidates.map((candidate) => candidate.name.trim().toLowerCase()));
  if (!used.has(base.toLowerCase())) return base;
  let suffix = 2;
  while (used.has(`${base} ${suffix}`.toLowerCase())) suffix += 1;
  return `${base} ${suffix}`;
}

export function appendRecommendationCandidate(
  candidates: RecommendationCandidate[],
  kind: RecommendationCandidateKind,
  seed: number,
  nextOrdinal: number,
): RecommendationCandidate[] {
  if (candidates.length >= 8) return candidates;
  const id = `candidate-${nextOrdinal}`;
  return [
    ...candidates,
    {
      id,
      name: nextCandidateName(kind, candidates),
      symbolKind: kind,
      coefficients: generateCandidateCoefficients(seed, id),
    },
  ];
}

export function candidateNameErrors(candidates: RecommendationCandidate[]): Record<string, string> {
  const errors: Record<string, string> = {};
  const counts = new Map<string, number>();
  for (const candidate of candidates) {
    const normalized = candidate.name.trim().toLowerCase();
    counts.set(normalized, (counts.get(normalized) ?? 0) + 1);
  }
  for (const candidate of candidates) {
    const normalized = candidate.name.trim().toLowerCase();
    if (!normalized) errors[candidate.id] = "Enter a candidate name.";
    else if ((counts.get(normalized) ?? 0) > 1)
      errors[candidate.id] = "Candidate names must be unique.";
  }
  return errors;
}

export function validRecommendationCandidates(candidates: RecommendationCandidate[]): boolean {
  return (
    candidates.length >= 2 &&
    candidates.length <= 8 &&
    Object.keys(candidateNameErrors(candidates)).length === 0 &&
    candidates.every(
      (candidate) =>
        recommendationCandidateKinds.includes(candidate.symbolKind) &&
        candidate.coefficients.length === 4 &&
        candidate.coefficients.every(Number.isFinite),
    )
  );
}
