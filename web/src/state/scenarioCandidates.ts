export const recommendationCandidateKinds = [
  "article",
  "product",
  "tutorial",
  "video",
  "podcast",
  "newsletter",
  "course",
  "event",
  "tool",
  "message",
  "offer",
  "download",
] as const;

export type RecommendationCandidateKind = (typeof recommendationCandidateKinds)[number];

export const recommendationCandidateKindLabels: Record<RecommendationCandidateKind, string> = {
  article: "Article",
  product: "Product",
  tutorial: "Tutorial",
  video: "Video",
  podcast: "Podcast",
  newsletter: "Newsletter",
  course: "Course",
  event: "Event",
  tool: "Tool",
  message: "Message",
  offer: "Offer",
  download: "Download",
};

export const recommendationFeatureIds = [
  "visitor",
  "engagement",
  "visit",
  "device",
  "account_age",
  "recent_activity",
  "price_sensitivity",
  "session_depth",
  "traffic_source",
] as const;

export type RecommendationFeatureId = (typeof recommendationFeatureIds)[number];
export type RecommendationModelFeatureId = "base" | RecommendationFeatureId;

export interface RecommendationFeatureDefinition {
  id: RecommendationFeatureId;
  label: string;
  type: "binary" | "numeric";
  help: string;
  negativeLabel?: string;
  positiveLabel?: string;
  minimum?: number;
  maximum?: number;
  unit?: string;
}

export const recommendationFeatureCatalog: Record<
  RecommendationFeatureId,
  RecommendationFeatureDefinition
> = {
  visitor: {
    id: "visitor",
    label: "Visitor type",
    type: "binary",
    help: "Whether the visitor is new or returning.",
    negativeLabel: "new",
    positiveLabel: "returning",
  },
  engagement: {
    id: "engagement",
    label: "Engagement",
    type: "numeric",
    help: "A score from 0 to 100 for activity during the current visit.",
    minimum: 0,
    maximum: 100,
    unit: "/100",
  },
  visit: {
    id: "visit",
    label: "Visit timing",
    type: "binary",
    help: "Whether the visit happens on a weekday or weekend.",
    negativeLabel: "weekday",
    positiveLabel: "weekend",
  },
  device: {
    id: "device",
    label: "Device",
    type: "binary",
    help: "Whether the visitor is using a desktop or mobile device.",
    negativeLabel: "desktop",
    positiveLabel: "mobile",
  },
  account_age: {
    id: "account_age",
    label: "Account age",
    type: "numeric",
    help: "The number of days since the account was created.",
    minimum: 0,
    maximum: 3650,
    unit: "days",
  },
  recent_activity: {
    id: "recent_activity",
    label: "Recent activity",
    type: "numeric",
    help: "The number of interactions in the visitor's recent history.",
    minimum: 0,
    maximum: 20,
    unit: "interactions",
  },
  price_sensitivity: {
    id: "price_sensitivity",
    label: "Price sensitivity",
    type: "numeric",
    help: "A score from 0 for low sensitivity to 100 for high sensitivity.",
    minimum: 0,
    maximum: 100,
    unit: "/100",
  },
  session_depth: {
    id: "session_depth",
    label: "Session depth",
    type: "numeric",
    help: "The number of pages viewed in the current session.",
    minimum: 1,
    maximum: 12,
    unit: "pages",
  },
  traffic_source: {
    id: "traffic_source",
    label: "Traffic source",
    type: "binary",
    help: "Whether the visit came from an organic or paid source.",
    negativeLabel: "organic",
    positiveLabel: "paid",
  },
};

export const defaultRecommendationFeatureIds: RecommendationFeatureId[] = [
  "visitor",
  "engagement",
  "visit",
];

export type RecommendationCoefficients = Partial<Record<RecommendationModelFeatureId, number>>;

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
    coefficients: { base: 0.1, visitor: -0.65, engagement: 0.2, visit: 0.35 },
  },
  {
    id: "candidate-2",
    name: "Product",
    symbolKind: "product",
    coefficients: { base: -0.35, visitor: 0.85, engagement: 0.7, visit: -0.15 },
  },
  {
    id: "candidate-3",
    name: "Tutorial",
    symbolKind: "tutorial",
    coefficients: { base: 0.05, visitor: -0.75, engagement: -0.55, visit: 0.45 },
  },
];

export function cloneRecommendationCandidates(
  candidates: RecommendationCandidate[] = defaultRecommendationCandidates,
): RecommendationCandidate[] {
  return candidates.map((candidate) => ({
    ...candidate,
    coefficients: { ...candidate.coefficients },
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

export function generateCandidateCoefficient(
  seed: number,
  candidateId: string,
  featureId: RecommendationModelFeatureId,
): number {
  let state = (seed ^ textHash(`${candidateId}:${featureId}`)) >>> 0;
  state = (state + 0x6d2b79f5) >>> 0;
  let value = state;
  value = Math.imul(value ^ (value >>> 15), value | 1);
  value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
  const random = ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  return roundCoefficient(featureId === "base" ? random * 0.8 - 0.4 : random * 1.7 - 0.85);
}

export function generateCandidateCoefficients(
  seed: number,
  candidateId: string,
  featureIds: RecommendationFeatureId[] = defaultRecommendationFeatureIds,
): RecommendationCoefficients {
  const modelFeatureIds: RecommendationModelFeatureId[] = ["base", ...featureIds];
  return Object.fromEntries(
    modelFeatureIds.map((featureId) => [
      featureId,
      generateCandidateCoefficient(seed, candidateId, featureId),
    ]),
  );
}

function nextCandidateName(
  kind: RecommendationCandidateKind,
  candidates: RecommendationCandidate[],
): string {
  const base = recommendationCandidateKindLabels[kind];
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
  featureIds: RecommendationFeatureId[] = defaultRecommendationFeatureIds,
): RecommendationCandidate[] {
  if (candidates.length >= 8) return candidates;
  const id = `candidate-${nextOrdinal}`;
  return [
    ...candidates,
    {
      id,
      name: nextCandidateName(kind, candidates),
      symbolKind: kind,
      coefficients: generateCandidateCoefficients(seed, id, featureIds),
    },
  ];
}

export function addRecommendationFeature(
  candidates: RecommendationCandidate[],
  featureId: RecommendationFeatureId,
  seed: number,
): RecommendationCandidate[] {
  return candidates.map((candidate) => ({
    ...candidate,
    coefficients: {
      ...candidate.coefficients,
      [featureId]:
        candidate.coefficients[featureId] ??
        generateCandidateCoefficient(seed, candidate.id, featureId),
    },
  }));
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

export function validRecommendationFeatureIds(featureIds: RecommendationFeatureId[]): boolean {
  return (
    featureIds.length <= 8 &&
    new Set(featureIds).size === featureIds.length &&
    featureIds.every((featureId) => recommendationFeatureIds.includes(featureId))
  );
}

export function validRecommendationCandidates(
  candidates: RecommendationCandidate[],
  featureIds: RecommendationFeatureId[] = defaultRecommendationFeatureIds,
): boolean {
  const requiredFeatures: RecommendationModelFeatureId[] = ["base", ...featureIds];
  return (
    candidates.length >= 2 &&
    candidates.length <= 8 &&
    Object.keys(candidateNameErrors(candidates)).length === 0 &&
    candidates.every(
      (candidate) =>
        recommendationCandidateKinds.includes(candidate.symbolKind) &&
        requiredFeatures.every((featureId) => Number.isFinite(candidate.coefficients[featureId])),
    )
  );
}
