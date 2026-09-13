export type PortalProbabilities = [number, number, number];
export type ProbabilitySource = "generated" | "custom";

const bands: ReadonlyArray<readonly [number, number]> = [
  [100, 350],
  [400, 650],
  [700, 950],
];

function seededRandom(seed: number): () => number {
  let state = 2166136261;
  const input = `${seed}:pymab-portals:v1`;
  for (let index = 0; index < input.length; index += 1) {
    state ^= input.charCodeAt(index);
    state = Math.imul(state, 16777619);
  }

  return () => {
    state = (state + 0x6d2b79f5) | 0;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  };
}

export function generatePortalProbabilities(seed: number): PortalProbabilities {
  const random = seededRandom(seed);
  const values = bands.map(([minimum, maximum]) =>
    Math.min(maximum, minimum + Math.floor(random() * (maximum - minimum + 1))),
  );

  for (let index = values.length - 1; index > 0; index -= 1) {
    const swapIndex = Math.floor(random() * (index + 1));
    [values[index], values[swapIndex]] = [values[swapIndex]!, values[index]!];
  }

  return values.map((value) => value / 1000) as PortalProbabilities;
}

export function formatProbabilityPercent(probability: number): string {
  return String(Number((probability * 100).toFixed(1)));
}

export function probabilityDraft(probabilities: PortalProbabilities): [string, string, string] {
  return probabilities.map(formatProbabilityPercent) as [string, string, string];
}
