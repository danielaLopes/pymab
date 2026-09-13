import { generatePortalProbabilities } from "./portalProbabilities";

describe("portal probability generation", () => {
  it("is deterministic and assigns one value from each teaching band", () => {
    const first = generatePortalProbabilities(123456);
    const replay = generatePortalProbabilities(123456);
    expect(replay).toEqual(first);

    const sorted = [...first].sort((left, right) => left - right);
    expect(sorted[0]).toBeGreaterThanOrEqual(0.1);
    expect(sorted[0]).toBeLessThanOrEqual(0.35);
    expect(sorted[1]).toBeGreaterThanOrEqual(0.4);
    expect(sorted[1]).toBeLessThanOrEqual(0.65);
    expect(sorted[2]).toBeGreaterThanOrEqual(0.7);
    expect(sorted[2]).toBeLessThanOrEqual(0.95);
    expect(first.every((value) => Number.isInteger(value * 1000))).toBe(true);
  });

  it("changes the generated environment when the seed changes", () => {
    expect(generatePortalProbabilities(10)).not.toEqual(generatePortalProbabilities(11));
  });
});
