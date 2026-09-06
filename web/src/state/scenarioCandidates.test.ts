import {
  addRecommendationFeature,
  appendRecommendationCandidate,
  candidateNameErrors,
  cloneRecommendationCandidates,
  defaultRecommendationFeatureIds,
  generateCandidateCoefficients,
  recommendationCandidateKinds,
  validRecommendationCandidates,
  validRecommendationFeatureIds,
} from "./scenarioCandidates";

describe("recommendation candidates", () => {
  it("generates stable bounded coefficients", () => {
    const first = generateCandidateCoefficients(2401, "candidate-4");
    expect(first).toEqual(generateCandidateCoefficients(2401, "candidate-4"));
    expect(first).not.toEqual(generateCandidateCoefficients(2402, "candidate-4"));
    expect(first.base).toBeGreaterThanOrEqual(-0.4);
    expect(first.base).toBeLessThanOrEqual(0.4);
    expect(
      defaultRecommendationFeatureIds.every((featureId) => {
        const value = first[featureId];
        return value !== undefined && value >= -0.85 && value <= 0.85;
      }),
    ).toBe(true);
  });

  it("adds a uniquely named candidate with a stable ID", () => {
    const candidates = appendRecommendationCandidate(
      cloneRecommendationCandidates(),
      "tutorial",
      2401,
      4,
    );
    expect(candidates.at(-1)).toMatchObject({
      id: "candidate-4",
      name: "Tutorial 2",
      symbolKind: "tutorial",
    });
    expect(validRecommendationCandidates(candidates)).toBe(true);
  });

  it("rejects empty and case-insensitively duplicated names", () => {
    const candidates = cloneRecommendationCandidates();
    candidates[0]!.name = " tutorial ";
    candidates[1]!.name = "";
    expect(candidateNameErrors(candidates)).toEqual({
      "candidate-1": "Candidate names must be unique.",
      "candidate-2": "Enter a candidate name.",
      "candidate-3": "Candidate names must be unique.",
    });
    expect(validRecommendationCandidates(candidates)).toBe(false);
  });

  it("supports the expanded visual catalogue", () => {
    expect(recommendationCandidateKinds).toContain("video");
    expect(recommendationCandidateKinds).toContain("newsletter");
    expect(recommendationCandidateKinds).toContain("download");
  });

  it("adds a signal coefficient and restores it after the signal is removed", () => {
    const candidates = cloneRecommendationCandidates();
    const withDevice = addRecommendationFeature(candidates, "device", 2401);
    const coefficient = withDevice[0]!.coefficients.device;
    expect(coefficient).toBeTypeOf("number");
    const restored = addRecommendationFeature(withDevice, "device", 9999);
    expect(restored[0]!.coefficients.device).toBe(coefficient);
    expect(
      validRecommendationCandidates(restored, [...defaultRecommendationFeatureIds, "device"]),
    ).toBe(true);
  });

  it("validates ordered feature selections", () => {
    expect(validRecommendationFeatureIds([])).toBe(true);
    expect(validRecommendationFeatureIds(["device", "recent_activity"])).toBe(true);
    expect(validRecommendationFeatureIds(["device", "device"])).toBe(false);
  });
});
