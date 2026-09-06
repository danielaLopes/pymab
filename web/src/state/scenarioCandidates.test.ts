import {
  appendRecommendationCandidate,
  candidateNameErrors,
  cloneRecommendationCandidates,
  generateCandidateCoefficients,
  validRecommendationCandidates,
} from "./scenarioCandidates";

describe("recommendation candidates", () => {
  it("generates stable bounded coefficients", () => {
    const first = generateCandidateCoefficients(2401, "candidate-4");
    expect(first).toEqual(generateCandidateCoefficients(2401, "candidate-4"));
    expect(first).not.toEqual(generateCandidateCoefficients(2402, "candidate-4"));
    expect(first[0]).toBeGreaterThanOrEqual(-0.4);
    expect(first[0]).toBeLessThanOrEqual(0.4);
    expect(first.slice(1).every((value) => value >= -0.85 && value <= 0.85)).toBe(true);
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
});
