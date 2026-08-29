import {
  configurationsMatch,
  defaultConfiguration,
  draftFromConfiguration,
  validateRunDraft,
} from "./runConfiguration";

describe("run configuration", () => {
  it("validates a complete parameter record without rounding it", () => {
    const active = { ...defaultConfiguration("linucb", "freePlay"), seed: 123 };
    const draft = draftFromConfiguration(active);
    expect(validateRunDraft(draft).configuration).toEqual({
      ...active,
      environment: validateRunDraft(draft).configuration?.environment,
    });
    expect(
      validateRunDraft({ ...draft, parameters: { ...draft.parameters, alpha: "1.05" } })
        .configuration,
    ).not.toBeNull();
  });

  it("requires safe integer seeds", () => {
    const draft = draftFromConfiguration(defaultConfiguration("epsilon-greedy", "freePlay"));
    const result = validateRunDraft({ ...draft, seed: String(Number.MAX_SAFE_INTEGER + 1) });
    expect(result.errors.seed).toBe("Enter a safe whole number.");
  });

  it("compares a valid draft with its active worker configuration", () => {
    const draft = draftFromConfiguration(defaultConfiguration("ucb"));
    const active = validateRunDraft(draft).configuration;
    expect(active).not.toBeNull();
    expect(configurationsMatch(draft, active)).toBe(true);
    expect(configurationsMatch({ ...draft, mode: "challenge" }, active)).toBe(false);
  });

  it("validates free-play probability environments", () => {
    const draft = draftFromConfiguration(defaultConfiguration("epsilon-greedy", "freePlay"));
    const valid = {
      ...draft,
      environment: {
        kind: "stationary-bernoulli" as const,
        probabilities: ["12.3", "45.6", "78.9"] as [string, string, string],
      },
    };
    expect(validateRunDraft(valid).configuration?.environment).toEqual({
      probabilities: [0.123, 0.456, 0.789],
    });
    valid.environment.probabilities[0] = "100.1";
    expect(validateRunDraft(valid).configuration).toBeNull();
  });
});
