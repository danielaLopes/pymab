import {
  configurationsMatch,
  draftFromConfiguration,
  validateRunDraft,
  type RunConfiguration,
} from "./runConfiguration";

const active: RunConfiguration = {
  lessonId: "linucb",
  mode: "freePlay",
  parameter: 1,
  seed: 123,
};

describe("run configuration", () => {
  it("validates an exact draft without rounding it", () => {
    expect(validateRunDraft(draftFromConfiguration(active))).toEqual({
      configuration: active,
      errors: {},
    });
    expect(
      validateRunDraft({ ...draftFromConfiguration(active), parameter: "1.03" }).configuration,
    ).toBeNull();
  });

  it("requires safe integer seeds", () => {
    const result = validateRunDraft({
      ...draftFromConfiguration(active),
      seed: String(Number.MAX_SAFE_INTEGER + 1),
    });
    expect(result.errors.seed).toBe("Enter a safe whole number.");
  });

  it("compares a valid draft with the active worker configuration", () => {
    expect(configurationsMatch(draftFromConfiguration(active), active)).toBe(true);
    expect(
      configurationsMatch({ ...draftFromConfiguration(active), mode: "challenge" }, active),
    ).toBe(false);
  });
});
