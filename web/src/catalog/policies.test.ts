import { familyOrder, policiesByFamily, policyCatalog, policyIds } from "./policies";

describe("policy catalog", () => {
  it("contains 27 unique concrete public policies", () => {
    expect(policyIds).toHaveLength(27);
    expect(new Set(policyIds)).toHaveLength(27);
    expect(Object.keys(policyCatalog)).toEqual([...policyIds]);
    expect(new Set(policyIds.map((id) => policyCatalog[id].className))).toHaveLength(27);
  });

  it("assigns every policy to exactly one ordered family", () => {
    const grouped = familyOrder.flatMap((family) =>
      policiesByFamily[family].map((item) => item.id),
    );
    expect(grouped).toHaveLength(27);
    expect(new Set(grouped)).toEqual(new Set(policyIds));
  });

  it("provides complete runnable metadata", () => {
    for (const id of policyIds) {
      const item = policyCatalog[id];
      expect(item.className.endsWith("Policy")).toBe(true);
      expect(item.title.length).toBeGreaterThan(0);
      expect(item.intro.length).toBeGreaterThan(0);
      expect(item.horizon).toBeGreaterThanOrEqual(3);
      expect(item.challengeHorizon).toBeGreaterThanOrEqual(item.horizon);
      expect(Object.keys(item.guidedParameters)).toEqual(Object.keys(item.defaults));
      expect(new Set(item.parameters.map((parameter) => parameter.key))).toEqual(
        new Set(Object.keys(item.defaults)),
      );
    }
  });
});
