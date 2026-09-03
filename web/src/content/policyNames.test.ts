import { readablePolicyClassName } from "./policyNames";

describe("readablePolicyClassName", () => {
  it.each([
    ["LogisticContextualBanditPolicy", "Logistic Contextual Bandit Policy"],
    ["BernoulliBayesianUCBPolicy", "Bernoulli Bayesian UCB Policy"],
    ["KLUCBPolicy", "KL UCB Policy"],
    ["CUSUMUCBPolicy", "CUSUM UCB Policy"],
    ["LinUCBPolicy", "Lin UCB Policy"],
    ["EXP3Policy", "EXP3 Policy"],
  ])("formats %s", (className, expected) => {
    expect(readablePolicyClassName(className)).toBe(expected);
  });
});
