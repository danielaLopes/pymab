import { familyDefinitions, policyCatalog, type PolicyId } from "@/catalog/policies";
import { readablePolicyClassName } from "./policyNames";

export const lessonContent = Object.fromEntries(
  Object.values(policyCatalog).map((policy) => [
    policy.id,
    {
      eyebrow: `${familyDefinitions[policy.family].label} · ${readablePolicyClassName(policy.className)}`,
      title: policy.title,
      intro: policy.intro,
      guidedSeed: policy.guidedSeed,
      challengeSeed: policy.challengeSeed,
      target: policy.target,
    },
  ]),
) as Record<
  PolicyId,
  {
    eyebrow: string;
    title: string;
    intro: string;
    guidedSeed: number;
    challengeSeed: number;
    target: string;
  }
>;

const familyMoments: Record<
  string,
  { initial: string; decision: string; update: string; tradeoff: string; repeat: string }
> = {
  foundations: {
    initial: "starts with the same amount of evidence for each portal.",
    decision: "made a choice from its current reward estimates and exploration rule.",
    update: "updated only the selected portal with the observed reward.",
    tradeoff: "now shows how repeated choices can concentrate attention or keep exploration alive.",
    repeat: "used the latest stationary reward evidence for this choice.",
  },
  optimism: {
    initial: "starts with uncertain confidence values for all three portals.",
    decision: "combined a reward estimate with a confidence bound before choosing.",
    update: "tightened its evidence for the selected portal after seeing the reward.",
    tradeoff: "is balancing high estimates against portals that remain uncertain.",
    repeat: "recalculated its confidence index from the evidence collected so far.",
  },
  bayesian: {
    initial: "begins with a prior belief for each portal.",
    decision: "used its current posterior beliefs to compare the portals.",
    update: "changed the selected portal's posterior after the new reward.",
    tradeoff: "now has tighter beliefs for familiar portals and wider uncertainty elsewhere.",
    repeat: "made this decision from the latest posterior state.",
  },
  changing: {
    initial: "starts with no recent evidence about the current reward phase.",
    decision: "used the evidence that its memory or detector currently retains.",
    update: "added the selected reward and adjusted its recent-state statistics.",
    tradeoff: "must keep enough history to learn while still reacting when rewards move.",
    repeat: "used its current window, discount, or change-detection state.",
  },
  "best-arm": {
    initial: "starts with all three portals as possible winners.",
    decision: "sampled a portal that still needs evidence.",
    update: "used this sample to refine its current best-portal recommendation.",
    tradeoff: "spends its budget separating close candidates rather than maximizing reward now.",
    repeat: "collected another sample for the final recommendation.",
  },
  adversarial: {
    initial: "starts with equal sampling weight on every portal.",
    decision: "sampled from its current EXP3 action probabilities.",
    update: "changed the selected portal's weight using its reward and selection probability.",
    tradeoff: "keeps every portal reachable because the reward pattern can be hostile.",
    repeat: "sampled again from the updated EXP3 distribution.",
  },
  contextual: {
    initial: "starts without learned coefficients for the current signals.",
    decision: "compared the portals using Light, Echo, Tide, and its current uncertainty.",
    update: "updated its model with the selected portal's signals and reward.",
    tradeoff: "can favor a different portal when the signal pattern changes.",
    repeat: "scored this round from the current signals and learned model.",
  },
};

const catalogExplanationCopy = Object.fromEntries(
  Object.values(policyCatalog).flatMap((policy) => {
    const moments = familyMoments[policy.family]!;
    return (Object.keys(moments) as Array<keyof typeof moments>).map((moment) => [
      `${policy.id}.${moment}`,
      `${policy.label} ${moments[moment]}`,
    ]);
  }),
);

export const explanationCopy: Record<string, string> = {
  ready: "The policy is ready. Advance to let PyMAB choose a portal.",
  "epsilon.explore": "Exploration: the ε draw selected a random portal.",
  "epsilon.exploit": "Exploitation: PyMAB chose among the portals with the highest estimate.",
  "epsilon.firstObservation":
    "One reward is evidence, not certainty. The selected portal's estimate moved toward the observed result.",
  "epsilon.firstExploration":
    "The ε draw tested a random portal instead of following the current estimate.",
  "epsilon.estimateUpdate":
    "Only the opened portal learned from this reward. The other estimates stayed unchanged.",
  "epsilon.cumulativeRegret":
    "Cumulative expected regret totals the expected reward forgone by every choice in this run.",
  "linucb.decision":
    "LinUCB combined its predicted reward with an uncertainty bonus for these signals.",
  "linucb.initialUncertainty":
    "With no evidence yet, every portal receives the same optimism bonus.",
  "linucb.contextPrediction":
    "The same learned coefficients produce new scores when Light, Echo, and Tide change.",
  "linucb.confidenceBonus":
    "Alpha scales how strongly LinUCB values evidence it has not gathered yet.",
  "linucb.update": "Only the chosen portal's coefficient vector and confidence matrix changed.",
  "linucb.changedContext":
    "A different signal pattern can recommend a different portal without navigation state.",
  "foundations.decision":
    "The policy updated its stationary reward evidence after observing the chosen portal.",
  "optimism.decision":
    "The policy balanced its reward estimate with a confidence value for uncertain portals.",
  "bayesian.decision": "The observed reward changed the selected portal's posterior belief.",
  "changing.decision":
    "The policy updated the recent evidence it uses to follow a changing environment.",
  "best-arm.decision": "This sample helps the policy decide which portals can still be the best.",
  "adversarial.decision":
    "EXP3 updated the selected portal's weight using its sampling probability and reward.",
  "contextual.decision":
    "The selected reward updated the model for the current Light, Echo, and Tide signals.",
  ...catalogExplanationCopy,
};
