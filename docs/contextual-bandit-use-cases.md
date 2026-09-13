# Contextual bandit use cases

This guide develops two practical uses for the contextual policies already available in PyMAB: a single recommendation slot and defensive verification for uncertain requests. Both can be explored in the Arcade, but the simulations are deliberately narrow. They show a decision loop that fits a contextual bandit instead of stretching the method to cover ranking, long-term planning or hard security policy.

## A quick fit check

A contextual bandit is a reasonable starting point when all of these statements are true:

1. One decision is made at a time.
2. The available actions are known before the decision.
3. Context is available before the action is selected.
4. Feedback mainly measures the immediate result of that action.
5. Only the selected action's result is observed.
6. Exploration is acceptable inside the action set.
7. The objective can be expressed as a reward with meaningful guardrails.

That last point matters. A good action set contains choices that are already safe and permitted. A bandit should not decide whether a mandatory control applies, invent actions that have not been reviewed, or explore actions with unacceptable consequences.

A different method is usually better when the problem involves a ranked page, a long sequence of dependent decisions, rewards that arrive weeks later, or constraints that must never be violated. Candidate alternatives include learning-to-rank, supervised prediction, reinforcement learning, constrained optimization and explicit rules.

## Scenario 1: one recommendation slot

### The decision

The Arcade chooses one of three items for one visible slot:

| Part | Arcade meaning | Possible production meaning |
| --- | --- | --- |
| Context | Visitor, engagement and visit signals | Features available before rendering |
| Actions | Article, Product or Tutorial | Eligible candidates for one slot |
| Reward | Click or no click | Immediate engagement with the shown item |
| Policy | `LogisticContextualBanditPolicy` | One logistic response model per action |

The lesson encodes an intercept plus three binary features. It repeats the same feature row for each candidate because the context describes the current visitor and visit. The policy keeps a separate coefficient vector for each action, so the same visitor can receive a different predicted click probability for Article, Product and Tutorial.

This is a natural fit because there is one choice, a small eligible action set and quick binary feedback. It is not presented as a complete recommendation platform.

### What the simulation teaches

Before every decision, the history board shows the current context and the policy's predicted click probability for all three items. After the selection, only the chosen cell reveals Click or No click. The other outcomes remain unknown. This is the central partial-feedback constraint in a bandit problem.

The policy uses epsilon-greedy exploration over its logistic estimates:

```python
from pymab.policies import LogisticContextualBanditPolicy

policy = LogisticContextualBanditPolicy(
    n_arms=3,
    n_features=4,
    epsilon=0.08,
    learning_rate=0.18,
    l2=0.01,
)

action = policy.select_action(context=context, rng=rng)
policy.update(action=action, reward=float(clicked), context=context)
```

Only the selected action's model receives the gradient update. Exploration is therefore not decorative. It creates observations for candidates that an early estimate might otherwise exclude.

### What this scenario does not solve

Do not reuse this setup unchanged for any of the following:

- Ranking several items on the same page. Position effects and interactions between items require a ranking or slate model.
- Optimizing a purchase that may happen days later. Attribution and delayed feedback need additional machinery.
- Optimizing session length or retention. Those are sequential and long-term outcomes, not immediate one-step rewards.
- Selecting from millions of changing items. Candidate generation and representation learning must narrow the action set first.
- Treating a click as user satisfaction. Clicks are only a proxy and can reward sensational or repetitive content.

### A useful synthetic environment

The default Arcade environment defines one hidden coefficient vector for each item. It samples a visitor context, calculates the true click probability for each candidate with a logistic function, and samples click feedback only for the selected candidate.

Free Play exposes the coefficient matrix. Changing it lets a learner test whether the policy adapts when one candidate responds strongly to returning visitors or weekend traffic. The environment remains stationary within a run. A production system with changing inventory or preferences should also test discounted, sliding-window or change-detection approaches.

### Baselines and experiment matrix

At minimum, compare the contextual policy with:

| Baseline | Question it answers |
| --- | --- |
| Uniform random | Does learning beat an uninformed choice? |
| Best global item | Does context add value beyond one popular default? |
| Non-contextual epsilon-greedy | Does personalization improve reward enough to justify complexity? |
| Logistic contextual policy with no exploration | How much value comes from deliberate exploration? |

Vary click sparsity, feature strength, exploration rate, learning rate, regularization and context frequency. Report cumulative reward and expected regret, but also show per-context performance. A policy can look healthy overall while performing poorly for a less frequent group.

### Offline evaluation

Production evaluation requires logged action propensities. For each event, store the context available at decision time, the eligible action set, the chosen action, the probability of choosing it and the observed reward. Never reconstruct propensities later from a newer policy version.

PyMAB's `LoggedBanditDataset` can hold this information. Depending on the data and target policy, use:

- Sequential replay for an intuitive accepted-event evaluation.
- Inverse propensity scoring (IPS) when support is adequate and propensities are trustworthy.
- Self-normalized IPS (SNIPS) to reduce some variance.
- Doubly robust (DR) estimation when a defensible reward model is available.

Check overlap before trusting any estimate. If the logging policy almost never selected Tutorial for new visitors, offline data cannot reliably estimate a target policy that often makes that choice. Report effective sample size and uncertainty, not just a point estimate.

### Rollout and monitoring

Start with simulation tests, then shadow scoring, then a small randomized deployment. Set rollout gates for reward, latency, errors and group-level behavior. Keep a stable holdout and a kill switch. Monitor action share, propensity distribution, feature drift, reward delay, missing feedback and concentration on a single item. Roll back when guardrails fail even if aggregate clicks improve.

## Scenario 2: defensive verification

### The decision

The Arcade assumes an existing risk model and explicit security policy. Requests with clearly low risk can follow a deterministic fast path. Requests that meet hard blocking rules remain blocked. The bandit operates only in the uncertain middle range and selects among already approved actions.

| Part | Arcade meaning | Possible production meaning |
| --- | --- | --- |
| Context | Risk score, account age and endpoint sensitivity | Signals known before the response |
| Actions | Allow, Light check or Strong verification | Approved responses for the review band |
| Reward | A scalar utility from the observed outcome | Protection benefit minus user friction and failures |
| Policy | `LinUCBPolicy` | Linear value estimate plus uncertainty bonus |

The risk score is an input. LinUCB is not the detector. This separation prevents the example from confusing supervised risk prediction with action selection.

### The utility model

The synthetic environment samples whether a request is abusive from the configured risk relationship. It then samples an outcome using fixed behavior for each action. Example outcomes include Passed with no check, Passed light check, Abuse stopped, Abuse missed and Legitimate user abandoned.

Each outcome has a utility. The defaults reward stopping abuse and allowing legitimate users through with low friction. They penalize allowed or missed abuse and legitimate-user abandonment. Strong verification catches more abusive requests, but it causes more legitimate-user friction. The expected utility of each action therefore changes with request context.

The exact numbers are teaching assumptions, not universal business values. A real deployment would need product, security, legal, accessibility and support input before assigning any utility.

### Why LinUCB is a reasonable teaching policy

LinUCB displays the tradeoff clearly. For each approved action, it adds a confidence bonus to its predicted utility. An uncertain action can be selected because learning about it has value, but only inside the pre-approved decision band.

```python
from pymab.policies import LinUCBPolicy

policy = LinUCBPolicy(n_arms=3, n_features=4, alpha=0.75, l2=1.0)

action = policy.select_action(context=context, rng=rng)
policy.update(action=action, reward=observed_utility, context=context)
```

The Arcade history keeps the context, prediction, uncertainty bonus, total UCB score, selected action, outcome and utility for every round. This makes it possible to see why a successful outcome can still be followed by a lower UCB score: the reward can raise the estimate while the new observation reduces uncertainty by a larger amount.

### Safety boundary

This example is defensive. It does not optimize bot impersonation, mouse movement, scrolling patterns or detection evasion. Those objectives would help abusive automation and are not part of the scenario.

The bandit must not override hard controls, statutory checks, account recovery protections or decisions that require deterministic treatment. It should not explore Strong verification if that action has not passed accessibility and abandonment review. It should not learn from protected attributes or hidden proxies without a fairness assessment.

An ordinary stationary LinUCB policy may also be a poor fit when attackers react strategically or traffic changes abruptly. Adversarial or nonstationary methods may model the learning problem better, but they do not remove the need for fixed guardrails.

### Baselines and experiment matrix

Compare at least these strategies:

| Baseline | Question it answers |
| --- | --- |
| Always allow in the review band | What protection does verification add? |
| Fixed risk thresholds | Does learning improve on a clear rules-only policy? |
| Always use a light check | Does context justify different treatment? |
| Supervised outcome model plus fixed utility rule | Is a bandit needed for action selection? |

Vary abuse prevalence, risk-model calibration, action catch rates, abandonment rates, utility weights, context drift and abrupt attack changes. Report allowed abuse, stopped abuse, legitimate passage, abandonment, action share and utility separately. A single utility number can hide unacceptable movement in a safety metric.

### Offline evaluation

Logged data must include the decision-time risk score and features, the eligible action set, the selected action, its propensity and the observed outcome. Deterministic historical thresholds often create poor overlap. For example, if high-risk requests always received Strong verification, logged data cannot estimate what Allow would have done for that group without strong modeling assumptions.

Use `LoggedBanditDataset` for a consistent event format. IPS and SNIPS are suitable only where the logging policy has support. DR may improve efficiency when the outcome model is credible, but it is not a cure for missing support. Sequential replay is useful for validating update behavior. None of these estimators makes unobserved security outcomes harmless to infer.

### Rollout gates and rollback

A cautious path is simulation, historical replay, shadow scoring, staff or test-account traffic, then a tightly bounded live experiment. Keep hard rules outside the experiment. Define maximum allowed abuse, abandonment and challenge-failure rates before launch. Segment monitoring by endpoint, account cohort, geography where lawful, accessibility needs and risk band. Stop or roll back automatically when any hard guardrail is crossed.

Monitor policy concentration, uncertainty, context drift, reward completeness and changes in the upstream detector. A detector update changes the meaning of the context and can invalidate learned coefficients even when the policy code has not changed.

## Current PyMAB coverage and gaps

The current public API is enough to demonstrate both one-step loops:

- `LogisticContextualBanditPolicy` supports per-action logistic prediction, epsilon exploration and binary updates.
- `LinUCBPolicy` exposes contextual upper confidence bounds and real-valued reward updates.
- `LoggedBanditDataset`, replay and IPS, SNIPS and DR estimators support offline experimentation.
- Seeded random generators make Arcade runs reproducible.

A production system would still need infrastructure outside PyMAB: candidate eligibility, feature serving, policy versioning, propensity logging, delayed-feedback joins, privacy controls, experimentation, monitoring, rollback and audit records. Large or changing action sets, slate recommendation, constrained policies and strategic adversaries may also require algorithms that are not represented by these two lessons.

## Privacy, fairness and operational review

Collect only context needed for the stated decision. Define retention and access controls for event logs. Avoid raw personal data when a coarse, justified feature will do. Document which features are excluded and why.

Measure outcomes by relevant groups where lawful and technically meaningful. Check both reward and exposure. Recommendation exploration can distribute low-quality experiences unevenly. Defensive verification can impose extra friction on particular communities or devices. Aggregate utility does not excuse either pattern.

Finally, keep an audit trail that connects a decision to the policy version, parameters, action set, context schema and reward definition used at the time. Reproducibility is part of safety, not just a debugging convenience.
