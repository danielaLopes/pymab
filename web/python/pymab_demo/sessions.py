"""Stateful, deterministic lesson sessions backed by real PyMAB policies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
from statistics import NormalDist
from typing import Any, cast

import numpy as np

import pymab
from pymab._random import generator
from pymab.policies import EpsilonGreedyPolicy, LinUCBPolicy
from pymab.policies.policy import ContextualPolicy, Policy
from pymab_demo.catalog import POLICY_CATALOG
from pymab_demo.diagnostics import epsilon_decision, linucb_decision
from pymab_demo.fixtures import (
    CUE_NAMES,
    EPSILON_MEANS,
    FIXTURES,
    GATE_IDS,
    LINUCB_THETA,
    LessonFixture,
    LessonId,
    Mode,
    horizon_for,
    validate_environment,
    validate_parameters,
)


class LessonSession(ABC):
    """Common lifecycle and snapshot contract for a single expedition."""

    def __init__(
        self,
        *,
        session_id: str,
        lesson_id: LessonId,
        mode: Mode,
        seed: int,
        parameters: dict[str, object],
        source_commit: str,
        environment: dict[str, object] | None = None,
    ) -> None:
        self.session_id = session_id
        self.lesson_id = lesson_id
        self.mode = mode
        self.seed = seed
        self.parameters = validate_parameters(lesson_id, parameters)
        self.environment = validate_environment(lesson_id, mode, environment)
        self.source_commit = source_commit
        self.horizon = horizon_for(lesson_id, mode)
        self.history: list[dict[str, Any]] = []
        self.total_reward = 0.0
        self.cumulative_regret = 0.0
        self.disposed = False
        self._initialize()

    @abstractmethod
    def _initialize(self) -> None:
        """Recreate policy and deterministic random streams."""

    @abstractmethod
    def _perform_step(self) -> dict[str, Any]:
        """Execute one policy/environment interaction."""

    def step(self) -> dict[str, Any]:
        """Advance once and return a complete presentation snapshot."""

        if self.disposed:
            raise RuntimeError("session has been disposed")
        if len(self.history) >= self.horizon:
            return self.snapshot()
        event = self._perform_step()
        self.total_reward += float(event["reward"])
        self.cumulative_regret += float(event["instantaneousExpectedRegret"])
        self.history.append(event)
        return self.snapshot()

    def run_to_end(self) -> dict[str, Any]:
        """Advance to the fixed horizon."""

        while len(self.history) < self.horizon:
            self.step()
        return self.snapshot()

    def reset(self) -> dict[str, Any]:
        """Reconstruct every mutable object from the configured seed."""

        self.history = []
        self.total_reward = 0.0
        self.cumulative_regret = 0.0
        self.disposed = False
        self._initialize()
        return self.snapshot()

    def dispose(self) -> None:
        """Make the session reject future work."""

        self.disposed = True

    def snapshot(self) -> dict[str, Any]:
        """Return current state without exposing challenge truth early."""

        complete = len(self.history) >= self.horizon
        last = self.history[-1] if self.history else None
        fixture = FIXTURES.get(self.lesson_id)
        passed = self._passed(complete, fixture)
        spec = POLICY_CATALOG[self.lesson_id]
        return {
            "policyId": self.lesson_id,
            "family": spec.family,
            "objective": spec.objective,
            "mode": self.mode,
            "seed": self.seed,
            "packageVersion": pymab.__version__,
            "sourceCommit": self.source_commit,
            "sessionId": self.session_id,
            "step": len(self.history),
            "horizon": self.horizon,
            "parameters": self.parameters,
            "environment": self._public_environment(),
            "gateIds": GATE_IDS,
            "selectedArm": None if last is None else last["selectedArm"],
            "reward": None if last is None else last["reward"],
            "totalReward": self.total_reward,
            "instantaneousExpectedRegret": None
            if last is None
            else last["instantaneousExpectedRegret"],
            "cumulativeExpectedRegret": self.cumulative_regret,
            "completed": complete,
            "passed": passed,
            "visibleCues": [] if last is None else last["visibleCues"],
            "publicContext": None if last is None else last["publicContext"],
            "explanationKey": "ready" if last is None else last["explanationKey"],
            "diagnostic": None if last is None else last["diagnostic"],
            "recommendation": self._recommendation() if complete else None,
            "history": self.history,
            "hiddenTruth": self._hidden_truth() if complete else None,
            "generatedCode": self.generated_code(),
        }

    def _passed(self, complete: bool, fixture: LessonFixture | None) -> bool:
        if not complete:
            return False
        if fixture is not None:
            return bool(
                self.total_reward >= fixture.reward_target
                and self.cumulative_regret <= fixture.regret_target
            )
        return self.cumulative_regret <= self.horizon * 0.35

    def _recommendation(self) -> int | None:
        return None

    @abstractmethod
    def _hidden_truth(self) -> dict[str, Any]:
        """Return environment truth for completed-run debriefs."""

    def _public_environment(self) -> dict[str, Any] | None:
        """Return environment values that are intentionally visible during a run."""

        return None

    @abstractmethod
    def generated_code(self) -> str:
        """Create an equivalent public-API example."""


class EpsilonLessonSession(LessonSession):
    """Three Bernoulli gates taught with ``EpsilonGreedyPolicy``."""

    def _initialize(self) -> None:
        self.probabilities = self.environment.get("probabilities", EPSILON_MEANS)
        self.policy = EpsilonGreedyPolicy(
            n_arms=3,
            epsilon=self.parameters["epsilon"],
            initial_value=self.parameters["initial_value"],
        )
        self.action_rng = generator(self.seed, "epsilon-greedy", "lesson", "action")
        self.reward_rng = generator(self.seed, "epsilon-greedy", "lesson", "reward")

    def _perform_step(self) -> dict[str, Any]:
        action, diagnostic = epsilon_decision(self.policy, self.action_rng)
        potential = (self.reward_rng.random(3) < np.asarray(self.probabilities)).astype(
            int
        )
        reward = int(potential[action])
        regret = float(max(self.probabilities) - self.probabilities[action])
        self.policy.update(action=action, reward=float(reward))
        diagnostic.update(
            {
                "countsAfter": self.policy.counts.copy(),
                "estimatesAfter": self.policy.estimates.copy(),
            }
        )
        step_number = len(self.history) + 1
        if self.mode == "guided" and step_number == 1:
            explanation_key = "epsilon.firstObservation"
        elif (
            self.mode == "guided"
            and diagnostic["selectionBranch"] == "explore"
            and not any(
                event["diagnostic"]["selectionBranch"] == "explore"
                for event in self.history
            )
        ):
            explanation_key = "epsilon.firstExploration"
        elif self.mode == "guided" and step_number == self.horizon:
            explanation_key = "epsilon.cumulativeRegret"
        elif self.mode == "guided" and step_number == 4:
            explanation_key = "epsilon.estimateUpdate"
        else:
            explanation_key = f"epsilon.{diagnostic['selectionBranch']}"
        return {
            "selectedArm": action,
            "reward": reward,
            "instantaneousExpectedRegret": regret,
            "visibleCues": [],
            "publicContext": None,
            "explanationKey": explanation_key,
            "diagnostic": diagnostic,
        }

    def _hidden_truth(self) -> dict[str, Any]:
        return {
            "probabilities": self.probabilities,
            "optimalArm": int(np.argmax(self.probabilities)),
        }

    def _public_environment(self) -> dict[str, Any] | None:
        if self.mode != "freePlay":
            return None
        return {"probabilities": self.probabilities}

    def generated_code(self) -> str:
        from pymab_demo.codegen import epsilon_example

        return epsilon_example(
            seed=self.seed,
            epsilon=self.parameters["epsilon"],
            horizon=self.horizon,
            probabilities=self.probabilities,
            initial_value=self.parameters["initial_value"],
        )


class LinUCBLessonSession(LessonSession):
    """Independent contextual chambers taught with disjoint LinUCB."""

    def _initialize(self) -> None:
        self.policy = LinUCBPolicy(
            n_arms=3,
            n_features=4,
            alpha=self.parameters["alpha"],
            l2=self.parameters["l2"],
        )
        self.context_rng = generator(self.seed, "arcade", 1, "context")
        self.action_rng = generator(self.seed, "arcade", 1, "action")
        self.reward_rng = generator(self.seed, "arcade", 1, "reward")
        self.theta = np.asarray(
            self.environment.get("theta", LINUCB_THETA), dtype=float
        )
        self.truth_history: list[tuple[np.ndarray, int]] = []

    def _perform_step(self) -> dict[str, Any]:
        cue_values = self.context_rng.choice(np.asarray([-1.0, 1.0]), size=3)
        feature = np.concatenate((np.ones(1), cue_values))
        context = np.repeat(feature[np.newaxis, :], 3, axis=0)
        probabilities = 1.0 / (1.0 + np.exp(-(self.theta @ feature)))
        potential = (self.reward_rng.random(3) < probabilities).astype(int)
        action, diagnostic = linucb_decision(self.policy, context, self.action_rng)
        reward = int(potential[action])
        optimal = int(np.argmax(probabilities))
        self.truth_history.append((probabilities.copy(), optimal))
        regret = float(probabilities[optimal] - probabilities[action])
        self.policy.update(action=action, reward=float(reward), context=context)
        diagnostic.update(
            {"aAfter": self.policy.a.copy(), "bAfter": self.policy.b.copy()}
        )
        cues = [
            {
                "name": name,
                "value": int(value),
                "label": self._cue_label(name, int(value)),
            }
            for name, value in zip(CUE_NAMES, cue_values, strict=True)
        ]
        step_number = len(self.history) + 1
        guided_keys = {
            1: "linucb.initialUncertainty",
            2: "linucb.contextPrediction",
            3: "linucb.confidenceBonus",
            4: "linucb.update",
            6: "linucb.changedContext",
        }
        return {
            "selectedArm": action,
            "reward": reward,
            "instantaneousExpectedRegret": regret,
            "visibleCues": cues,
            "publicContext": context,
            "explanationKey": (
                guided_keys.get(step_number, "linucb.decision")
                if self.mode == "guided"
                else "linucb.decision"
            ),
            "diagnostic": diagnostic,
        }

    @staticmethod
    def _cue_label(name: str, value: int) -> str:
        labels = {
            "light": ("red light", "blue light"),
            "echo": ("low echo", "high echo"),
            "tide": ("low tide", "high tide"),
        }
        return labels[name][1 if value == 1 else 0]

    def _hidden_truth(self) -> dict[str, Any]:
        probabilities = [item[0] for item in self.truth_history]
        optimal = [item[1] for item in self.truth_history]
        return {
            "theta": self.theta,
            "probabilities": probabilities,
            "optimalArms": optimal,
        }

    def generated_code(self) -> str:
        from pymab_demo.codegen import linucb_example

        return linucb_example(
            seed=self.seed,
            alpha=self.parameters["alpha"],
            l2=self.parameters["l2"],
            horizon=self.horizon,
            theta=self.theta,
        )

    def _public_environment(self) -> dict[str, Any] | None:
        if self.mode != "freePlay":
            return None
        return {"theta": self.theta}


class CatalogPolicySession(LessonSession):
    """Catalog-driven lesson session for every additional public policy."""

    def _initialize(self) -> None:
        self.spec = POLICY_CATALOG[self.lesson_id]
        self.policy = self.spec.create(self.parameters, horizon=self.horizon)
        self.action_rng = generator(self.seed, self.lesson_id, "arcade", "action")
        self.reward_rng = generator(self.seed, self.lesson_id, "arcade", "reward")
        self.context_rng = generator(self.seed, self.lesson_id, "arcade", "context")
        self._truth_history: list[dict[str, Any]] = []

    @staticmethod
    def _cue_label(name: str, value: int) -> str:
        labels = {
            "light": ("red light", "blue light"),
            "echo": ("low echo", "high echo"),
            "tide": ("low tide", "high tide"),
        }
        return labels[name][1 if value == 1 else 0]

    def _context(self) -> tuple[np.ndarray, list[dict[str, Any]]]:
        cue_values = self.context_rng.choice(np.asarray([-1.0, 1.0]), size=3)
        feature = np.concatenate((np.ones(1), cue_values))
        context = np.repeat(feature[np.newaxis, :], 3, axis=0)
        cues = [
            {
                "name": name,
                "value": int(value),
                "label": self._cue_label(name, int(value)),
            }
            for name, value in zip(CUE_NAMES, cue_values, strict=True)
        ]
        return context, cues

    def _environment_round(
        self,
    ) -> tuple[
        np.ndarray, np.ndarray, np.ndarray | None, list[dict[str, Any]], dict[str, Any]
    ]:
        step = len(self.history)
        kind = self.spec.environment
        context: np.ndarray | None = None
        cues: list[dict[str, Any]] = []
        details: dict[str, Any] = {"environmentKind": kind}

        if kind in {"stationary-bernoulli", "best-arm"}:
            probabilities = np.asarray(
                self.environment.get("probabilities", EPSILON_MEANS), dtype=float
            )
            rewards = (self.reward_rng.random(3) < probabilities).astype(float)
            expected = probabilities
            details["probabilities"] = probabilities
        elif kind == "stationary-gaussian":
            means = np.asarray(
                self.environment.get("means", (-0.25, 0.3, 0.85)), dtype=float
            )
            standard_deviation = float(self.environment.get("standardDeviation", 0.5))
            rewards = self.reward_rng.normal(means, standard_deviation)
            expected = means
            details.update({"means": means, "standardDeviation": standard_deviation})
        elif kind == "changing-bernoulli":
            phases = self.environment.get(
                "phases",
                (
                    {"start": 0, "probabilities": (0.75, 0.5, 0.25)},
                    {
                        "start": max(1, self.horizon // 3),
                        "probabilities": (0.2, 0.75, 0.45),
                    },
                    {
                        "start": max(2, (2 * self.horizon) // 3),
                        "probabilities": (0.45, 0.2, 0.8),
                    },
                ),
            )
            active_phase = max(
                (phase for phase in phases if int(phase["start"]) <= step),
                key=lambda phase: int(phase["start"]),
            )
            probabilities = np.asarray(active_phase["probabilities"], dtype=float)
            rewards = (self.reward_rng.random(3) < probabilities).astype(float)
            expected = probabilities
            details.update(
                {
                    "phaseStart": int(active_phase["start"]),
                    "probabilities": probabilities,
                    "phases": phases,
                }
            )
        elif kind == "adversarial":
            configured = self.environment.get("rewards")
            if configured is not None:
                reward_matrix = np.asarray(configured, dtype=float)
                rewards = reward_matrix[step]
            else:
                leader = (step // 3 + self.seed % 3) % 3
                rewards = np.full(3, 0.1, dtype=float)
                rewards[leader] = 1.0
                rewards[(leader + 1) % 3] = 0.4
            expected = rewards.copy()
            details["rewardVector"] = rewards
        else:
            context, cues = self._context()
            theta = np.asarray(self.environment.get("theta", LINUCB_THETA), dtype=float)
            linear_means = theta @ context[0]
            if kind == "contextual-logistic":
                probabilities = 1.0 / (1.0 + np.exp(-linear_means))
                rewards = (self.reward_rng.random(3) < probabilities).astype(float)
                expected = probabilities
                details["probabilities"] = probabilities
            else:
                noise = float(self.environment.get("standardDeviation", 0.2))
                rewards = self.reward_rng.normal(linear_means, noise)
                expected = linear_means
                details["standardDeviation"] = noise
            details.update({"theta": theta, "expectedRewards": expected})

        return rewards, expected, context, cues, details

    @staticmethod
    def _array(value: object) -> object:
        if isinstance(value, np.ndarray):
            return value.copy()
        return value

    def _policy_state(self, policy: object | None = None) -> dict[str, Any]:
        inspected = self.policy if policy is None else policy
        state: dict[str, Any] = {"policyClass": type(inspected).__name__}
        attributes = (
            "step",
            "counts",
            "estimates",
            "epsilon",
            "preferences",
            "probabilities",
            "average_reward",
            "successes",
            "failures",
            "means",
            "precisions",
            "weights",
            "log_weights",
            "discounted_counts",
            "discounted_sums",
            "active",
            "phase_counts",
            "phase_epsilon",
            "phase_delta",
            "change_counts",
            "positive_cusum",
            "negative_cusum",
            "ph_cumulative",
            "theta",
            "a",
            "b",
        )
        for name in attributes:
            if hasattr(inspected, name):
                state[name] = self._array(getattr(inspected, name))
        action_probabilities = getattr(inspected, "action_probabilities", None)
        if callable(action_probabilities):
            state["actionProbabilities"] = action_probabilities()
        indices = getattr(inspected, "indices", None)
        counts = getattr(inspected, "counts", None)
        if callable(indices) and isinstance(counts, np.ndarray) and np.all(counts > 0):
            state["indices"] = indices()
        return state

    def _preview_decision(
        self, context: np.ndarray | None
    ) -> tuple[int, dict[str, Any]]:
        """Preview the real decision without consuming policy or RNG state."""

        probe_policy = deepcopy(self.policy)
        probe_rng = deepcopy(self.action_rng)
        if isinstance(probe_policy, ContextualPolicy):
            if context is None:
                raise RuntimeError("contextual policy did not receive a context")
            action = int(probe_policy.select_action(context=context, rng=probe_rng))
        elif isinstance(probe_policy, Policy):
            action = int(probe_policy.select_action(rng=probe_rng))
        else:  # pragma: no cover - catalog type invariant
            raise RuntimeError("catalog returned an unsupported policy object")
        return action, self._decision_presentation(context, probe_policy)

    def _decision_presentation(
        self, context: np.ndarray | None, probe_policy: object
    ) -> dict[str, Any]:
        """Return exact, labelled values used to explain the pending decision."""

        policy_id = self.lesson_id
        policy = self.policy
        typed_policy = cast(Any, policy)
        typed_probe = cast(Any, probe_policy)
        state = self._policy_state()
        decision: dict[str, Any] = {}

        def set_values(label: str, values: object) -> None:
            decision["label"] = label
            decision["values"] = self._array(values)

        if policy_id == "random":
            set_values("Action probability", np.full(3, 1.0 / 3.0))
        elif policy_id in {
            "greedy",
            "decaying-epsilon-greedy",
            "successive-elimination",
            "median-elimination",
        }:
            set_values("Estimate", state.get("estimates", np.zeros(3)))
        elif policy_id == "softmax":
            set_values("Action probability", state["actionProbabilities"])
        elif policy_id == "gradient-bandit":
            set_values("Action probability", typed_probe.probabilities)
            decision["secondaryLabel"] = "Preference"
            decision["secondaryValues"] = state["preferences"]
        elif policy_id in {
            "ucb",
            "kl-ucb",
            "moss",
            "sliding-window-ucb",
            "discounted-ucb",
            "change-point-ucb",
            "cusum-ucb",
            "page-hinkley-ucb",
        }:
            indices = state.get("indices")
            if indices is not None:
                set_values("Confidence index", indices)
            else:
                set_values("Estimate", state.get("estimates", np.zeros(3)))
                counts = np.asarray(state.get("counts", np.zeros(3)), dtype=float)
                decision["unseenArms"] = np.flatnonzero(counts == 0)
        elif policy_id in {
            "bernoulli-thompson-sampling",
            "sliding-window-bernoulli-thompson-sampling",
            "discounted-bernoulli-thompson-sampling",
        }:
            rng = deepcopy(self.action_rng)
            samples = rng.beta(
                float(typed_policy.alpha_prior) + np.asarray(state["successes"]),
                float(typed_policy.beta_prior) + np.asarray(state["failures"]),
            )
            set_values("Posterior sample", samples)
        elif policy_id == "gaussian-thompson-sampling":
            rng = deepcopy(self.action_rng)
            samples = rng.normal(
                np.asarray(state["means"]),
                1.0 / np.sqrt(np.asarray(state["precisions"])),
            )
            set_values("Posterior sample", samples)
        elif policy_id == "bernoulli-bayesian-ucb":
            from scipy.stats import beta as beta_distribution

            bounds = beta_distribution.ppf(
                float(typed_policy.quantile),
                float(typed_policy.alpha_prior) + np.asarray(state["successes"]),
                float(typed_policy.beta_prior) + np.asarray(state["failures"]),
            )
            set_values("Credible upper bound", bounds)
        elif policy_id == "gaussian-bayesian-ucb":
            z_value = NormalDist().inv_cdf(float(typed_policy.quantile))
            bounds = np.asarray(state["means"]) + z_value / np.sqrt(
                np.asarray(state["precisions"])
            )
            set_values("Credible upper bound", bounds)
        elif policy_id == "exp3":
            set_values("Action probability", state["actionProbabilities"])
        elif policy_id in {"linear-epsilon-greedy", "logistic-contextual-bandit"}:
            if context is None:
                raise RuntimeError("contextual decision presentation requires context")
            scores = np.einsum("ij,ij->i", context, np.asarray(state["theta"]))
            if policy_id == "logistic-contextual-bandit":
                set_values("Predicted reward chance", 1.0 / (1.0 + np.exp(-scores)))
            else:
                set_values("Predicted reward", scores)
        elif policy_id == "linear-thompson-sampling":
            if context is None:
                raise RuntimeError("contextual decision presentation requires context")
            rng = deepcopy(self.action_rng)
            samples = np.zeros(3, dtype=float)
            matrices = np.asarray(state["a"])
            vectors = np.asarray(state["b"])
            scale = float(typed_policy.exploration_scale)
            for arm in range(3):
                inverse = np.linalg.solve(matrices[arm], np.eye(context.shape[1]))
                mean = np.linalg.solve(matrices[arm], vectors[arm])
                theta_sample = rng.multivariate_normal(mean, (scale**2) * inverse)
                samples[arm] = float(theta_sample @ context[arm])
            set_values("Posterior sample", samples)

        if policy_id in {
            "decaying-epsilon-greedy",
            "linear-epsilon-greedy",
            "logistic-contextual-bandit",
        }:
            rng = deepcopy(self.action_rng)
            sampled = float(rng.random())
            epsilon = float(typed_policy.epsilon)
            decision.update(
                {
                    "epsilon": epsilon,
                    "selectionBranch": "explore" if sampled < epsilon else "exploit",
                }
            )

        if self.spec.environment == "changing-bernoulli":
            phases = self.environment.get(
                "phases",
                (
                    {"start": 0},
                    {"start": max(1, self.horizon // 3)},
                    {"start": max(2, (2 * self.horizon) // 3)},
                ),
            )
            step = len(self.history)
            active_index = max(
                index
                for index, phase in enumerate(phases)
                if int(phase["start"]) <= step
            )
            decision["environmentPhase"] = active_index + 1

        return decision

    def _recommendation(self) -> int | None:
        if self.spec.objective != "best-arm":
            return None
        if not isinstance(self.policy, Policy):
            raise RuntimeError("best-arm objective requires a non-contextual policy")
        return int(self.policy.recommend_action())

    def _perform_step(self) -> dict[str, Any]:
        rewards, expected, context, cues, truth = self._environment_round()
        before = self._policy_state()
        preview_action, decision = self._preview_decision(context)
        if isinstance(self.policy, ContextualPolicy):
            if context is None:
                raise RuntimeError("contextual policy did not receive a context")
            action = int(
                self.policy.select_action(context=context, rng=self.action_rng)
            )
            reward = float(rewards[action])
            self.policy.update(action=action, reward=reward, context=context)
        elif isinstance(self.policy, Policy):
            action = int(self.policy.select_action(rng=self.action_rng))
            reward = float(rewards[action])
            self.policy.update(action=action, reward=reward)
        else:  # pragma: no cover - catalog type invariant
            raise RuntimeError("catalog returned an unsupported policy object")
        if action != preview_action:
            raise RuntimeError("decision diagnostic diverged from policy selection")
        optimal = int(np.argmax(expected))
        regret = float(expected[optimal] - expected[action])
        after = self._policy_state()
        recommendation: int | None = None
        if self.spec.objective == "best-arm":
            recommendation = self._recommendation()
        truth.update({"expectedRewards": expected, "optimalArm": optimal})
        self._truth_history.append(truth)
        step_number = len(self.history) + 1
        if step_number == 1:
            explanation_key = f"{self.lesson_id}.initial"
        elif step_number == 2:
            explanation_key = f"{self.lesson_id}.decision"
        elif step_number == 3:
            explanation_key = f"{self.lesson_id}.update"
        elif step_number == max(4, self.horizon // 2):
            explanation_key = f"{self.lesson_id}.tradeoff"
        else:
            explanation_key = f"{self.lesson_id}.repeat"
        return {
            "selectedArm": action,
            "reward": reward,
            "instantaneousExpectedRegret": regret,
            "visibleCues": cues,
            "publicContext": context,
            "explanationKey": explanation_key,
            "diagnostic": {
                "before": before,
                "after": after,
                "decision": decision,
                "recommendation": recommendation,
                "contextMatrix": context,
            },
        }

    def _passed(self, complete: bool, fixture: object | None) -> bool:
        if not complete:
            return False
        if self.spec.objective == "best-arm":
            recommendation = self._recommendation()
            truth = self._truth_history[-1]
            return recommendation == int(truth["optimalArm"])
        return self.cumulative_regret <= self.horizon * 0.35

    def _hidden_truth(self) -> dict[str, Any]:
        result: dict[str, Any] = {"rounds": self._truth_history}
        if self._truth_history:
            result.update(self._truth_history[-1])
        if self.spec.objective == "best-arm":
            result["recommendation"] = self._recommendation()
        return result

    def _public_environment(self) -> dict[str, Any] | None:
        return self.environment or None if self.mode == "freePlay" else None

    def generated_code(self) -> str:
        from pymab_demo.codegen import catalog_example

        return catalog_example(
            policy_id=self.lesson_id,
            class_name=self.spec.policy_class.__name__,
            constructor=self.spec.constructor_parameters(
                self.parameters, horizon=self.horizon
            ),
            environment_kind=self.spec.environment,
            objective=self.spec.objective,
            seed=self.seed,
            horizon=self.horizon,
            environment=self.environment,
        )


def create_session(
    *,
    session_id: str,
    lesson_id: LessonId,
    mode: Mode,
    seed: int,
    parameters: dict[str, object],
    source_commit: str,
    environment: dict[str, object] | None = None,
) -> LessonSession:
    """Construct the correct concrete lesson session."""

    session_type: type[LessonSession]
    if lesson_id == "epsilon-greedy":
        session_type = EpsilonLessonSession
    elif lesson_id == "linucb":
        session_type = LinUCBLessonSession
    else:
        session_type = CatalogPolicySession
    return session_type(
        session_id=session_id,
        lesson_id=lesson_id,
        mode=mode,
        seed=seed,
        parameters=parameters,
        source_commit=source_commit,
        environment=environment,
    )
