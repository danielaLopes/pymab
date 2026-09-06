"""Applied contextual-bandit scenarios backed by public PyMAB policies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, Literal

import numpy as np

import pymab
from pymab._random import generator
from pymab.policies import LinUCBPolicy, LogisticContextualBanditPolicy
from pymab_demo.diagnostics import linucb_decision

ScenarioId = Literal["recommendations", "defensive-verification"]
Mode = Literal["guided", "challenge", "freePlay"]

SCENARIO_IDS: tuple[ScenarioId, ...] = (
    "recommendations",
    "defensive-verification",
)


def _as_float(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(
        value, (int, float, str, np.integer, np.floating)
    ):
        raise ValueError(f"{name} must be numeric")
    try:
        return float(value)
    except ValueError as error:
        raise ValueError(f"{name} must be numeric") from error


def _number(
    values: dict[str, object], key: str, default: float, *, minimum: float
) -> float:
    value = _as_float(values.get(key, default), name=key)
    if not np.isfinite(value) or value < minimum:
        raise ValueError(f"{key} must be at least {minimum}")
    return value


class ScenarioSession(ABC):
    """Common deterministic lifecycle for an applied scenario."""

    scenario_id: ScenarioId
    policy_id: str
    family = "contextual"
    objective = "cumulative-reward"

    def __init__(
        self,
        *,
        session_id: str,
        mode: Mode,
        seed: int,
        parameters: dict[str, object],
        source_commit: str,
        environment: dict[str, object] | None = None,
    ) -> None:
        self.session_id = session_id
        self.mode = mode
        self.seed = seed
        self.parameters = self.validate_parameters(parameters)
        self.environment = dict(environment or {})
        self.source_commit = source_commit
        self.horizon = 20 if mode == "challenge" else 12
        self.history: list[dict[str, Any]] = []
        self.total_reward = 0.0
        self.cumulative_regret = 0.0
        self.disposed = False
        self._initialize()

    @abstractmethod
    def validate_parameters(self, parameters: dict[str, object]) -> dict[str, object]:
        """Validate and normalize public policy parameters."""

    @abstractmethod
    def _initialize(self) -> None:
        """Recreate the policy and random streams."""

    @abstractmethod
    def _perform_step(self) -> dict[str, Any]:
        """Run one interaction."""

    @abstractmethod
    def presentation(self) -> dict[str, Any]:
        """Return safe labels and icon identifiers for the interface."""

    @abstractmethod
    def generated_code(self) -> str:
        """Return a compact equivalent public-API example."""

    @abstractmethod
    def _hidden_truth(self) -> dict[str, Any]:
        """Return the simulation truth after a run completes."""

    def step(self) -> dict[str, Any]:
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
        while len(self.history) < self.horizon:
            self.step()
        return self.snapshot()

    def reset(self) -> dict[str, Any]:
        self.history = []
        self.total_reward = 0.0
        self.cumulative_regret = 0.0
        self.disposed = False
        self._initialize()
        return self.snapshot()

    def dispose(self) -> None:
        self.disposed = True

    def snapshot(self) -> dict[str, Any]:
        complete = len(self.history) >= self.horizon
        last = self.history[-1] if self.history else None
        return {
            "policyId": self.policy_id,
            "scenarioId": self.scenario_id,
            "family": self.family,
            "objective": self.objective,
            "mode": self.mode,
            "seed": self.seed,
            "packageVersion": pymab.__version__,
            "sourceCommit": self.source_commit,
            "sessionId": self.session_id,
            "step": len(self.history),
            "horizon": self.horizon,
            "parameters": self.parameters,
            "environment": self.environment if self.mode == "freePlay" else None,
            "gateIds": [
                arm["shortName"].lower() for arm in self.presentation()["arms"]
            ],
            "presentation": self.presentation(),
            "selectedArm": None if last is None else last["selectedArm"],
            "reward": None if last is None else last["reward"],
            "totalReward": self.total_reward,
            "instantaneousExpectedRegret": (
                None if last is None else last["instantaneousExpectedRegret"]
            ),
            "cumulativeExpectedRegret": self.cumulative_regret,
            "completed": complete,
            "passed": complete and self.cumulative_regret <= self.horizon * 0.4,
            "visibleCues": [] if last is None else last["visibleCues"],
            "publicContext": None if last is None else last["publicContext"],
            "explanationKey": "ready" if last is None else last["explanationKey"],
            "diagnostic": None if last is None else last["diagnostic"],
            "recommendation": None,
            "history": self.history,
            "hiddenTruth": self._hidden_truth() if complete else None,
            "generatedCode": self.generated_code(),
        }


class RecommendationScenarioSession(ScenarioSession):
    """Choose one item for a single recommendation slot with immediate feedback."""

    scenario_id: ScenarioId = "recommendations"
    policy_id = "logistic-contextual-bandit"
    default_theta = np.asarray(
        [
            [0.10, -0.65, 0.20, 0.35],
            [-0.35, 0.85, 0.70, -0.15],
            [0.05, -0.75, -0.55, 0.45],
        ],
        dtype=float,
    )

    def validate_parameters(self, parameters: dict[str, object]) -> dict[str, object]:
        epsilon = _number(parameters, "epsilon", 0.08, minimum=0.0)
        if epsilon > 1:
            raise ValueError("epsilon must be at most 1")
        return {
            "epsilon": epsilon,
            "learning_rate": _number(
                parameters, "learning_rate", 0.18, minimum=np.finfo(float).eps
            ),
            "l2": _number(parameters, "l2", 0.01, minimum=0.0),
        }

    def _initialize(self) -> None:
        self.policy = LogisticContextualBanditPolicy(
            n_arms=3,
            n_features=4,
            epsilon=_as_float(self.parameters["epsilon"], name="epsilon"),
            learning_rate=_as_float(
                self.parameters["learning_rate"], name="learning_rate"
            ),
            l2=_as_float(self.parameters["l2"], name="l2"),
        )
        self.context_rng = generator(self.seed, self.scenario_id, "context")
        self.action_rng = generator(self.seed, self.scenario_id, "action")
        self.reward_rng = generator(self.seed, self.scenario_id, "reward")
        self.theta = np.asarray(
            self.environment.get("theta", self.default_theta), dtype=float
        )
        if self.theta.shape != (3, 4) or not np.all(np.isfinite(self.theta)):
            raise ValueError("theta must be a finite 3 by 4 matrix")
        self.truth_history: list[dict[str, Any]] = []

    def _perform_step(self) -> dict[str, Any]:
        values = self.context_rng.choice(np.asarray([-1.0, 1.0]), size=3)
        feature = np.concatenate((np.ones(1), values))
        context = np.repeat(feature[np.newaxis, :], 3, axis=0)
        probabilities = 1.0 / (1.0 + np.exp(-(self.theta @ feature)))
        predictions = self.policy.predicted_probabilities(context).copy()
        probe_rng = deepcopy(self.action_rng)
        sampled = float(probe_rng.random())
        preview = int(
            self.policy.select_action(context=context, rng=deepcopy(self.action_rng))
        )
        action = int(self.policy.select_action(context=context, rng=self.action_rng))
        if action != preview:
            raise RuntimeError(
                "recommendation diagnostic diverged from policy selection"
            )
        reward = int(self.reward_rng.random() < probabilities[action])
        optimal = int(np.argmax(probabilities))
        regret = float(probabilities[optimal] - probabilities[action])
        theta_before = self.policy.theta.copy()
        self.policy.update(action=action, reward=float(reward), context=context)
        self.truth_history.append(
            {"probabilities": probabilities.copy(), "optimalArm": optimal}
        )
        labels = {
            "visitor": ("new", "returning"),
            "engagement": ("low", "high"),
            "visit": ("weekday", "weekend"),
        }
        cues = [
            {
                "name": name,
                "value": labels[name][int(value > 0)],
                "label": labels[name][int(value > 0)],
            }
            for name, value in zip(labels, values, strict=True)
        ]
        return {
            "selectedArm": action,
            "reward": reward,
            "instantaneousExpectedRegret": regret,
            "visibleCues": cues,
            "publicContext": context,
            "explanationKey": f"recommendations.{min(len(self.history) + 1, 4)}",
            "diagnostic": {
                "kind": "scenario-logistic",
                "decision": {
                    "label": "Predicted click probability",
                    "values": predictions,
                    "selectionBranch": "explore"
                    if sampled < self.policy.epsilon
                    else "exploit",
                },
                "thetaBefore": theta_before,
                "thetaAfter": self.policy.theta.copy(),
                "outcomeLabel": "Click" if reward else "No click",
            },
        }

    def presentation(self) -> dict[str, Any]:
        return {
            "experienceKind": "scenario",
            "experienceId": self.scenario_id,
            "arms": [
                {"name": "Article", "shortName": "Article", "symbolKind": "article"},
                {"name": "Product", "shortName": "Product", "symbolKind": "product"},
                {"name": "Tutorial", "shortName": "Tutorial", "symbolKind": "tutorial"},
            ],
            "rewardPresentation": "binary",
            "positiveOutcomeLabel": "Click",
            "zeroOutcomeLabel": "No click",
        }

    def _hidden_truth(self) -> dict[str, Any]:
        return {"theta": self.theta, "rounds": self.truth_history}

    def generated_code(self) -> str:
        return f"""from pymab.policies import LogisticContextualBanditPolicy

policy = LogisticContextualBanditPolicy(
    n_arms=3,
    n_features=4,
    epsilon={self.parameters["epsilon"]!r},
    learning_rate={self.parameters["learning_rate"]!r},
    l2={self.parameters["l2"]!r},
)
# Build one row per candidate from the current visitor context.
action = policy.select_action(context=context, rng=rng)
policy.update(action=action, reward=float(clicked), context=context)
"""


class DefensiveVerificationScenarioSession(ScenarioSession):
    """Choose proportionate verification inside an approved decision band."""

    scenario_id: ScenarioId = "defensive-verification"
    policy_id = "linucb"

    def validate_parameters(self, parameters: dict[str, object]) -> dict[str, object]:
        return {
            "alpha": _number(parameters, "alpha", 0.75, minimum=np.finfo(float).eps),
            "l2": _number(parameters, "l2", 1.0, minimum=np.finfo(float).eps),
        }

    def _initialize(self) -> None:
        self.policy = LinUCBPolicy(
            n_arms=3,
            n_features=4,
            alpha=_as_float(self.parameters["alpha"], name="alpha"),
            l2=_as_float(self.parameters["l2"], name="l2"),
        )
        self.context_rng = generator(self.seed, self.scenario_id, "context")
        self.action_rng = generator(self.seed, self.scenario_id, "action")
        self.reward_rng = generator(self.seed, self.scenario_id, "reward")
        self.risk_model = {
            "baseRisk": _as_float(
                self.environment.get("baseRisk", 0.32), name="baseRisk"
            ),
            "riskWeight": _as_float(
                self.environment.get("riskWeight", 0.16), name="riskWeight"
            ),
            "newAccountWeight": _as_float(
                self.environment.get("newAccountWeight", 0.07),
                name="newAccountWeight",
            ),
            "sensitiveWeight": _as_float(
                self.environment.get("sensitiveWeight", 0.06),
                name="sensitiveWeight",
            ),
        }
        low = self.risk_model["baseRisk"] - sum(
            self.risk_model[key]
            for key in ("riskWeight", "newAccountWeight", "sensitiveWeight")
        )
        high = self.risk_model["baseRisk"] + sum(
            self.risk_model[key]
            for key in ("riskWeight", "newAccountWeight", "sensitiveWeight")
        )
        if not all(np.isfinite(list(self.risk_model.values()))) or low < 0 or high > 1:
            raise ValueError(
                "risk coefficients must keep abuse probability between 0 and 1"
            )
        if self.mode == "freePlay":
            self.environment = dict(self.risk_model)
        self.truth_history: list[dict[str, Any]] = []

    @staticmethod
    def _expected_utilities(abuse_probability: float) -> np.ndarray:
        legitimate = 1.0 - abuse_probability
        allow = abuse_probability * -1.2 + legitimate * 1.0
        light = abuse_probability * (0.72 * 1.0 + 0.28 * -1.2) + legitimate * (
            0.06 * -0.7 + 0.94 * 0.72
        )
        strong = abuse_probability * (0.96 * 1.05 + 0.04 * -1.2) + legitimate * (
            0.18 * -0.9 + 0.82 * 0.38
        )
        return np.asarray([allow, light, strong], dtype=float)

    def _sample_outcome(self, action: int, abusive: bool) -> tuple[float, str]:
        draw = float(self.reward_rng.random())
        if action == 0:
            return (-1.2, "Abuse allowed") if abusive else (1.0, "Passed with no check")
        if action == 1:
            if abusive:
                return (1.0, "Abuse stopped") if draw < 0.72 else (-1.2, "Abuse missed")
            return (
                (-0.7, "Legitimate user abandoned")
                if draw < 0.06
                else (0.72, "Passed light check")
            )
        if abusive:
            return (1.05, "Abuse stopped") if draw < 0.96 else (-1.2, "Abuse missed")
        return (
            (-0.9, "Legitimate user abandoned")
            if draw < 0.18
            else (0.38, "Passed strong verification")
        )

    def _perform_step(self) -> dict[str, Any]:
        risk = float(self.context_rng.uniform(0.15, 0.85))
        account_new = bool(self.context_rng.integers(2))
        sensitive = bool(self.context_rng.integers(2))
        feature = np.asarray(
            [
                1.0,
                2.0 * risk - 1.0,
                1.0 if account_new else -1.0,
                1.0 if sensitive else -1.0,
            ]
        )
        context = np.repeat(feature[np.newaxis, :], 3, axis=0)
        abuse_probability = (
            self.risk_model["baseRisk"]
            + self.risk_model["riskWeight"] * feature[1]
            + self.risk_model["newAccountWeight"] * feature[2]
            + self.risk_model["sensitiveWeight"] * feature[3]
        )
        expected = self._expected_utilities(abuse_probability)
        abusive = bool(self.reward_rng.random() < abuse_probability)
        action, diagnostic = linucb_decision(self.policy, context, self.action_rng)
        reward, outcome = self._sample_outcome(action, abusive)
        optimal = int(np.argmax(expected))
        regret = float(expected[optimal] - expected[action])
        self.policy.update(action=action, reward=reward, context=context)
        diagnostic.update(
            {
                "aAfter": self.policy.a.copy(),
                "bAfter": self.policy.b.copy(),
                "outcomeLabel": outcome,
            }
        )
        self.truth_history.append(
            {
                "abuseProbability": abuse_probability,
                "expectedUtilities": expected.copy(),
                "abusive": abusive,
                "optimalArm": optimal,
            }
        )
        cues = [
            {"name": "risk", "value": round(risk, 2), "label": f"{risk:.0%}"},
            {
                "name": "account",
                "value": "new" if account_new else "established",
                "label": "new" if account_new else "established",
            },
            {
                "name": "endpoint",
                "value": "sensitive" if sensitive else "routine",
                "label": "sensitive" if sensitive else "routine",
            },
        ]
        return {
            "selectedArm": action,
            "reward": reward,
            "instantaneousExpectedRegret": regret,
            "visibleCues": cues,
            "publicContext": context,
            "explanationKey": f"defensive-verification.{min(len(self.history) + 1, 4)}",
            "diagnostic": diagnostic,
        }

    def presentation(self) -> dict[str, Any]:
        return {
            "experienceKind": "scenario",
            "experienceId": self.scenario_id,
            "arms": [
                {"name": "Allow", "shortName": "Allow", "symbolKind": "allow"},
                {
                    "name": "Light check",
                    "shortName": "Light",
                    "symbolKind": "light-check",
                },
                {
                    "name": "Strong verification",
                    "shortName": "Strong",
                    "symbolKind": "strong-verification",
                },
            ],
            "rewardPresentation": "utility",
            "positiveOutcomeLabel": "Positive utility",
            "zeroOutcomeLabel": "Negative utility",
        }

    def _hidden_truth(self) -> dict[str, Any]:
        return {"rounds": self.truth_history}

    def generated_code(self) -> str:
        return f"""from pymab.policies import LinUCBPolicy

policy = LinUCBPolicy(
    n_arms=3,
    n_features=4,
    alpha={self.parameters["alpha"]!r},
    l2={self.parameters["l2"]!r},
)
# Use only actions already approved by your security policy.
action = policy.select_action(context=context, rng=rng)
policy.update(action=action, reward=observed_utility, context=context)
"""


def create_scenario_session(
    *,
    session_id: str,
    scenario_id: str,
    mode: Mode,
    seed: int,
    parameters: dict[str, object],
    source_commit: str,
    environment: dict[str, object] | None = None,
) -> ScenarioSession:
    """Construct a scenario session from its public identifier."""

    if scenario_id == "recommendations":
        scenario_type: type[ScenarioSession] = RecommendationScenarioSession
    elif scenario_id == "defensive-verification":
        scenario_type = DefensiveVerificationScenarioSession
    else:
        raise ValueError("unknown scenarioId")
    return scenario_type(
        session_id=session_id,
        mode=mode,
        seed=seed,
        parameters=parameters,
        source_commit=source_commit,
        environment=environment,
    )
