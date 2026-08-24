"""Stateful, deterministic lesson sessions backed by real PyMAB policies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

import pymab
from pymab._random import generator
from pymab.policies import EpsilonGreedyPolicy, LinUCBPolicy
from pymab_demo.diagnostics import epsilon_decision, linucb_decision
from pymab_demo.fixtures import (
    CUE_NAMES,
    EPSILON_MEANS,
    FIXTURES,
    GATE_IDS,
    LINUCB_THETA,
    LessonId,
    Mode,
    horizon_for,
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
    ) -> None:
        self.session_id = session_id
        self.lesson_id = lesson_id
        self.mode = mode
        self.seed = seed
        self.parameters = validate_parameters(lesson_id, parameters)
        self.source_commit = source_commit
        self.horizon = horizon_for(lesson_id, mode)
        self.history: list[dict[str, Any]] = []
        self.total_reward = 0
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
        self.total_reward += int(event["reward"])
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
        self.total_reward = 0
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
        fixture = FIXTURES[self.lesson_id]
        passed = (
            complete
            and self.total_reward >= fixture.reward_target
            and self.cumulative_regret <= fixture.regret_target
        )
        return {
            "lessonId": self.lesson_id,
            "mode": self.mode,
            "seed": self.seed,
            "packageVersion": pymab.__version__,
            "sourceCommit": self.source_commit,
            "sessionId": self.session_id,
            "step": len(self.history),
            "horizon": self.horizon,
            "parameters": self.parameters,
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
            "history": self.history,
            "hiddenTruth": self._hidden_truth() if complete else None,
            "generatedCode": self.generated_code(),
        }

    @abstractmethod
    def _hidden_truth(self) -> dict[str, Any]:
        """Return environment truth for completed-run debriefs."""

    @abstractmethod
    def generated_code(self) -> str:
        """Create an equivalent public-API example."""


class EpsilonLessonSession(LessonSession):
    """Three Bernoulli gates taught with ``EpsilonGreedyPolicy``."""

    def _initialize(self) -> None:
        self.policy = EpsilonGreedyPolicy(n_arms=3, epsilon=self.parameters["epsilon"])
        self.action_rng = generator(self.seed, "epsilon-greedy", "lesson", "action")
        self.reward_rng = generator(self.seed, "epsilon-greedy", "lesson", "reward")

    def _perform_step(self) -> dict[str, Any]:
        action, diagnostic = epsilon_decision(self.policy, self.action_rng)
        potential = (self.reward_rng.random(3) < np.asarray(EPSILON_MEANS)).astype(int)
        reward = int(potential[action])
        regret = float(max(EPSILON_MEANS) - EPSILON_MEANS[action])
        self.policy.update(action=action, reward=float(reward))
        diagnostic.update(
            {
                "countsAfter": self.policy.counts.copy(),
                "estimatesAfter": self.policy.estimates.copy(),
            }
        )
        return {
            "selectedArm": action,
            "reward": reward,
            "instantaneousExpectedRegret": regret,
            "visibleCues": [],
            "publicContext": None,
            "explanationKey": f"epsilon.{diagnostic['selectionBranch']}",
            "diagnostic": diagnostic,
        }

    def _hidden_truth(self) -> dict[str, Any]:
        return {"probabilities": EPSILON_MEANS, "optimalArm": 2}

    def generated_code(self) -> str:
        from pymab_demo.codegen import epsilon_example

        return epsilon_example(
            seed=self.seed, epsilon=self.parameters["epsilon"], horizon=self.horizon
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
        self.truth_history: list[tuple[np.ndarray, int]] = []

    def _perform_step(self) -> dict[str, Any]:
        cue_values = self.context_rng.choice(np.asarray([-1.0, 1.0]), size=3)
        feature = np.concatenate((np.ones(1), cue_values))
        context = np.repeat(feature[np.newaxis, :], 3, axis=0)
        probabilities = 1.0 / (1.0 + np.exp(-(LINUCB_THETA @ feature)))
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
        return {
            "selectedArm": action,
            "reward": reward,
            "instantaneousExpectedRegret": regret,
            "visibleCues": cues,
            "publicContext": context,
            "explanationKey": "linucb.decision",
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
            "theta": LINUCB_THETA,
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
        )


def create_session(
    *,
    session_id: str,
    lesson_id: LessonId,
    mode: Mode,
    seed: int,
    parameters: dict[str, object],
    source_commit: str,
) -> LessonSession:
    """Construct the correct concrete lesson session."""

    session_type = (
        EpsilonLessonSession if lesson_id == "epsilon-greedy" else LinUCBLessonSession
    )
    return session_type(
        session_id=session_id,
        lesson_id=lesson_id,
        mode=mode,
        seed=seed,
        parameters=parameters,
        source_commit=source_commit,
    )
