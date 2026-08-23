"""Fixed-policy off-policy estimator services."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import numpy as np

from pymab.errors import OverlapError, ValidationError
from pymab.offline.bootstrap import Bootstrapper
from pymab.offline.data import (
    BatchTargetPolicy,
    CrossFittedRewardModel,
    EstimateMethod,
    LoggedBanditDataset,
    OfflineEstimate,
    OverlapStatus,
    TargetPolicy,
    WeightDiagnostics,
)
from pymab.types import FloatArray
from pymab.validation import finite_float, float_array, probability_vector


@dataclass(frozen=True)
class EstimatorConfig:
    """Validated configuration for policy-value estimation."""

    method: EstimateMethod = EstimateMethod.IPS
    weight_clip: float | None = None
    confidence_level: float = 0.95
    bootstrap_resamples: int = 10_000
    bootstrap_max_index_elements: int = 1_000_000
    seed: int = 0

    def __post_init__(self) -> None:
        try:
            method = EstimateMethod(self.method)
        except ValueError as exc:
            raise ValidationError("method must be 'ips', 'snips', or 'dr'") from exc
        object.__setattr__(self, "method", method)
        if self.weight_clip is not None:
            clip = finite_float(self.weight_clip, name="weight_clip")
            if clip <= 0:
                raise ValidationError("weight_clip must be positive")
            object.__setattr__(self, "weight_clip", clip)
        confidence = finite_float(self.confidence_level, name="confidence_level")
        if not 0 < confidence < 1:
            raise ValidationError("confidence_level must be in (0, 1)")
        object.__setattr__(self, "confidence_level", confidence)
        if isinstance(self.bootstrap_resamples, bool) or not isinstance(
            self.bootstrap_resamples, Integral
        ):
            raise TypeError("bootstrap_resamples must be an integer")
        if int(self.bootstrap_resamples) <= 0:
            raise ValidationError("bootstrap_resamples must be positive")
        object.__setattr__(self, "bootstrap_resamples", int(self.bootstrap_resamples))
        if isinstance(self.bootstrap_max_index_elements, bool) or not isinstance(
            self.bootstrap_max_index_elements, Integral
        ):
            raise TypeError("bootstrap_max_index_elements must be an integer")
        if int(self.bootstrap_max_index_elements) <= 0:
            raise ValidationError("bootstrap_max_index_elements must be positive")
        object.__setattr__(
            self,
            "bootstrap_max_index_elements",
            int(self.bootstrap_max_index_elements),
        )
        if isinstance(self.seed, bool) or not isinstance(self.seed, Integral):
            raise TypeError("seed must be an integer")
        object.__setattr__(self, "seed", int(self.seed))


class PolicyValueEstimator:
    """Evaluate a fixed target policy against immutable logged feedback."""

    def __init__(self, config: EstimatorConfig) -> None:
        self.config = config

    def estimate(
        self,
        dataset: LoggedBanditDataset,
        target_policy: TargetPolicy | BatchTargetPolicy,
        *,
        reward_model: CrossFittedRewardModel | None = None,
    ) -> OfflineEstimate:
        if self.config.bootstrap_max_index_elements < dataset.n_events:
            raise ValidationError(
                "bootstrap_max_index_elements must accommodate one complete "
                "logged-data resample"
            )
        if self.config.method is EstimateMethod.DOUBLY_ROBUST and reward_model is None:
            raise ValidationError(
                "doubly robust estimation requires a cross-fitted reward model"
            )
        target = self._target_quantities(
            dataset=dataset,
            target_policy=target_policy,
            reward_model=reward_model,
        )
        raw_weights = np.asarray(
            target.selected_probabilities / dataset.propensities, dtype=float
        )
        if not np.all(np.isfinite(raw_weights)):
            raise OverlapError(
                "importance weights overflowed under insufficient overlap"
            )
        raw_total = float(np.sum(raw_weights))
        if raw_total <= 0 and self.config.method is not EstimateMethod.DOUBLY_ROBUST:
            raise OverlapError(
                f"{self.config.method.value.upper()} is undefined because the target "
                "policy has zero support on every logged action"
            )
        weights = raw_weights.copy()
        if self.config.weight_clip is not None:
            np.minimum(weights, self.config.weight_clip, out=weights)
        contributions = self._contributions(
            dataset=dataset,
            target=target,
            weights=weights,
        )
        estimate = self._point_estimate(contributions=contributions, weights=weights)
        uncertainty = Bootstrapper(
            n_resamples=self.config.bootstrap_resamples,
            confidence_level=self.config.confidence_level,
            seed=self.config.seed,
            max_index_elements=self.config.bootstrap_max_index_elements,
        ).summarize(
            contributions=contributions,
            weights=weights,
            method=self.config.method,
            clusters=(None if dataset.clusters is None else tuple(dataset.clusters)),
        )
        diagnostics = _weight_diagnostics(
            raw_weights=raw_weights,
            effective_weights=weights,
            clipping_threshold=self.config.weight_clip,
        )
        overlap = _overlap_status(
            raw_weights=raw_weights,
            raw_effective_sample_size=diagnostics.raw_effective_sample_size,
            method=self.config.method,
        )
        return OfflineEstimate(
            method=self.config.method,
            estimate=estimate,
            standard_error=uncertainty.standard_error,
            ci_lower=uncertainty.lower,
            ci_upper=uncertainty.upper,
            weights=diagnostics,
            overlap_status=overlap,
            resampling_unit=uncertainty.unit,
            confidence_method="percentile_bootstrap",
            confidence_level=self.config.confidence_level,
            n_events=dataset.n_events,
        )

    def _target_quantities(
        self,
        *,
        dataset: LoggedBanditDataset,
        target_policy: TargetPolicy | BatchTargetPolicy,
        reward_model: CrossFittedRewardModel | None,
    ) -> _TargetQuantities:
        if isinstance(target_policy, BatchTargetPolicy):
            return self._batch_target_quantities(
                dataset=dataset,
                target_policy=target_policy,
                reward_model=reward_model,
            )
        selected = np.empty(dataset.n_events, dtype=float)
        direct = (
            np.empty(dataset.n_events, dtype=float)
            if self.config.method is EstimateMethod.DOUBLY_ROBUST
            else None
        )
        logged_predictions = (
            np.empty(dataset.n_events, dtype=float)
            if self.config.method is EstimateMethod.DOUBLY_ROBUST
            else None
        )
        for index in range(dataset.n_events):
            context = dataset.context_at(index)
            probabilities = probability_vector(
                target_policy.probabilities(context),
                n_arms=dataset.n_arms,
                name=f"target probabilities at event {index}",
            )
            logged_action = int(dataset.actions[index])
            selected[index] = probabilities[logged_action]
            if direct is None or logged_predictions is None:
                continue
            if reward_model is None:
                raise RuntimeError("reward-model validation failed")
            predictions = float_array(
                reward_model.predict_event(index, context),
                name=f"reward-model predictions at event {index}",
                ndim=1,
            )
            if predictions.shape != (dataset.n_arms,):
                raise ValidationError(
                    f"reward model must return one prediction per arm at event {index}"
                )
            direct[index] = float(probabilities @ predictions)
            logged_predictions[index] = predictions[logged_action]
        return _TargetQuantities(
            selected_probabilities=selected,
            direct_values=direct,
            logged_predictions=logged_predictions,
        )

    def _batch_target_quantities(
        self,
        *,
        dataset: LoggedBanditDataset,
        target_policy: BatchTargetPolicy,
        reward_model: CrossFittedRewardModel | None,
    ) -> _TargetQuantities:
        probabilities = float_array(
            target_policy.probabilities_batch(
                dataset.contexts,
                n_events=dataset.n_events,
            ),
            name="batch target probabilities",
            ndim=2,
        )
        if probabilities.shape != (dataset.n_events, dataset.n_arms):
            raise ValidationError(
                "batch target probabilities must have shape (event, arm)"
            )
        if np.any(probabilities < 0) or not np.allclose(
            np.sum(probabilities, axis=1),
            1.0,
            rtol=1e-12,
            atol=1e-12,
        ):
            raise ValidationError(
                "every row of batch target probabilities must be non-negative "
                "and sum to one"
            )
        selected = np.asarray(
            probabilities[np.arange(dataset.n_events), dataset.actions],
            dtype=float,
        )
        if self.config.method is not EstimateMethod.DOUBLY_ROBUST:
            return _TargetQuantities(selected, None, None)
        if reward_model is None:
            raise RuntimeError("reward-model validation failed")
        direct = np.empty(dataset.n_events, dtype=float)
        logged_predictions = np.empty(dataset.n_events, dtype=float)
        for index in range(dataset.n_events):
            context = dataset.context_at(index)
            predictions = float_array(
                reward_model.predict_event(index, context),
                name=f"reward-model predictions at event {index}",
                ndim=1,
            )
            if predictions.shape != (dataset.n_arms,):
                raise ValidationError(
                    f"reward model must return one prediction per arm at event {index}"
                )
            direct[index] = float(probabilities[index] @ predictions)
            logged_predictions[index] = predictions[int(dataset.actions[index])]
        return _TargetQuantities(selected, direct, logged_predictions)

    def _contributions(
        self,
        *,
        dataset: LoggedBanditDataset,
        target: _TargetQuantities,
        weights: FloatArray,
    ) -> FloatArray:
        if self.config.method in {EstimateMethod.IPS, EstimateMethod.SNIPS}:
            return np.asarray(weights * dataset.rewards, dtype=float)
        if target.direct_values is None or target.logged_predictions is None:
            raise RuntimeError("doubly robust target quantities are missing")
        return np.asarray(
            target.direct_values
            + weights * (dataset.rewards - target.logged_predictions),
            dtype=float,
        )

    def _point_estimate(
        self, *, contributions: FloatArray, weights: FloatArray
    ) -> float:
        if self.config.method is not EstimateMethod.SNIPS:
            return float(np.mean(contributions))
        denominator = float(np.sum(weights))
        if denominator <= 0:
            raise OverlapError("SNIPS is undefined because all target weights are zero")
        return float(np.sum(contributions) / denominator)


@dataclass(frozen=True)
class _TargetQuantities:
    selected_probabilities: FloatArray
    direct_values: FloatArray | None
    logged_predictions: FloatArray | None


def _weight_diagnostics(
    *,
    raw_weights: FloatArray,
    effective_weights: FloatArray,
    clipping_threshold: float | None,
) -> WeightDiagnostics:
    return WeightDiagnostics(
        raw_effective_sample_size=_effective_sample_size(raw_weights),
        effective_sample_size=_effective_sample_size(effective_weights),
        raw_max_weight=float(np.max(raw_weights)),
        max_weight=float(np.max(effective_weights)),
        raw_mean_weight=float(np.mean(raw_weights)),
        mean_weight=float(np.mean(effective_weights)),
        clipped_fraction=float(np.mean(raw_weights != effective_weights)),
        clipping_threshold=clipping_threshold,
    )


def _effective_sample_size(weights: FloatArray) -> float:
    squared_sum = float(np.sum(weights**2))
    if squared_sum == 0:
        return 0.0
    return float(np.sum(weights)) ** 2 / squared_sum


def _overlap_status(
    *,
    raw_weights: FloatArray,
    raw_effective_sample_size: float,
    method: EstimateMethod,
) -> OverlapStatus:
    if not np.any(raw_weights > 0):
        return (
            OverlapStatus.MODEL_ONLY
            if method is EstimateMethod.DOUBLY_ROBUST
            else OverlapStatus.NONE
        )
    if (
        raw_effective_sample_size < 0.1 * raw_weights.size
        or float(np.max(raw_weights)) > 100.0
    ):
        return OverlapStatus.WEAK
    return OverlapStatus.ADEQUATE


def estimate_policy_value(
    dataset: LoggedBanditDataset,
    target_policy: TargetPolicy | BatchTargetPolicy,
    *,
    method: EstimateMethod | str = EstimateMethod.IPS,
    reward_model: CrossFittedRewardModel | None = None,
    weight_clip: float | None = None,
    confidence_level: float = 0.95,
    bootstrap_resamples: int = 10_000,
    bootstrap_max_index_elements: int = 1_000_000,
    seed: int = 0,
) -> OfflineEstimate:
    """Estimate a fixed target policy using a validated estimator service."""

    config = EstimatorConfig(
        method=_validate_method(method),
        weight_clip=weight_clip,
        confidence_level=confidence_level,
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_max_index_elements=bootstrap_max_index_elements,
        seed=seed,
    )
    return PolicyValueEstimator(config).estimate(
        dataset,
        target_policy,
        reward_model=reward_model,
    )


def _validate_method(value: EstimateMethod | str) -> EstimateMethod:
    try:
        return EstimateMethod(value)
    except ValueError as exc:
        raise ValidationError("method must be 'ips', 'snips', or 'dr'") from exc


__all__ = ["EstimatorConfig", "PolicyValueEstimator", "estimate_policy_value"]
