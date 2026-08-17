"""Property-based checks for strict public data boundaries."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from pymab import BanditEnvironment, Experiment, ExperimentConfig, SimulationResult
from pymab.policies import RandomPolicy
from pymab.validation import integer_array, probability_vector


@given(st.lists(st.integers(min_value=0, max_value=10), min_size=1, max_size=50))
def test_integer_array_preserves_valid_integer_labels(values: list[int]) -> None:
    result = integer_array(values, name="labels", ndim=1, minimum=0)
    assert result.dtype == np.int64
    assert result.tolist() == values


@given(
    st.lists(
        st.one_of(
            st.booleans(),
            st.text(min_size=0, max_size=3),
            st.floats(allow_nan=True, allow_infinity=True),
        ),
        min_size=1,
        max_size=10,
    ).filter(
        lambda values: any(isinstance(value, (bool, str, float)) for value in values)
    )
)
def test_integer_array_never_coerces_non_integer_labels(values: list[object]) -> None:
    with pytest.raises(ValueError, match="integers"):
        integer_array(values, name="labels", ndim=1)


@given(
    st.lists(
        st.floats(
            min_value=1e-6,
            max_value=1e6,
            allow_nan=False,
            allow_infinity=False,
        ),
        min_size=1,
        max_size=20,
    )
)
def test_probability_vector_accepts_normalized_finite_weights(
    weights: list[float],
) -> None:
    values = np.asarray(weights, dtype=float)
    values /= np.sum(values)
    result = probability_vector(values, n_arms=len(values), name="probabilities")
    assert np.all(result >= 0)
    assert np.sum(result) == pytest.approx(1.0)


@settings(max_examples=12, deadline=None)
@given(
    horizon=st.integers(min_value=1, max_value=8),
    n_replicates=st.integers(min_value=1, max_value=4),
    seed=st.integers(min_value=0, max_value=2**32 - 1),
)
def test_result_payload_roundtrip_is_value_equal(
    horizon: int, n_replicates: int, seed: int
) -> None:
    result = Experiment(
        environment=BanditEnvironment(means=np.array([0.0, 1.0])),
        policies={"random": RandomPolicy(n_arms=2)},
        config=ExperimentConfig(
            horizon=horizon,
            n_replicates=n_replicates,
            seed=seed,
        ),
    ).run()
    restored = SimulationResult.from_dict(result.to_dict())
    assert restored.equals(result)
