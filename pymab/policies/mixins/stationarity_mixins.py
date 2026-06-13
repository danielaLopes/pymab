"""Deprecated compatibility mixins.

The v1 policy API no longer uses mixins for stationarity. Use
``pymab.policies.UCBPolicy``, ``SlidingWindowUCBPolicy``, or
``DiscountedUCBPolicy`` directly.
"""

from __future__ import annotations

from warnings import warn


class StationaryPolicyMixin:
    """Deprecated no-op compatibility mixin."""

    def __init_subclass__(cls, **kwargs: object) -> None:
        warn(
            "StationaryPolicyMixin is deprecated; inherit from a concrete v1 "
            "policy instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init_subclass__(**kwargs)


class NonStationaryPolicyMixin:
    """Deprecated no-op compatibility mixin."""

    def __init_subclass__(cls, **kwargs: object) -> None:
        warn(
            "NonStationaryPolicyMixin is deprecated; use v1 non-stationary "
            "policies instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init_subclass__(**kwargs)


class SlidingWindowMixin(NonStationaryPolicyMixin):
    """Deprecated compatibility name."""

    def __init__(self, *, window_size: int = 100) -> None:
        self.window_size = window_size


class DiscountedMixin(NonStationaryPolicyMixin):
    """Deprecated compatibility name."""

    def __init__(self, *, discount_factor: float = 0.9) -> None:
        self.discount_factor = discount_factor
