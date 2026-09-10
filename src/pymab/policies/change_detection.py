"""Compatibility imports for Python reference change-detection policies."""

from pymab._reference.policies.change_detection import (
    ChangePointUCBPolicy as _ChangePointUCBPolicy,
)
from pymab._reference.policies.change_detection import (
    CUSUMUCBPolicy as _CUSUMUCBPolicy,
)
from pymab._reference.policies.change_detection import (
    PageHinkleyUCBPolicy as _PageHinkleyUCBPolicy,
)
from pymab.policies._native_mixin import native_policy_class

ChangePointUCBPolicy = native_policy_class(
    "change_point_ucb", _ChangePointUCBPolicy, module=__name__
)
CUSUMUCBPolicy = native_policy_class("cusum_ucb", _CUSUMUCBPolicy, module=__name__)
PageHinkleyUCBPolicy = native_policy_class(
    "page_hinkley_ucb", _PageHinkleyUCBPolicy, module=__name__
)
ChangePointUCBPolicy.register(CUSUMUCBPolicy)
ChangePointUCBPolicy.register(PageHinkleyUCBPolicy)

__all__ = ["CUSUMUCBPolicy", "ChangePointUCBPolicy", "PageHinkleyUCBPolicy"]
