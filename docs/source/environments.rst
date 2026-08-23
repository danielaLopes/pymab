Environments
============

Environment dynamics declare the reward domains they support. Use
``ProbabilityDrift`` for Bernoulli means; additive ``GradualDrift`` and
``AbruptShift`` are restricted to unbounded real-valued means. Use
``LogisticContextualEnvironment`` when binary contextual rewards follow a logit
link.

Custom reward models, dynamics, and stateful context providers must implement a
``clone`` method that returns independent state for each replicate. The default
base implementations use ``deepcopy``. Plain context callables are wrapped as
stateless providers and must not close over mutable state; subclass
``ContextProvider`` when state is required.

.. automodule:: pymab.environments
   :members:
   :undoc-members:
   :show-inheritance:
