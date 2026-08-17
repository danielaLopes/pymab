Reward models and arm priors
============================

Reward models describe observation noise conditional on true arm means. Arm
priors generate those means. Keeping the concepts separate avoids interpreting
one generic ``scale`` as a Gaussian standard deviation in one place and a Beta
concentration in another.

Available reward models are ``GaussianReward``, ``BernoulliReward``, and
``UniformReward``. Available priors are ``GaussianArmPrior``, ``BetaArmPrior``,
and ``UniformArmPrior``.

.. automodule:: pymab.distributions
   :members:
   :show-inheritance:
