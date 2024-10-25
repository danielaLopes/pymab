.. PyMultiBandits documentation master file, created by
   sphinx-quickstart on Fri Aug  2 17:05:35 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

============================
Policies documentation
============================


Policy (policy.py)
------------------

The Policy class serves as the base class for all bandit algorithms in PyMultiBandits. It defines the common interface and basic functionality that all specific policy implementations should follow.

.. automodule:: pymab.policies.policy
   :members:
   :undoc-members:
   :show-inheritance:


Bayesian UCB (bayesian_ucb.py)
------------------------------

Bayesian Upper Confidence Bound (UCB) is an advanced policy that uses Bayesian inference to balance exploration and exploitation. It maintains a probability distribution over the expected rewards of each arm and uses this to make decisions.

.. automodule:: pymab.policies.bayesian_policy
   :members:
   :undoc-members:
   :show-inheritance:


Contextual Bandits (contextual_bandits.py)
------------------------------------------

Contextual Bandits extend the standard multi-armed bandit problem by incorporating contextual information. This policy adapts its arm selection based on additional features or context provided with each decision.

.. automodule:: pymab.policies.contextual_bandits
   :members:
   :undoc-members:
   :show-inheritance:


Epsilon Greedy (epsilon_greedy.py)
----------------------------------

Epsilon Greedy is a simple yet effective policy that balances exploration and exploitation. It chooses the best-known arm with probability 1-ε, and explores randomly with probability ε.

.. automodule:: pymab.policies.epsilon_greedy
   :members:
   :undoc-members:
   :show-inheritance:


Gradient (gradient.py)
----------------------

The Gradient policy uses a preference-based approach, updating numerical preferences for each arm based on the rewards received. It selects arms probabilistically based on these preferences.

.. automodule:: pymab.policies.gradient
   :members:
   :undoc-members:
   :show-inheritance:


Greedy (greedy.py)
------------------

The Greedy policy always selects the arm with the highest estimated value. While simple, it can be effective in certain scenarios, especially with optimistic initialization.

.. automodule:: pymab.policies.greedy
   :members:
   :undoc-members:
   :show-inheritance:


Softmax Selection (softmax_selection.py)
----------------------------------------

Softmax Selection chooses arms probabilistically based on their estimated values. Arms with higher estimated values have a higher probability of being selected, allowing for a degree of exploration.

.. automodule:: pymab.policies.softmax_selection
   :members:
   :undoc-members:
   :show-inheritance:


Thompson Sampling (thompson_sampling.py)
----------------------------------------

Thompson Sampling is a probabilistic algorithm that chooses arms based on randomly drawn samples from the posterior distribution of each arm's reward. It naturally balances exploration and exploitation.

.. automodule:: pymab.policies.thompson_sampling
   :members:
   :undoc-members:
   :show-inheritance:


UCB (ucb.py)
------------

Upper Confidence Bound (UCB) is a deterministic policy that balances exploration and exploitation by selecting the arm with the highest upper confidence bound. It's known for its strong theoretical guarantees.

.. automodule:: pymab.policies.ucb
   :members:
   :undoc-members:
   :show-inheritance:



--------------------------------------------------------------------------
Example Usage
--------------------------------------------------------------------------

.. code-block:: python

   from pymab.policies.greedy import GreedyPolicy
   from pymab.policies.epsilon_greedy import EpsilonGreedyPolicy
   from pymab.policies.bayesian_ucb import BayesianUCBPolicy

   from pymab.policies.thompson_sampling import ThompsonSamplingPolicy
   from pymab.game import Game

   n_bandits = 10

   greedy_policy = GreedyPolicy(n_bandits=n_bandits,
                              optimistic_initialization=0)

   greedy_policy_optimistic_initialization_1 = GreedyPolicy(n_bandits=n_bandits,
                                                         optimistic_initialization=1)

   greedy_policy_optimistic_initialization_5 = GreedyPolicy(n_bandits=n_bandits,
                                                         optimistic_initialization=5)

   epsilon_greedy_policy_0_01 = EpsilonGreedyPolicy(n_bandits=n_bandits,
                                          epsilon=0.01)

   epsilon_greedy_policy_0_1 = EpsilonGreedyPolicy(n_bandits=n_bandits,
                                          epsilon=0.1)

   epsilon_greedy_policy_0_5 = EpsilonGreedyPolicy(n_bandits=n_bandits,
                                          epsilon=0.5)

   ucb_policy_0 = BayesianUCBPolicy(n_bandits=n_bandits,
                        c=0)

   ucb_policy_1 = BayesianUCBPolicy(n_bandits=n_bandits,
                        c=1)

   ucb_policy_2 = BayesianUCBPolicy(n_bandits=n_bandits,
                        c=2)

   thomson_sampling = ThompsonSamplingPolicy(n_bandits=n_bandits)

   n_bandits = 10

   # Setup the game
   game = Game(n_episodes=200, 
               n_steps=100, 
               policies=[greedy_policy,
                     greedy_policy_optimistic_initialization_1,
                     greedy_policy_optimistic_initialization_5,
                     epsilon_greedy_policy_0_01,
                     epsilon_greedy_policy_0_1,
                     epsilon_greedy_policy_0_5,
                     ucb_policy_0,
                     ucb_policy_1,
                     ucb_policy_2,
                     thomson_sampling
                  ], 
               n_bandits=n_bandits,
               )

   game.game_loop()
   
   game.plot_average_reward_by_step()