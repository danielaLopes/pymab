.. PyMAB documentation master file, created by
   sphinx-quickstart on Fri Aug  2 17:05:35 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

PyMAB documentation
============================

Python Multi-Armed Bandit Library
Tame the randomness, pull the right levers!
PyMab: Your trusty sidekick in the wild world of exploration and exploitation.

PyMAB offers an exploratory framework to compare the performance of multiple Multi Armed Bandit algorithms in a variety of scenarios. The library is designed to be flexible and easy to use, allowing users to quickly set up and run experiments with different configurations.

--------------------------------------------
Simple Example
--------------------------------------------
.. code-block:: python

    from pymab.policies.greedy import GreedyPolicy
    from pymab.policies.thompson_sampling import ThompsonSamplingPolicy
    from pymab.game import Game

    n_bandits = 5

    # Define the policies
    greedy_policy = GreedyPolicy(
                        optimistic_initialization=1,
                        n_bandits=n_bandits
                    )
    ts_policy = ThompsonSamplingPolicy(n_bandits=n_bandits)

    # Define the game
    game = Game(
         n_episodes=2000,
         n_steps=1000,
         policies=[greedy_policy, ts_policy],
         n_bandits=n_bandits
    )

    # Run the game
    game.game_loop()

    # Plot the results
    game.plot_average_reward_by_step()



.. toctree::
   :maxdepth: 2
   :caption: Contents:

   policies
   game
   reward_distribution



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`