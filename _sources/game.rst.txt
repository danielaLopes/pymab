.. PyMultiBandits documentation master file, created by
   sphinx-quickstart on Fri Aug  2 17:05:35 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

============================
Game documentation
============================

--------------------------------------------------------------------------
Game (game.py)
--------------------------------------------------------------------------
The Game class is the core component of the PyMultiBandits library. It simulates a multi-armed bandit environment where various policies can be tested and compared. Key features include:

- Configurable number of bandits and episodes
- Support for multiple policies in a single game
- Built-in methods for running simulations and collecting data
- Visualization tools for analyzing policy performance

This class provides a flexible framework for experimenting with different bandit algorithms and understanding their behavior in various scenarios.

.. automodule:: pymab.game
   :members:
   :undoc-members:
   :show-inheritance:



--------------------------------------------------------------------------
Example Usage
--------------------------------------------------------------------------

.. code-block:: python

   from pymab.policies.greedy import GreedyPolicy
   from pymab.game import Game

   n_bandits = 10

   policy = GreedyPolicy(optimistic_initialization=1, n_bandits=n_bandits)
   
   game = Game(n_episodes=2000, 
            n_steps=1000, 
            policies=[policy], 
            n_bandits=n_bandits)

   game.game_loop()

   game.plot_average_reward_by_step()
