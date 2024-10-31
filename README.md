# PyMAB

<p align="center">
  <img src="assets/icon.png" alt="Icon description" style="width:200px; height:auto;">
</p>


[![PyPI version](https://badge.fury.io/py/pymab.svg)](https://badge.fury.io/py/pymab)
[![GitHub license](https://img.shields.io/github/license/danielaLopes/pymab)](https://github.com/yourusername/pymab/blob/main/LICENSE)
[![GitHub issues](https://img.shields.io/github/issues/danielaLopes/pymab)](https://github.com/yourusername/pymab/issues)


Python Multi-Armed Bandit Library
Tame the randomness, pull the right levers!
PyMab: Your trusty sidekick in the wild world of exploration and exploitation.

PyMAB offers an exploratory framework to compare the performance of multiple Multi Armed Bandit algorithms in a variety of scenarios. The library is designed to be flexible and easy to use, allowing users to quickly set up and run experiments with different configurations.


## Simple Example
```python
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
```

### Other examples
In ./examples/ you can find detailed examples of how to use the library:
* [Basic Usage](examples/basic_usage.ipynb): A simple example of how to use the library with the Greedy Policy.
* [Bayesian UCB Bernoulli](examples/bayesian_ucb_bernoulli.ipynb): An example comparing the multiple configurations of the UCB and the Bayesian UCB policies with a Bernoulli reward distribution.
* [Comparing Policies Gaussian](examples/comparing_policies_gaussian.ipynb): An example comparing the multiple configurations of the Greedy, the Epsilon Greedy, the Bayesian UCB, and the Thompson Sampling policies with a Gaussian reward distribution.
* [Contextual Bandits Proxies](examples/contextual_bandits_proxies.ipynb): An example comparing the multiple configurations of the Greedy abd the Contextual Bandits for **proxy server selection** based on latency, bandwidth, downtime rate, success rate, proximity and load.
* [Non-Stationary Policies](examples/non_stationary_policies.ipynb): An example comparing the multiple configurations of the UCB, the Sliding Window UCB, and the Discounted UCB policies in a stationary environment, non-stationary with gradual changes, non-stationary with abrupt changes, and non-stationary with random arm swapping.
* [Thompson Sampling Bernoulli](examples/thompson_sampling_bernoulli.ipynb): An example comparing the multiple configurations of the Greedy, and the Thompson Sampling policies with a Bernoulli reward distribution, showing the evolution of the reward distribution estimations.
* [Thompson Sampling Gaussian](examples/thompson_sampling_gaussian.ipynb): An example comparing the multiple configurations of the Greedy, and the Thompson Sampling policies with a Gaussian reward distribution, showing the evolution of the reward distribution estimations.


## Features
* Design to compare different algorithms in the same environment.
* Built-in plotting functions to visualize the results.
* Support for several Multi-Armed Bandit algorithms.
* Support for different types of reward distributions (Gaussian, Bernoulli).
* Support for different types of environments (Stationary, Non-Stationary).


### Environments
### Stationary
* The reward distribution remains the same during the whole execution.

### Non-Stationary
* The reward distribution changes during the execution. This library contains a mixin to create non-stationary environments that change the reward distribution in multiple ways, and easily extensible.

* **Gradual Change:** The reward distribution will change slightly each step.

* **Abrupt Change:** The reward distribution will change more, periodically.

* **Random Arm Swapping:** The reward distribution will change by swapping the rewards between arms and at random steps.


### Policies
#### Multi-Armed Bandit algorithms and reward distributions
##### Basic Exploration-Exploitation Algorithms
* **Greedy:**
  * Always selects the arm with the highest estimated reward.
  * Very simple but can get stuck on suboptimal arms if initial estimates are inaccurate.
  * No exploration, all exploitation.

* **Epsilon-Greedy:**
  * Most of the time (1-ε), selects the best arm like Greedy.
  * Sometimes (ε), randomly selects an arm to explore.
  * Balances exploration and exploitation, but exploration doesn't decrease over time.


##### Upper Confidence Bound (UCB) Algorithms
* **UCB:** 
  * Selects the arm with the highest upper confidence bound.
  * Automatically balances exploration and exploitation.
  * Explores less-pulled arms more, but focuses on promising arms over time.
  * Has adaptations for non-stationary environments.
    * **SlidingWindowUCB:** 
      * Like UCB, but only considers recent observations.
      * Adapts better to abrupt changes in reward distributions.
    * **DiscountedUCB:** 
      * Like UCB, but gives more weight to recent observations.
      * Adapts better to gradual changes in reward distributions.
  
* **Bayesian UCB:** 
  * Can incorporate prior knowledge about reward distributions.
  * Has adaptations for Bernoulli and Gaussian reward distributions.
  

##### Bayesian Methods
* **Thompson Sampling:** 
  * Samples from estimated reward distributions and picks the highest sample.
  * Naturally balances exploration and exploitation.
  * Often performs very well in practice, especially with enough samples.
  * Has adaptations for Bernoulli and Gaussian reward distributions.

##### Softmax and Gradient-Based Methods
* **Softmax Selection:** 
  * To be implemented
  * Selects arms probabilistically based on their estimated values.
  * Higher estimated values have higher probability of being selected.
  * Temperature parameter controls exploration-exploitation trade-off.

* **Gradient:** 
  * To be implemented
  * Updates a preference for each arm based on the reward received.
  * Doesn't maintain estimates of actual reward values.
  * Can work well in relative reward scenarios.

##### Contextual Bandits
* **Contextual Bandits:**
  * Takes into account additional information (context) when making decisions.
  * Can learn to select different arms in different contexts.
  * More powerful but also more complex than standard bandits.


## Roadmap
* [ ] Add implementation for otimised Greedy policies for non-stationary environments.
* [ ] Add more complex policy adaptions for non-stationary environments.
  * [ ] https://towardsdatascience.com/reinforcement-learning-basics-stationary-and-non-stationary-multi-armed-bandit-problem-cfe06d33b815
  * [ ] https://gdmarmerola.github.io/non-stationary-bandits/
    * [ ] Exponentially weighted means
    * [ ] Weighted block means
    * [ ] Fitting a time series model to find an indicator of when a distribution changes and tune the exploration rate accordingly
* [ ] Add more complex non-stationary environments, like changing the variance, mean, random abrupt changes, machine learning, ...
* [ ] Add unit tests for non-stationary environments and policies.
* [ ] Make mixin for optimistic initialisation, since not all policies use it.
* [ ] Add implementation for softmax_selection policy.
* [ ] Add implementation for gradient policy. 


* [Github Project Board](examples/basic_usage.ipynb):

**This is an ongoing project, and we are always looking for suggestions and contributions. If you have any ideas or want to help, please reach out to us!**
  
