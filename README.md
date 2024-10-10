# PyMAB
<img src="assets/icon.png" alt="Icon description" style="width:200px; height:auto;">

Python Multi-Armed Bandit Library
Tame the randomness, pull the right levers!
PyMab: Your trusty sidekick in the wild world of exploration and exploitation.


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



## Examples
In ./examples/ you can find detailed examples of how to use the library:
* [Basic Usage](examples/basic_usage.ipynb): A simple example of how to use the library with the Greedy Policy.
* [Bayesian UCB Bernoulli](examples/bayesian_ucb_bernoulli.ipynb): An example comparing the UCB Policy (with c=0, c=1, and c=2) with the Bayesian UCB Policy with a Bernoulli Reward Distribution.
* [Comparing Policies Gaussian](examples/comparing_policies_gaussian.ipynb): 
* [Contextual Bandits Proxies](examples/contextual_bandits_proxies.ipynb): 
* [Thompson Sampling Bernoulli](examples/thompson_sampling_bernoulli.ipynb): 
* [Thompson Sampling Gaussian](examples/thompson_sampling_gaussian.ipynb): 


## Build documentation locally
```bash
sphinx-build -b html docs/source/ docs/build/
```


## TODOs
* Add implementation for softmax_selection, and bayesian_ucb
* Check why the thompson sampling is doesn't have amazing results with gaussian distribution (might just be how it works and not an error),
  * Check if it works better with other armed bandits
* Test if algorithms are working with bernoulli distribution
* add tests
* make class with optimized initialization, without it being in Policy since most policies don't use it.
* Complete remaining algorithms
* Create testing pipeline on git
* Check better way to do this
```python
for policy in game.policies:
    policy.Q_values = game.Q_values
    policy.plot_distribution()
```
If I don't pass Q_values, these will only be set in the game loop, and this will fail. Find a better way to initialize the first values.

* Test all examples and apply fixes
* Add example for non stationary with particular appropriate algorithms
* Fix stationary, make tests, and ensures this can also change the variance, and probably we should have a variance per arm.
* Change the plots to plotly or something
* Handle project dependencies in setup.py and all of that, to make pip installable
* Make tests for non stationary and mixin 
* Make non stationary for other algorithms like greedy.

* Add more complex algorithms for non-stationary, like:
  * https://towardsdatascience.com/reinforcement-learning-basics-stationary-and-non-stationary-multi-armed-bandit-problem-cfe06d33b815
  * https://gdmarmerola.github.io/non-stationary-bandits/
    * exponentially weighted means
    * weighted block means
    * fitting a time series model to find an indicator of when a distribution changes and tune the exploration rate accordingly
  * Add a new environment change mixin that changes at random steps, or changes the variance, or changes the mean, etc.
  * Finish tests
  