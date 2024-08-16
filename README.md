# PyMultiBandits
<img src="assets/icon.png" alt="Icon description" style="width:200px; height:auto;">

Python library for Multi-Armed Bandit algorithms.



## Features
* Design to compare different algorithms in the same environment.
* Built-in plotting functions to visualize the results.
* Support for several Multi-Armed Bandit algorithms.
* Support for different types of reward distributions (Gaussian, Bernoulli, Stationary, Stochastic, etc).


### Multi-Armed Bandit algorithms and reward distributions
#### Basic Exploration-Exploitation Algorithms
* **Greedy:**
  * Reward Distribution: Assumes stationary, deterministic, or stochastic rewards.
* **Epsilon-Greedy:**
  * Reward Distribution: Assumes stationary, stochastic rewards. Supports Bernoulli and Gaussian distributions.

#### Upper Confidence Bound (UCB) Algorithms
* **UCB:**
  * Reward Distribution: Assumes stationary, stochastic rewards. Typically supports Gaussian distributions, but can be adapted for Bernoulli distributions.
* **Bayesian UCB:**
  * Reward Distribution: Can handle non-stationary rewards by updating beliefs about the distribution over time. This library contains implementations for both Bernoulli and Gaussian distributions.

#### Bayesian Methods
* **Thompson Sampling:**
  * Reward Distribution: Assumes stationary, stochastic rewards but can be adapted for non-stationary settings by updating posterior distributions. This library contains implementations for both Bernoulli and Gaussian distributions.

#### Softmax and Gradient-Based Methods
* **Softmax Selection:**
  * Reward Distribution: Assumes stationary, stochastic rewards.
* **Gradient:**
  * Reward Distribution: Assumes stationary, stochastic rewards.

#### Contextual Bandits
* **Contextual Bandits:**
  * Reward Distribution: Can handle both stationary and non-stationary rewards. Assumes that rewards are stochastic and can vary with the context.


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
sphinx-build -b html source/ build/
```


## TODOs
* Add implementation for softmax_selection, contextual_bandits, and bayesian_ucb
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