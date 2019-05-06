<img align="left" src="https://github.com/danaugrs/huskarl/blob/master/logo.png">

# Huskarl [![PyPI version](https://badge.fury.io/py/huskarl.svg)](https://badge.fury.io/py/huskarl)

Huskarl is a framework for deep reinforcement learning focused on research and fast prototyping.
It's built on TensorFlow 2.0 and uses the `tf.keras` API when possible for conciseness and readability.

Huskarl makes it easy to parallelize computation of environment dynamics across multiple CPUs.
This is useful for speeding up on-policy learning algorithms that benefit from multiple concurrent sources of experience such as A2C or PPO.
It is specially useful for computationally intensive environments such as physics-based ones.

Huskarl works seamlessly with [OpenAI Gym](https://gym.openai.com/) environments.

There are plans to support multi-agent environments and [Unity3D environments](https://unity3d.ai).

## Algorithms

Several algorithms are implemented already and many more are planned.

* [x] Deep Q-Learning Network (DQN)
* [x] Multi-step DQN
* [x] Double DQN
* [x] Dueling Architecture DQN
* [x] Advantage Actor-Critic (A2C)
* [x] Deep Deterministic Policy Gradient (DDPG)
* [ ] Proximal Policy Optimization (PPO)
* [ ] Prioritized Experience Replay
* [ ] Curiosity-Driven Exploration


## Installation
To install the latest published version:
```
pip install huskarl
```
To install directly from source clone this repository and run the following command inside it:
```
pip install -e .
```

## Examples
To run the examples you will need `matplotlib` and `gym`.
### dqn-cartpole.py
![dqn-cartpole.gif](examples/dqn-cartpole.gif)
### ddpg-pendulum.py
![ddpg-pendulum.gif](examples/ddpg-pendulum.gif)
### a2c-cartpole.py
![a2c-cartpole.gif](examples/a2c-cartpole.gif)

## Citing

If you use Huskarl in your research, you can cite it as follows:
```
@misc{salvadori2019huskarl,
    author = {Daniel Salvadori},
    title = {huskarl},
    year = {2019},
    publisher = {GitHub},
    journal = {GitHub repository},
    howpublished = {\url{https://github.com/danaugrs/huskarl}},
}
```

## About

_hùskarl_ in Old Norse means a warrior who works in his/her lord's service.
