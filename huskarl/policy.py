import random
import numpy as np
from scipy.stats import truncnorm


class Policy:
	"""Abstract base class for all implemented policies.
	
	Do not use this abstract base class directly but instead use one of the concrete policies implemented.

	A policy ultimately returns the action to be taken based on the output of the agent.
	The policy is the place to implement action-space exploration strategies.
	If the action space is discrete, the policy usually receives action values and has to pick an action/index.
	A discrete action-space policy can explore by pick an action at random with a small probability e.g. EpsilonGreedy.
	If the action space is continuous, the policy usually receives a single action or a distribution over actions.
	A continuous action-space policy can simply sample from the distribution and/or add noise to the received action.	
	
	To implement your own policy, you have to implement the following method:
	"""
	def act(self, **kwargs):
		raise NotImplementedError()


# Discrete action-space policies =======================================================================================


class Greedy(Policy):
	"""Greedy Policy

	This policy always picks the action with largest value.
	"""
	def act(self, qvals):
		return np.argmax(qvals)


class EpsGreedy(Policy):
	"""Epsilon-Greedy Policy
	
	This policy picks the action with largest value with probability 1-epsilon.
	It picks a random action and therefore explores with probability epsilon.
	"""
	def __init__(self, eps):
		self.eps = eps

	def act(self, qvals):
		if random.random() > self.eps:
			return np.argmax(qvals)
		return random.randrange(len(qvals))


class GaussianEpsGreedy(Policy):
	"""Gaussian Epsilon-Greedy Policy

	Like the Epsilon-Greedy Policy except it samples epsilon from a [0,1]-truncated Gaussian distribution.
	This method is used in "Asynchronous Methods for Deep Reinforcement Learning" (Mnih et al., 2016).
	"""
	def __init__(self, eps_mean, eps_std):
		self.eps_mean = eps_mean
		self.eps_std = eps_std
	
	def act(self, qvals):
		eps = truncnorm.rvs((0 - self.eps_mean) / self.eps_std, (1 - self.eps_mean) / self.eps_std)
		if random.random() > eps:
			return np.argmax(qvals)
		return random.randrange(len(qvals))


# Continuous action-space policies (noise generators) ==================================================================


class PassThrough(Policy):
	"""Pass-Through Policy

	This policy simply outputs the model's output, unchanged.
	"""
	def act(self, action):
		return action

