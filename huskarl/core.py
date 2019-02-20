
class HkException(Exception):
	"""Basic exception for errors raised by Huskarl."""


class Agent:
	"""Abstract base class for all implemented agents.

	Do not use this abstract base class directly but instead use one of the concrete agents implemented.

	To implement your own agent, you have to implement the following methods:
	"""
	def save(self, filename, overwrite=False):
		"""Saves the model parameters to the specified file."""
		raise NotImplementedError()

	def act(self, state, instance=0):
		"""Returns the action to be taken given a state."""
		raise NotImplementedError()

	def push(self, transition, instance=0):
		"""Stores the transition in memory."""
		raise NotImplementedError()

	def train(self, step):
		"""Trains the agent for one step."""
		raise NotImplementedError()
