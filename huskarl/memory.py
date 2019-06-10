from collections import namedtuple, deque
import random
import numpy as np


Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state'])


class Memory:
	"""Abstract base class for all implemented memories.

	Do not use this abstract base class directly but instead use one of the concrete memories implemented.

	A memory stores interaction sequences between an agent and one or multiple environments.
	To implement your own memory, you have to implement the following methods:
	"""
	def put(self, *args, **kwargs):
		raise NotImplementedError()

	def get(self, *args, **kwargs):
		raise NotImplementedError()

	def __len__(self):
		raise NotImplementedError()


def unpack(traces):
	"""Returns states, actions, rewards, end_states, and a mask for episode boundaries given traces."""
	states = [t[0].state for t in traces]
	actions = [t[0].action for t in traces]
	rewards = [[e.reward for e in t] for t in traces]
	end_states = [t[-1].next_state for t in traces]
	not_done_mask = [[1 if n.next_state is not None else 0 for n in t] for t in traces]
	return states, actions, rewards, end_states, not_done_mask


class OnPolicy(Memory):
	"""Stores multiple steps of interaction with multiple environments."""
	def __init__(self, steps=1, instances=1):
		self.buffers = [[] for _ in range(instances)]
		self.steps = steps
		self.instances = instances

	def put(self, transition, instance=0):
		"""Stores transition into the appropriate buffer."""
		self.buffers[instance].append(transition)

	def get(self):
		"""Returns all traces and clears the memory."""
		traces = [list(tb) for tb in self.buffers]
		self.buffers = [[] for _ in range(self.instances)]
		return unpack(traces)

	def __len__(self):
		"""Returns the number of traces stored."""
		return sum([len(b) - self.steps + 1 for b in self.buffers])


class ExperienceReplay:
	"""Stores interaction with an environment as a double-ended queue of Transition instances.
	
	Provides efficient sampling of multistep traces.
	If exclude_boundaries==True, then traces are sampled such that they don't include episode boundaries.
	"""
	def __init__(self, capacity, steps=1, exclude_boundaries=False):
		"""
		Args:
			capacity (int): The maximum number of traces the memory should be able to store.
			steps (int): The number of steps (transitions) each sampled trace should include.
			exclude_boundaries (bool): If True, sampled traces will not include episode boundaries.
		"""
		self.traces = deque(maxlen=capacity)
		self.buffer = [] # Rolling buffer of size at most self.steps
		self.capacity = capacity
		self.steps = steps
		self.exclude_boundaries = exclude_boundaries

	def put(self, transition):
		"""Adds transition to memory."""
		# Append transition to temporary rolling buffer
		self.buffer.append(transition)
		# If buffer doesn't yet contain a full trace - return
		if len(self.buffer) < self.steps: return
		# If self.traces not at max capacity, append new trace and priority (use highest existing priority if available)
		self.traces.append(tuple(self.buffer))
		# If excluding boundaries and we've reached a boundary - clear the buffer
		if self.exclude_boundaries and transition.next_state is None:
			self.buffer = []
			return
		# Roll buffer
		self.buffer = self.buffer[1:]

	def get(self, batch_size):
		"""Samples the specified number of traces uniformly from the buffer."""
		# Sample batch_size traces
		traces = random.sample(self.traces, batch_size)
		return unpack(traces)

	def __len__(self):
		"""Returns the number of traces stored."""
		return len(self.traces)


EPS = 1e-3 # Constant added to all priorities to prevent them from being zero


class PrioritizedExperienceReplay:
	"""Stores prioritized interaction with an environment in a priority queue implemented via a heap.

	Provides efficient prioritized sampling of multistep traces.
	If exclude_boundaries==True, then traces are sampled such that they don't include episode boundaries.
	For more information see "Prioritized Experience Replay" (Schaul et al., 2016).
	"""
	def __init__(self, capacity, steps=1, exclude_boundaries=False, prob_alpha=0.6):
		"""
		Args:
			capacity (int): The maximum number of traces the memory should be able to store.
			steps (int): The number of steps (transitions) each sampled trace should include.
			exclude_boundaries (bool): If True, sampled traces will not include episode boundaries.
			prob_alpha (float): Value between 0 and 1 that specifies how strongly priorities are taken into account.
		"""
		self.traces = [] # Each element is a tuple containing self.steps transitions
		self.priorities = np.array([]) # Each element is the priority for the same-index trace in self.traces
		self.buffer = [] # Rolling buffer of size at most self.steps
		self.capacity = capacity
		self.steps = steps
		self.exclude_boundaries = exclude_boundaries
		self.prob_alpha = prob_alpha
		self.traces_idxs = [] # Temporary list that contains the indexes associated to the last retrieved traces

	def put(self, transition):
		"""Adds transition to memory."""
		# Append transition to temporary rolling buffer
		self.buffer.append(transition)
		# If buffer doesn't yet contain a full trace - return
		if len(self.buffer) < self.steps: return
		# If self.traces not at max capacity, append new trace and priority (use highest existing priority if available)
		if len(self.traces) < self.capacity:
			self.traces.append(tuple(self.buffer))
			self.priorities = np.append(self.priorities, EPS if self.priorities.size == 0 else self.priorities.max())
		else:
			# If self.traces at max capacity, substitute lowest priority trace and use highest existing priority
			idx = np.argmin(self.priorities)
			self.traces[idx] = tuple(self.buffer)
			self.priorities[idx] = self.priorities.max()
		# If excluding boundaries and we've reached a boundary - clear the buffer
		if self.exclude_boundaries and transition.next_state is None:
			self.buffer = []
			return
		# Roll buffer
		self.buffer = self.buffer[1:]

	def get(self, batch_size):
		"""Samples the specified number of traces from the buffer according to the prioritization and prob_alpha."""
		# Transform priorities into probabilities using self.prob_alpha
		probs = self.priorities ** self.prob_alpha
		probs /= probs.sum()
		# Sample batch_size traces according to probabilities and store indexes
		self.traces_idxs = np.random.choice(len(self.traces), batch_size, p=probs, replace=False)
		traces = [self.traces[idx] for idx in self.traces_idxs]
		return unpack(traces)

	def last_traces_idxs(self):
		"""Returns the indexes associated with the last retrieved traces."""
		return self.traces_idxs.copy()

	def update_priorities(self, traces_idxs, new_priorities):
		"""Updates the priorities of the traces with specified indexes."""
		self.priorities[traces_idxs] = new_priorities + EPS

	def __len__(self):
		"""Returns the number of traces stored."""
		return len(self.traces)
