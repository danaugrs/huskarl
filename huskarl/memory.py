from collections import namedtuple, deque
from itertools import groupby, count
import random


Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state'])


class Memory:
	"""Abstract base class for all implemented memories.

	Do not use this abstract base class directly but instead use one of the concrete memories implemented.

	A memory stores interaction sequences between an agent and one or multiple environments.
	To implement your own memory, you have to implement the following methods:
	"""    
	def put(self, transition, instance=0):
		raise NotImplementedError()

	def get(self, batch_size):
		raise NotImplementedError()

	def __len__(self):
		raise NotImplementedError()


def unpack_traces(traces):
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
		self.buffers = [TransitionBuffer(steps, steps, False) for i in range(instances)]
		self.steps = steps
		self.instances = instances

	def put(self, transition, instance=0):
		"""Stores transition into the appropriate buffer."""
		self.buffers[instance].push(transition)

	def get(self):
		"""Returns all traces and clears the memory."""
		return unpack_traces([tb.pop() for tb in self.buffers])

	def __len__(self):
		return sum([len(b) for b in self.buffers])        


class ExperienceReplay(Memory):
	"""Stores interaction with multiple environments.

	Provides efficient sampling of multistep traces.
	"""
	def __init__(self, capacity=10_000, steps=1, instances=1, exclude_boundaries=False):
		self.buffers = [TransitionBuffer(capacity, steps, exclude_boundaries) for i in range(instances)]
		self.capacity = capacity
		self.steps = steps
		self.exclude_boundaries = exclude_boundaries
		self.instances = instances

	def put(self, transition, instance=0):
		"""Stores transition into the appropriate buffer."""
		self.buffers[instance].push(transition)

	def get(self, batch_size):
		"""Samples traces from buffers uniformly."""
		if len(self.buffers) == 1:
			return unpack_traces(self.buffers[0].sample(batch_size))

		# Sample from each buffer according to its size
		batch_origins = random.choices(list(range(self.instances)),
			weights=[len(self.buffers[i]) for i in range(self.instances)], k=batch_size)
		batch_sizes = [len(list(group)) for key, group in groupby(batch_origins)]

		# Sample and combine traces from all buffers
		traces = []
		for i in range(self.instances):
			traces.extend(self.buffers[i].sample(batch_sizes[i]))
		return unpack_traces(traces)

	def __len__(self):
		return sum([len(b) for b in self.buffers])


class TransitionBuffer:
	"""Stores interaction with an environment as a double-ended queue of Transition instances.
	
	Provides efficient sampling of multistep traces.
	If exclude_boundaries==True, then traces are sampled such that they don't include episode boundaries.
	"""
	def __init__(self, capacity, steps=1, exclude_boundaries=False):
		"""
		Args:
			capacity (int): The maximum number of elements the buffer should be able to store.
			steps (int): The number of steps each sampled trace should include.
			exclude_boundaries (bool): If True, sampled traces will not include episode boundaries.
		"""
		self.buffer = deque(maxlen=capacity)
		self.capacity = capacity
		self.steps = steps

		self.exclude_boundaries = exclude_boundaries
		# If the user wants traces without episode boundaries
		if exclude_boundaries:
			# Then we need to keep track of more things to do it efficiently, in an incremental manner
			self.idxs = [[]] # Each element is a list (associated to an episode) of indexes of self.buffer 
			self.traces = [] # Each element is a list of size self.steps of indexes of self.buffer
		else:
			self.idxs = [] # Each element is an index of self.buffer
						   # It's best to store this instead of generating it every time the user samples traces

	def push(self, transition):
		"""Appends transition to the buffer."""
		self.buffer.append(transition)

		if self.exclude_boundaries:
			# If at max capacity then we need to remove the first index from self.idxs and reduce all indexes by 1
			# We might also need to remove the first trace from self.traces
			if len(self.buffer) == self.capacity:
				if len(self.idxs[0]) == 0: 
					self.idxs.pop(0) # Remove leading empty list from self.idxs
				if len(self.idxs[0]) >= self.steps:
					self.traces.pop(0) # Remove leading trace from self.traces
				self.idxs[0].pop(0) # Remove first index
				# Reduce all indexes by 1
				self.idxs = [[i-1 for i in episode] for episode in self.idxs] 
				self.traces = [[i-1 for i in episode] for episode in self.traces]

			# Append new index to self.idxs
			self.idxs[-1].append(len(self.buffer)-1)

			# Append new trace of length self.steps
			if len(self.idxs[-1]) >= self.steps:
				self.traces.append(self.idxs[-1][-self.steps:])

			if transition.state is None:
				# Episode is done - append a new list to self.idxs i.e. start a new episode
				self.idxs.append([])
		else:
			# Append new index to self.idxs
			self.idxs.append(len(self.buffer)-1)

	def sample(self, batch_size):
		"""Samples the specified number of traces from the buffer."""
		# Assemble list of all possible traces (using indexes)
		traces = self.traces if self.exclude_boundaries else list(zip(*[self.idxs[i:] for i in range(self.steps)]))
		# Sample from self.traces
		traces_idxs = random.sample(traces, batch_size)
		# Return traces with actual Transition instances based on indexes
		return [[self.buffer[i] for i in tidxs] for tidxs in traces_idxs]

	def pop(self):
		"""Returns and clears the entire buffer."""
		buf = list(self.buffer)
		self.buffer.clear()
		return buf

	def __len__(self):
		if self.exclude_boundaries:
			return len(self.traces)
		else:
			return max(0, len(self.buffer) - self.steps + 1)
