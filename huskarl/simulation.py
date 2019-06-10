from itertools import count
from collections import namedtuple
from queue import Empty
from time import sleep
import multiprocessing as mp

import numpy as np
import cloudpickle # For pickling lambda functions and more

from huskarl.memory import Transition
from huskarl.core import HkException


# Packet used to transmit experience from environment subprocesses to main process
# The first packet of every episode will have reward set to None
# The last packet of every episode will have state set to None
RewardState = namedtuple('RewardState', ['reward', 'state'])


class Simulation:
	"""Simulates an agent interacting with one of multiple environments."""
	def __init__(self, create_env, agent, mapping=None):
		self.create_env = create_env
		self.agent = agent
		self.mapping = mapping

	def train(self, max_steps=100_000, instances=1, visualize=False, plot=None, max_subprocesses=0):
		"""Trains the agent on the specified number of environment instances."""
		self.agent.training = True
		if max_subprocesses == 0:
			# Use single process implementation
			self._sp_train(max_steps, instances, visualize, plot)
		elif max_subprocesses is None or max_subprocesses > 0:
			# Use multiprocess implementation
			self._mp_train(max_steps, instances, visualize, plot, max_subprocesses)
		else:
			raise HkException(f"Invalid max_subprocesses setting: {max_subprocesses}")

	def _sp_train(self, max_steps, instances, visualize, plot):
		"""Trains using a single process."""
		# Keep track of rewards per episode per instance
		episode_reward_sequences = [[] for i in range(instances)]
		episode_step_sequences = [[] for i in range(instances)]
		episode_rewards = [0] * instances

		# Create and initialize environment instances
		envs = [self.create_env() for i in range(instances)]
		states = [env.reset() for env in envs]

		for step in range(max_steps):
			for i in range(instances):
				if visualize: envs[i].render()
				action = self.agent.act(states[i], i)
				next_state, reward, done, _ = envs[i].step(action)
				self.agent.push(Transition(states[i], action, reward, None if done else next_state), i)
				episode_rewards[i] += reward
				if done:
					episode_reward_sequences[i].append(episode_rewards[i])
					episode_step_sequences[i].append(step)
					episode_rewards[i] = 0
					if plot: plot(episode_reward_sequences, episode_step_sequences)
					states[i] = envs[i].reset()
				else:
					states[i] = next_state
			# Perform one step of the optimization
			self.agent.train(step)

		if plot: plot(episode_reward_sequences, episode_step_sequences, done=True)

	def _mp_train(self, max_steps, instances, visualize, plot, max_subprocesses):
		"""Trains using multiple processes.
		
		Useful to parallelize the computation of heavy environments.
		"""
		# Unless specified set the maximum number of processes to be the number of cores in the machine
		if max_subprocesses is None:
			max_subprocesses = mp.cpu_count()
		nprocesses = min(instances, max_subprocesses)

		# Split instances into processes as homogeneously as possibly
		instances_per_process = [instances//nprocesses] * nprocesses
		leftover = instances % nprocesses
		if leftover > 0:
			for i in range(leftover):
				instances_per_process[i] += 1

		# Create a unique id (index) for each instance, grouped by process
		instance_ids = [list(range(i, instances, nprocesses))[:ipp] for i, ipp in enumerate(instances_per_process)]

		# Create processes and pipes (one pipe for each environment instance)
		pipes = []
		processes = []
		for i in range(nprocesses):
			child_pipes = []
			for j in range(instances_per_process[i]):
				parent, child = mp.Pipe()
				pipes.append(parent)
				child_pipes.append(child)
			pargs = (cloudpickle.dumps(self.create_env), instance_ids[i], max_steps, child_pipes, visualize)
			processes.append(mp.Process(target=_train, args=pargs))

		# Start all processes
		print(f"Starting {nprocesses} process(es) for {instances} environment instance(s)... {instance_ids}")
		for p in processes: p.start()

		# Keep track of rewards per episode per instance
		episode_reward_sequences = [[] for i in range(instances)]
		episode_step_sequences = [[] for i in range(instances)]
		episode_rewards = [0] * instances	

		# Temporarily record RewardState instances received from each subprocess
		# Each Transition instance requires two RewardState instances to be created
		rss = [None] * instances

		# Keep track of last actions sent to subprocesses
		last_actions = [None] * instances

		for step in range(max_steps):
			
			# Keep track from which environments we have already constructed a full Transition instance
			# and sent it to agent. This is to synchronize steps.
			step_done = [False] * instances			

			while sum(step_done) < instances: # Steps across environments are synchronized

				# Within each step, Transitions are received and processed on a first-come first-served basis
				awaiting_pipes = [p for iid, p in enumerate(pipes) if step_done[iid] == 0]
				ready_pipes = mp.connection.wait(awaiting_pipes, timeout=None)
				pipe_indexes = [pipes.index(rp) for rp in ready_pipes]

				# Do a round-robin over processes to best divide computation
				pipe_indexes.sort()
				for iid in pipe_indexes:
					rs = pipes[iid].recv() # Receive a RewardState

					# If we already had a RewardState for this environment then we are able to create and push a Transition
					if rss[iid] is not None:
						exp = Transition(rss[iid].state, last_actions[iid], rs.reward, rs.state)
						self.agent.push(exp, iid)
						step_done[iid] = True
					rss[iid] = rs

					# Check if episode is done
					if rs.state is None:
						# Episode is done - store rewards and update plot
						rss[iid] = None
						episode_reward_sequences[iid].append(episode_rewards[iid])
						episode_step_sequences[iid].append(step)
						episode_rewards[iid] = 0
						if plot: plot(episode_reward_sequences, episode_step_sequences)
					else:
						# Episode is NOT done - act according to state and send action to the subprocess
						action = self.agent.act(rs.state, iid)
						last_actions[iid] = action
						try:
							pipes[iid].send(action)
						# Disregard BrokenPipeError on last step
						except BrokenPipeError as bpe:
							if step < (max_steps - 1): raise bpe
						if rs.reward: episode_rewards[iid] += rs.reward

			# Train the agent at the end of every synchronized step
			self.agent.train(step)

		if plot: plot(episode_reward_sequences, episode_step_sequences, done=True)

	def test(self, max_steps, visualize=True):
		"""Test the agent on the environment."""
		self.agent.training = False

		# Create and initialize environment
		env = self.create_env()
		state = env.reset()

		for step in range(max_steps):
			if visualize: env.render()
			action = self.agent.act(state)
			next_state, reward, done, _ = env.step(action)
			state = env.reset() if done else next_state


def _train(create_env, instance_ids, max_steps, pipes, visualize):
	"""This function is to be executed in a subprocess."""
	pipes = {iid: p for iid, p in zip(instance_ids, pipes)}
	actions = {iid: None for iid in instance_ids} # Reused dictionary of actions

	# Initialize environments and send initial state to agent in parent process
	create_env = cloudpickle.loads(create_env)
	envs = {iid: create_env() for iid in instance_ids}
	for iid in instance_ids:
		state = envs[iid].reset()
		pipes[iid].send(RewardState(None, state))

	# Run for the specified number of steps
	for step in range(max_steps):
		for iid in instance_ids:
			# Get action from agent in main process via pipe
			actions[iid] = pipes[iid].recv()
			if visualize: envs[iid].render()

			# Step environment and send experience to agent in main process via pipe
			next_state, reward, done, _ = envs[iid].step(actions[iid])
			pipes[iid].send(RewardState(reward, None if done else next_state))

			# If episode is over reset the environment and transmit initial state to agent
			if done:
				state = envs[iid].reset()
				pipes[iid].send(RewardState(None, state))
