from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import matplotlib.pyplot as plt
import numpy as np
import gym

import huskarl as hk

if __name__ == '__main__':

	# Setup gym environment
	create_env = lambda: gym.make('CartPole-v0').unwrapped
	dummy_env = create_env()

	# Build a simple neural network with 3 fully connected layers as our model
	model = Sequential([
		Dense(16, activation='relu', input_shape=dummy_env.observation_space.shape),
		Dense(16, activation='relu'),
		Dense(16, activation='relu'),
	])

	# We will be running multiple concurrent environment instances
	instances = 16

	# Create a policy for each instance with a different distribution for epsilon
	policy = [hk.policy.Greedy()] + [hk.policy.GaussianEpsGreedy(eps, 0.1) for eps in np.arange(0, 1, 1/(instances-1))]

	# Create Advantage Actor-Critic agent
	agent = hk.agent.A2C(model, actions=dummy_env.action_space.n, nsteps=2, instances=instances, policy=policy)

	def plot_rewards(episode_rewards, episode_steps, done=False):
		plt.clf()
		plt.xlabel('Step')
		plt.ylabel('Reward')
		for i, (ed, steps) in enumerate(zip(episode_rewards, episode_steps)):
			plt.plot(steps, ed, alpha=0.5 if i == 0 else 0.2, linewidth=2 if i == 0 else 1)
		plt.show() if done else plt.pause(0.001) # Pause a bit so that the graph is updated
		
	# Create simulation, train for a short period, and then test
	sim = hk.Simulation(create_env, agent)
	sim.train(max_steps=5000, instances=instances, plot=plot_rewards)
	sim.test(max_steps=1000)
