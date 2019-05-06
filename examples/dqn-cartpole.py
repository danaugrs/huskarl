from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import matplotlib.pyplot as plt
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

	# Create Deep Q-Learning Network agent
	agent = hk.agent.DQN(model, actions=dummy_env.action_space.n, nsteps=2)

	def plot_rewards(episode_rewards, episode_steps, done=False):
		plt.clf()
		plt.xlabel('Step')
		plt.ylabel('Reward')
		for ed, steps in zip(episode_rewards, episode_steps):
			plt.plot(steps, ed)
		plt.show() if done else plt.pause(0.001) # Pause a bit so that the graph is updated

	# Create simulation, train for a short period, and then test
	sim = hk.Simulation(create_env, agent)
	sim.train(max_steps=3000, visualize=True, plot=plot_rewards)
	sim.test(max_steps=1000)
