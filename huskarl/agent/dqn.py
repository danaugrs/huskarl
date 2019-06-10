from tensorflow.keras.layers import Dense, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import numpy as np

from huskarl.policy import EpsGreedy, Greedy
from huskarl.core import Agent, HkException
from huskarl import memory


class DQN(Agent):
	"""Deep Q-Learning Network

	Base implementation:
		"Playing Atari with Deep Reinforcement Learning" (Mnih et al., 2013)

	Extensions:
		Multi-step returns: "Reinforcement Learning: An Introduction" 2nd ed. (Sutton & Barto, 2018)
		Double Q-Learning: "Deep Reinforcement Learning with Double Q-learning" (van Hasselt et al., 2015)
		Dueling Q-Network: "Dueling Network Architectures for Deep Reinforcement Learning" (Wang et al., 2016)
	"""
	def __init__(self, model, actions, optimizer=None, policy=None, test_policy=None,
				 memsize=10_000, target_update=10, gamma=0.99, batch_size=64, nsteps=1,
				 enable_double_dqn=True, enable_dueling_network=False, dueling_type='avg'):
		"""
		TODO: Describe parameters
		"""
		self.actions = actions
		self.optimizer = Adam(lr=3e-3) if optimizer is None else optimizer

		self.policy = EpsGreedy(0.1) if policy is None else policy
		self.test_policy = Greedy() if test_policy is None else test_policy

		self.memsize = memsize
		self.memory = memory.PrioritizedExperienceReplay(memsize, nsteps)

		self.target_update = target_update
		self.gamma = gamma
		self.batch_size = batch_size
		self.nsteps = nsteps
		self.training = True

		# Extension options
		self.enable_double_dqn = enable_double_dqn
		self.enable_dueling_network = enable_dueling_network
		self.dueling_type = dueling_type

		# Create output layer based on number of actions and (optionally) a dueling architecture
		raw_output = model.layers[-1].output
		if self.enable_dueling_network:
			# "Dueling Network Architectures for Deep Reinforcement Learning" (Wang et al., 2016)
			# Output the state value (V) and the action-specific advantages (A) separately then compute the Q values: Q = A + V
			dueling_layer = Dense(self.actions + 1, activation='linear')(raw_output)
			if   self.dueling_type == 'avg':   f = lambda a: tf.expand_dims(a[:,0], -1) + a[:,1:] - tf.reduce_mean(a[:,1:], axis=1, keepdims=True)
			elif self.dueling_type == 'max':   f = lambda a: tf.expand_dims(a[:,0], -1) + a[:,1:] - tf.reduce_max(a[:,1:], axis=1, keepdims=True)
			elif self.dueling_type == 'naive': f = lambda a: tf.expand_dims(a[:,0], -1) + a[:,1:]
			else: raise HkException("dueling_type must be one of {'avg','max','naive'}")
			output_layer = Lambda(f, output_shape=(self.actions,))(dueling_layer)
		else:
			output_layer = Dense(self.actions, activation='linear')(raw_output)

		self.model = Model(inputs=model.input, outputs=output_layer)

		# Define loss function that computes the MSE between target Q-values and cumulative discounted rewards
		# If using PrioritizedExperienceReplay, the loss function also computes the TD error and updates the trace priorities
		def masked_q_loss(data, y_pred):
			"""Computes the MSE between the Q-values of the actions that were taken and	the cumulative discounted
			rewards obtained after taking those actions. Updates trace priorities if using PrioritizedExperienceReplay.
			"""
			action_batch, target_qvals = data[:, 0], data[:, 1]
			seq = tf.cast(tf.range(0, tf.shape(action_batch)[0]), tf.int32)
			action_idxs = tf.transpose(tf.stack([seq, tf.cast(action_batch, tf.int32)]))
			qvals = tf.gather_nd(y_pred, action_idxs)
			if isinstance(self.memory, memory.PrioritizedExperienceReplay):
				def update_priorities(_qvals, _target_qvals, _traces_idxs):
					"""Computes the TD error and updates memory priorities."""
					td_error = np.abs((_target_qvals - _qvals).numpy())
					_traces_idxs = (tf.cast(_traces_idxs, tf.int32)).numpy()
					self.memory.update_priorities(_traces_idxs, td_error)
					return _qvals
				qvals = tf.py_function(func=update_priorities, inp=[qvals, target_qvals, data[:,2]], Tout=tf.float32)
			return tf.keras.losses.mse(qvals, target_qvals)

		self.model.compile(optimizer=self.optimizer, loss=masked_q_loss)

		# Clone model to use for delayed Q targets
		self.target_model = tf.keras.models.clone_model(self.model)
		self.target_model.set_weights(self.model.get_weights())

	def save(self, filename, overwrite=False):
		"""Saves the model parameters to the specified file."""
		self.model.save_weights(filename, overwrite=overwrite)

	def act(self, state, instance=0):
		"""Returns the action to be taken given a state."""
		qvals = self.model.predict(np.array([state]))[0]
		return self.policy.act(qvals) if self.training else self.test_policy.act(qvals)

	def push(self, transition, instance=0):
		"""Stores the transition in memory."""
		self.memory.put(transition)

	def train(self, step):
		"""Trains the agent for one step."""
		if len(self.memory) == 0:
			return

		# Update target network
		if self.target_update >= 1 and step % self.target_update == 0:
			# Perform a hard update
			self.target_model.set_weights(self.model.get_weights())
		elif self.target_update < 1:
			# Perform a soft update
			mw = np.array(self.model.get_weights())
			tmw = np.array(self.target_model.get_weights())
			self.target_model.set_weights(self.target_update * mw + (1 - self.target_update) * tmw)

		# Train even when memory has fewer than the specified batch_size
		batch_size = min(len(self.memory), self.batch_size)

		# Sample batch_size traces from memory
		state_batch, action_batch, reward_batches, end_state_batch, not_done_mask = self.memory.get(batch_size)

		# Compute the value of the last next states
		target_qvals = np.zeros(batch_size)
		non_final_last_next_states = [es for es in end_state_batch if es is not None]

		if len(non_final_last_next_states) > 0:		
			if self.enable_double_dqn:
				# "Deep Reinforcement Learning with Double Q-learning" (van Hasselt et al., 2015)
				# The online network predicts the actions while the target network is used to estimate the Q-values
				q_values = self.model.predict_on_batch(np.array(non_final_last_next_states))
				actions = np.argmax(q_values, axis=1)
				# Estimate Q-values using the target network but select the values with the
				# highest Q-value wrt to the online model (as computed above).
				target_q_values = self.target_model.predict_on_batch(np.array(non_final_last_next_states))
				selected_target_q_vals = target_q_values[range(len(target_q_values)), actions]
			else:
				# Use delayed target network to compute target Q-values
				selected_target_q_vals = self.target_model.predict_on_batch(np.array(non_final_last_next_states)).max(1)
			non_final_mask = list(map(lambda s: s is not None, end_state_batch))
			target_qvals[non_final_mask] = selected_target_q_vals

		# Compute n-step discounted return
		# If episode ended within any sampled nstep trace - zero out remaining rewards
		for n in reversed(range(self.nsteps)):
			rewards = np.array([b[n] for b in reward_batches])
			target_qvals *= np.array([t[n] for t in not_done_mask])
			target_qvals = rewards + (self.gamma * target_qvals)

		# Compile information needed by the custom loss function
		loss_data = [action_batch, target_qvals]

		# If using PrioritizedExperienceReplay then we need to provide the trace indexes
		# to the loss function as well so we can update the priorities of the traces
		if isinstance(self.memory, memory.PrioritizedExperienceReplay):
			loss_data.append(self.memory.last_traces_idxs())

		# Train model
		self.model.train_on_batch(np.array(state_batch), np.stack(loss_data).transpose())
