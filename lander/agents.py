"""
Contains the agents that can be used.
@author: Jesse Hagenaars
"""

import json
import pickle
from collections import deque

import gym
import numpy as np
import tensorflow as tf
from gym import logger
from gym.wrappers import Monitor


class RandomAgent:
    """
    Base/random class for RL agents.
    """

    def __init__(self, config):
        """
        Agent initialization.
        :param config:
        """

        # Running configuration
        self.run = 0
        self.step = 0
        self.episode = 0
        self.episode_count = config['EPISODES']

        # Env
        self.env_id = config['ENV_ID']
        self.env_seed = config['ENV_SEED']
        self.env = Monitor(gym.make(self.env_id), directory=config['RECORD_DIR'] + f'run_{self.run}',
                           video_callable=lambda episode_id: (episode_id + 1) % config['SAVE_EVERY'] == 0,
                           force=True, uid=config['AGENT'])  # record every nth episode, clear monitor files if present
        self.env.seed(self.env_seed)

        # Get random number generator
        self.prng = np.random.RandomState(self.env_seed)

        # Score/rewards over time
        # Deque allows quick appends and pops and has a max length
        self.score = deque(maxlen=self.episode_count)
        self.score_100 = deque(maxlen=100)  # for keeping track of mean of last 100

    def act(self, *args):
        """
        Perform a random action!
        :param args:
        :return:
        """
        return self.prng.randint(self.env.action_space.n)

    def do_episode(self, config):
        """

        :param config:
        :return:
        """

        # Initial values
        done = False
        score_e = 0
        step_e = 0

        # Reset environment
        self.env.reset()

        # Continue while not crashed
        while not done:

            # Show on screen
            if config['RENDER']:
                self.env.render()

            # Act
            action = self.act()
            _, reward, done, _ = self.env.step(action)

            # Increment score and steps
            score_e += reward
            step_e += 1
            self.step += 1

        # Append score
        self.score.append(score_e)
        self.score_100.append(score_e)
        mean_score = np.mean(self.score_100)

        # Increment episode
        self.episode += 1

        logger.info(f'[Episode {self.episode}] - score: {score_e:.2f}, steps: {step_e}, 100-score: {mean_score:.2f}.')

    def save_checkpoint(self, config):
        """

        :param config:
        :return:
        """

        # Env can't be saved
        dummy_env = self.env
        self.env = None

        # Save checkpoint
        with open(config['RECORD_DIR'] + 'checkpoint.pickle', 'wb') as p_file:
            pickle.dump(self, p_file)

        # Save config
        with open(config['RECORD_DIR'] + 'config.json', 'w') as c_file:
            json.dump(config, c_file, sort_keys=True, indent=4)

        # Put env back
        self.env = dummy_env


class SarsaAgent(RandomAgent):
    """
    Agent that makes use of Sarsa (on-policy TD control).
    """

    def __init__(self, config):
        """

        :param config:
        """

        # Initialize base class
        super().__init__(config)

        # State
        self.state_bounds = config['STATE_BOUNDS']
        self.state_bins = tuple(config['STATE_BINS'])

        # Float conversion
        for i, lr in enumerate(config['LEARNING_RATE']):
            if type(lr) is str:
                config['LEARNING_RATE'][i] = float(lr)
        for i, eps in enumerate(config['E_GREEDY']):
            if type(eps) is str:
                config['E_GREEDY'][i] = float(eps)

        # Learning parameters
        # First linear decay, then exponential decay
        self.alpha_start, self.alpha_end, self.alpha_steps, self.alpha_decay = config['LEARNING_RATE']
        self.epsilon_start, self.epsilon_end, self.epsilon_steps, self.epsilon_decay = config['E_GREEDY']
        self.alpha, self.epsilon = self.alpha_start, self.epsilon_start
        self.gamma = float(config['DISCOUNT_RATE'])

        # Q-table
        self.q_table = self.prng.uniform(low=-1.0, high=1.0, size=self.state_bins + (self.env.action_space.n,))

    def act(self, state):
        """

        :param state:
        :return:
        """

        if self.prng.random_sample() < self.epsilon:
            return self.prng.randint(self.env.action_space.n)
        else:
            return np.argmax(self.q_table[state])

    def discretize_state(self, state):
        """

        :param state:
        :return:
        """

        # First calculate the ratios, then convert to bin indices
        ratios = [(state[i] + abs(self.state_bounds[i][0])) / (self.state_bounds[i][1] - self.state_bounds[i][0]) for i
                  in range(len(state))]
        state_d = [int(round((self.state_bins[i] - 1) * ratios[i])) for i in range(len(state))]
        state_d = [min(self.state_bins[i] - 1, max(0, state_d[i])) for i in range(len(state))]

        return tuple(state_d)

    def learn(self, done, state, action, reward, state_, action_):
        """

        :param done:
        :param state:
        :param action:
        :param reward:
        :param state_:
        :param action_:
        :return:
        """

        # Get current Q(s, a)
        q_value = self.q_table[state][action]

        # Check if next state is terminal, get next Q(s', a')
        if not done:
            q_value_ = reward + self.gamma * self.q_table[state_][action_]
        else:
            q_value_ = reward

        # Update current Q(s, a)
        self.q_table[state][action] += self.alpha * (q_value_ - q_value)

    def do_episode(self, config):
        """

        :param config:
        :return:
        """

        # Initial values
        done = False
        score_e = 0
        step_e = 0

        # Get epsilon for initial state
        self.update_epsilon_step()

        # Episodic decay (only after linear decay)
        self.update_alpha_episode()
        self.update_epsilon_episode()

        # Get current state s, act based on s
        state = self.discretize_state(self.env.reset())
        action = self.act(state)

        # Continue while not crashed
        while not done:

            # Show on screen
            if config['RENDER']:
                self.env.render()

            # Update for other steps
            self.update_alpha_step()
            self.update_epsilon_step()

            # Get next state s' and reward, act based on s'
            state_, reward, done, _ = self.env.step(action)
            state_ = self.discretize_state(state_)
            action_ = self.act(state_)

            # Learn
            self.learn(done, state, action, reward, state_, action_)

            # Set next state and action to current
            state = state_
            action = action_

            # Increment score and steps
            score_e += reward
            step_e += 1
            self.step += 1

        # Append score
        self.score.append(score_e)
        self.score_100.append(score_e)
        mean_score = np.mean(self.score_100)

        # Increment episode
        self.episode += 1

        logger.info(f'[Episode {self.episode}] - score: {score_e:.2f}, steps: {step_e}, e: {self.epsilon:.4f}, '
                    f'a: {self.alpha:.4f}, 100-score: {mean_score:.2f}.')

    def update_alpha_step(self):
        """

        :return:
        """

        # Linear decay
        if self.step <= self.alpha_steps and self.alpha_steps > 0:
            self.alpha = self.alpha_start - self.step * (self.alpha_start - self.alpha_end) / self.alpha_steps

    def update_epsilon_step(self):
        """

        :return:
        """

        # Linear decay
        if self.step <= self.epsilon_steps and self.epsilon_steps > 0:
            self.epsilon = self.epsilon_start - self.step * (self.epsilon_start - self.epsilon_end) / self.epsilon_steps

    def update_alpha_episode(self):
        """

        :return:
        """

        # Exponential decay
        if self.step > self.alpha_steps:
            self.alpha *= self.alpha_decay

    def update_epsilon_episode(self):
        """

        :return:
        """

        # Exponential decay
        if self.step > self.epsilon_steps:
            self.epsilon *= self.epsilon_decay


class QAgent(SarsaAgent):
    """
    Agent that makes use of Q-learning (off-policy TD control).
    """

    def __init__(self, config):
        """

        :param config:
        """
        super().__init__(config)

    def learn(self, done, state, action, reward, state_, action_=None):
        """

        :param done:
        :param state:
        :param action:
        :param reward:
        :param state_:
        :param action_:
        :return:
        """

        # Get current Q(s, a)
        q_value = self.q_table[state][action]

        # Check if next state is terminal, get next maximum Q-value
        if not done:
            q_value_ = reward + self.gamma * max(self.q_table[state_])
        else:
            q_value_ = reward

        # Update current Q(s, a)
        self.q_table[state][action] += self.alpha * (q_value_ - q_value)

    def do_episode(self, config):
        """

        :param config:
        :return:
        """

        # Initial values
        done = False
        score_e = 0
        step_e = 0

        # Episodic decay (only after linear decay)
        self.update_alpha_episode()
        self.update_epsilon_episode()

        # Get current state s
        state = self.discretize_state(self.env.reset())

        # Continue while not crashed
        while not done:

            # Show on screen
            if config['RENDER']:
                self.env.render()

            # Get learning parameters
            self.update_alpha_step()
            self.update_epsilon_step()

            # Act based on current state s
            action = self.act(state)
            state_, reward, done, _ = self.env.step(action)
            state_ = self.discretize_state(state_)

            # Learn
            self.learn(done, state, action, reward, state_)

            # Set next state to current
            state = state_

            # Increment score and steps
            score_e += reward
            step_e += 1
            self.step += 1

        # Append score
        self.score.append(score_e)
        self.score_100.append(score_e)
        mean_score = np.mean(self.score_100)

        # Increment episode
        self.episode += 1

        logger.info(f'[Episode {self.episode}] - score: {score_e:.2f}, steps: {step_e}, e: {self.epsilon:.4f}, '
                    f'a: {self.alpha:.4f}, 100-score: {mean_score:.2f}.')


class DoubleDQNAgent(QAgent):
    """
    Agent that makes use of double deep Q-learning, where Q(s, a) is approximated using neural networks.
    """

    def __init__(self, config):
        """

        :param config:
        """

        # Initialize base class
        super().__init__(config)

        # Regularization
        self.l2_reg = float(config['L2_REG'])

        # Replay memory
        self.replay_memory = deque(maxlen=int(float(config['REPLAY_MEMORY_SIZE'])))

        # Network configuration
        self.layers = config['LAYER_SIZES']
        self.batch_size = config['BATCH_SIZE']

        # Set random seed for TF
        # TODO: is this the correct place for seed?
        tf.set_random_seed(self.env_seed)

        # Also episode as TF variable
        self.tf_episode = tf.get_variable('episode', shape=(), dtype=tf.int32, trainable=False,
                                          initializer=tf.zeros_initializer)

        # Initialize placeholders
        self.initialize_placeholders()

        # Build Q-network
        with tf.variable_scope('q_network'):
            # Not actually two different networks, since they both use the same weights
            # NOTE: stop_gradient prevents network from contributing to gradient computation
            self.q_network = self.build_network(self.ph_state,
                                                regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg),
                                                trainable=True)  # current state s
            self.q_network_ = tf.stop_gradient(self.build_network(self.ph_state_, reuse=True))  # next state s'

        # Build target network
        with tf.variable_scope('target_network', reuse=False):
            self.target_network = tf.stop_gradient(self.build_network(self.ph_state_))  # target for next state s'

        # Initialize operations
        self.initialize_ops()

        # Create session
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def initialize_placeholders(self):
        """

        :return:
        """

        # Placeholders are needed for feeding data to the networks
        # NOTE: None indicates the batch dimension, which can be any size (determined later)
        self.ph_done = tf.placeholder(tf.float32, shape=(None,))  # float since multiplication will give error otherwise
        self.ph_state = tf.placeholder(tf.float32, shape=(None,) + self.env.observation_space.shape)  # s
        self.ph_action = tf.placeholder(tf.int32, shape=(None,))  # a
        self.ph_reward = tf.placeholder(tf.float32, shape=(None,))  # r
        self.ph_state_ = tf.placeholder(tf.float32, shape=(None,) + self.env.observation_space.shape)  # s'

        # Combine them in a list (comes in handy when feeding dicts)
        self.ph_list = [self.ph_done, self.ph_state, self.ph_action, self.ph_reward, self.ph_state_]

    def initialize_ops(self):
        """

        :return:
        """

        # Episode increment op
        self.episode_op = self.tf_episode.assign_add(1)

        # Separate variables of Q-network and target network
        # NOTE: we use trainable variables to only select from the Q-network for the current state
        v_q_network = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='q_network')
        v_target_network = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_network')

        # Build op for updating the target network based on the Q-network for the current state
        update_target_op = []
        for i, v_target in enumerate(v_target_network):
            update_target_op.append(v_target.assign(v_q_network[i]))
        # Group together
        self.update_target_op = tf.group(*update_target_op, name='update_target')  # * unpacks the list

        # Build training op
        self.training_op = self.train()

    def build_network(self, ph_input, regularizer=None, trainable=False, reuse=False):
        """

        :param ph_input:
        :param regularizer:
        :param trainable:
        :param reuse:
        :return:
        """

        # Fully-connected network with 3 hidden layers
        h1 = tf.layers.dense(ph_input, self.layers[0], activation=tf.nn.relu, kernel_regularizer=regularizer,
                             trainable=trainable, reuse=reuse, name='dense1')
        h2 = tf.layers.dense(h1, self.layers[1], activation=tf.nn.relu, kernel_regularizer=regularizer,
                             trainable=trainable, reuse=reuse, name='dense2')
        h3 = tf.layers.dense(h2, self.layers[2], activation=tf.nn.relu, kernel_regularizer=regularizer,
                             trainable=trainable, reuse=reuse, name='dense3')

        # Output layer (squeeze removes dimensions of 1)
        Q = tf.squeeze(tf.layers.dense(h3, self.env.action_space.n, kernel_regularizer=regularizer,
                                       trainable=trainable, reuse=reuse, name='dense4'))

        return Q

    def act(self, state):
        """

        :param state:
        :return:
        """

        if self.prng.random_sample() < self.epsilon:
            return self.prng.randint(self.env.action_space.n)
        else:
            # NOTE: None is used as index to put another list around it
            return np.argmax(self.sess.run(self.q_network, feed_dict={self.ph_state: state[None]}))

    def remember(self, done, state, action, reward, state_):
        """

        :param done:
        :param state:
        :param action:
        :param reward:
        :param state_:
        :return:
        """

        # Negate done because of the multiplication in the training function
        self.replay_memory.append((0.0 if done else 1.0, state, action, reward, state_))

    def train(self):
        """

        :return:
        """

        # Learning targets based on experience replay
        # Q(s, a) = r if s' terminal
        # Q(s, a) = r + gamma * Q_target(s', argmax_{a} Q(s', a)) else
        # NOTE: using Q_target(s', argmax_{a} Q(s', a)) instead of max_{a} Q_target(s', a) makes it double Q-learning
        # EXPLANATION: first take argmax of next Q, cast this to an int, stack it with a sample index and
        #   then gather Q-values from the target network based on these indices
        q_target = self.ph_reward + self.ph_done * self.gamma * tf.gather_nd(self.target_network, tf.stack(
            (tf.range(self.batch_size), tf.cast(tf.argmax(self.q_network_, axis=1), tf.int32)), axis=1))

        # Now do the same for Q-values from the Q-network (the one that we actually train)
        # Not the actions we 'wanted' to take (targets), but the actions we would have taken
        q_taken = tf.gather_nd(self.q_network, tf.stack((tf.range(self.batch_size), self.ph_action), axis=1))

        # Compute loss based on the mean squared error between the two
        loss = tf.losses.mean_squared_error(q_target, q_taken)

        # Add L2 regularization
        loss += 0.5 * tf.losses.get_regularization_loss()

        # Get train op
        train_op = tf.train.AdamOptimizer(self.alpha * self.alpha_decay ** self.episode).minimize(loss)

        return train_op

    def get_batch(self):
        """

        :return:
        """

        # First create random sample, then select from memory
        sample = self.prng.randint(len(self.replay_memory), size=self.batch_size)
        minibatch = [self.replay_memory[i] for i in sample]

        return minibatch

    def do_episode(self, config):
        """

        :param config:
        :return:
        """

        # Initial values
        done = False
        score_e = 0
        step_e = 0

        # Episodic decay (only after linear decay)
        self.update_epsilon_episode()

        # Get current state s
        state = self.env.reset()

        # Continue while not crashed
        while not done:

            # Show on screen
            if config['RENDER']:
                self.env.render()

            # Get learning parameters
            self.update_epsilon_step()

            # Act based on current state s
            action = self.act(state)
            state_, reward, done, _ = self.env.step(action)

            # Add to memory
            self.remember(done, state, action, reward, state_)

            # Set weights of target network to Q-network
            if self.step % config['UPDATE_EVERY'] == 0:
                self.sess.run(self.update_target_op)

            # Train
            if len(self.replay_memory) >= self.batch_size:
                minibatch = self.get_batch()
                self.sess.run(self.training_op,
                              feed_dict={ph: data for ph, data in zip(self.ph_list, map(list, zip(*minibatch)))})

            # Set next state to current
            state = state_

            # Increment score and steps
            score_e += reward
            step_e += 1
            self.step += 1

        # Append score
        self.score.append(score_e)
        self.score_100.append(score_e)
        mean_score = np.mean(self.score_100)

        # Increment episode
        self.episode += 1
        self.sess.run(self.episode_op)

        logger.info(f'[Episode {self.episode}] - score: {score_e:.2f}, steps: {step_e}, e: {self.epsilon:.4f}, '
                    f'100-score: {mean_score:.2f}.')

    def save_checkpoint(self, config):
        """

        :param config:
        :return:
        """

        # Saver and save
        saver = tf.train.Saver()
        saver.save(self.sess, config['RECORD_DIR'] + 'model')

        # TODO: fix saving for TF!
        # Execute save function of base class
        # super().save_checkpoint(config)
