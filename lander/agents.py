"""
Contains the agents that can be used.
@author: Jesse Hagenaars
"""

import pickle
from collections import deque

import gym
import numpy as np
from gym import logger
from gym.wrappers import Monitor
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam
from keras.regularizers import l2


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
        self.step = 0
        self.episode = 0
        self.episode_count = config['EPISODES']

        # Env
        self.env_id = config['ENV_ID']
        self.env_seed = config['ENV_SEED']
        self.env = Monitor(gym.make(self.env_id), directory=config['RECORD_DIR'],
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

        logger.info(
            f'[Episode {self.episode}] - score: {score_e}, steps: {step_e}, mean score (100 ep.): {mean_score}.')

    def save_checkpoint(self, config):
        """

        :param config:
        :return:
        """

        # Env can't be saved
        dummy_env = self.env
        self.env = None

        # Save checkpoint
        pickle.dump(self, open(config['RECORD_DIR'] + 'checkpoint.pickle', 'wb'))

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
        self.gamma = float(config['DISCOUNT_RATE'])

        # Q-table
        self.q_table = self.prng.uniform(low=-1.0, high=1.0, size=self.state_bins + (self.env.action_space.n,))

    def act(self, state, epsilon):
        """

        :param state:
        :param epsilon:
        :return:
        """

        if self.prng.random_sample() < epsilon:
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

    def learn(self, done, alpha, state, action, reward, state_, action_):
        """

        :param done:
        :param alpha:
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
        self.q_table[state][action] += alpha * (q_value_ - q_value)

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
        epsilon = self.get_epsilon()

        # Get current state s, act based on s
        state = self.discretize_state(self.env.reset())
        action = self.act(state, epsilon)

        # Continue while not crashed
        while not done:

            # Show on screen
            if config['RENDER']:
                self.env.render()

            # Update for other steps
            alpha = self.get_alpha()
            epsilon = self.get_epsilon()

            # Get next state s' and reward, act based on s'
            state_, reward, done, _ = self.env.step(action)
            state_ = self.discretize_state(state_)
            action_ = self.act(state_, epsilon)

            # Learn
            self.learn(done, alpha, state, action, reward, state_, action_)

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

        logger.info(
            f'[Episode {self.episode}] - score: {score_e}, steps: {step_e}, mean score (100 ep.): {mean_score}.')

    def get_alpha(self):
        """

        :return:
        """

        # Linear decay, then exponential decay
        if self.step <= self.alpha_steps and self.alpha_steps > 0:
            alpha = self.alpha_start - self.step * (self.alpha_start - self.alpha_end) / self.alpha_steps
        else:
            alpha = self.alpha_end * self.alpha_decay ** (self.step - self.alpha_steps)

        return alpha

    def get_epsilon(self):
        """

        :return:
        """

        # Linear decay, then exponential decay
        if self.step <= self.epsilon_steps and self.epsilon_steps > 0:
            epsilon = self.epsilon_start - self.step * (self.epsilon_start - self.epsilon_end) / self.epsilon_steps
        else:
            epsilon = self.epsilon_end * self.epsilon_decay ** (self.step - self.epsilon_steps)

        return epsilon


class QAgent(SarsaAgent):
    """
    Agent that makes use of Q-learning (off-policy TD control).
    """

    def __init__(self, config):
        """

        :param config:
        """
        super().__init__(config)

    def learn(self, done, alpha, state, action, reward, state_, action_=None):
        """

        :param done:
        :param alpha:
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
        self.q_table[state][action] += alpha * (q_value_ - q_value)

    def do_episode(self, config):
        """

        :param config:
        :return:
        """

        # Initial values
        done = False
        score_e = 0
        step_e = 0

        # Get current state s
        state = self.discretize_state(self.env.reset())

        # Continue while not crashed
        while not done:

            # Show on screen
            if config['RENDER']:
                self.env.render()

            # Get learning parameters
            alpha = self.get_alpha()
            epsilon = self.get_epsilon()

            # Act based on current state s
            action = self.act(state, epsilon)
            state_, reward, done, _ = self.env.step(action)
            state_ = self.discretize_state(state_)

            # Learn
            self.learn(done, alpha, state, action, reward, state_)

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

        logger.info(
            f'[Episode {self.episode}] - score: {score_e}, steps: {step_e}, mean score (100 ep.): {mean_score}.')


class DoubleDQNAgent(QAgent):
    """
    Agent that makes use of Deep Q-learning, where Q(s, a) is approximated using a neural network.
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

        # Build Q-network
        self.q_network = self.build_network()

        # Build target network and initialize to weights of Q-network
        self.target_network = self.build_network()
        self.update_target_network()

    def build_network(self):
        """

        :return:
        """
        network = Sequential()
        network.add(Dense(self.layers[0], input_shape=self.env.observation_space.shape, activation='relu',
                          kernel_regularizer=l2(self.l2_reg)))
        network.add(Dense(self.layers[1], activation='relu', kernel_regularizer=l2(self.l2_reg)))
        network.add(Dense(self.layers[2], activation='relu', kernel_regularizer=l2(self.l2_reg)))
        network.add(Dense(self.env.action_space.n, activation='linear', kernel_regularizer=l2(self.l2_reg)))
        network.compile(loss='mse', optimizer=Adam(lr=self.alpha_end, decay=self.alpha_decay))

        return network

    def update_target_network(self):
        """

        :return:
        """
        self.target_network.set_weights(self.q_network.get_weights())

    def act(self, state, epsilon):
        """

        :param state:
        :param epsilon:
        :return:
        """

        if self.prng.random_sample() < epsilon:
            return self.prng.randint(self.env.action_space.n)
        else:
            return np.argmax(self.q_network.predict(state))

    def remember(self, done, state, action, reward, state_):
        """

        :param done:
        :param state:
        :param action:
        :param reward:
        :param state_:
        :return:
        """
        self.replay_memory.append((done, state, action, reward, state_))

    def train(self):
        """

        :return:
        """

        # Create minibatch
        x_batch, y_batch = [], []
        sample = self.prng.randint(len(self.replay_memory), size=self.batch_size)
        minibatch = [self.replay_memory[i] for i in sample]

        # Get input and target
        for done, state, action, reward, state_ in minibatch:
            y_current = self.q_network.predict(state)
            y_current_ = self.q_network.predict(state_)
            y_target_ = self.target_network.predict(state_)

            # Check if next state is terminal, update
            if done:
                y_current[0][action] = reward
            else:
                y_current[0][action] = reward + self.gamma * y_target_[0][np.argmax(y_current_[0])]

            # Append to training batch
            x_batch.append(state[0])
            y_batch.append(y_current[0])

        # Train
        self.q_network.fit(np.array(x_batch), np.array(y_batch), batch_size=self.batch_size, epochs=1, verbose=0)

    def preprocess_state(self, state):
        """

        :param state:
        :return:
        """
        return np.reshape(state, (1,) + self.env.observation_space.shape)

    def do_episode(self, config):
        """

        :param config:
        :return:
        """

        # Initial values
        done = False
        score_e = 0
        step_e = 0

        # Get current state s
        state = self.preprocess_state(self.env.reset())

        # Continue while not crashed
        while not done:

            # Show on screen
            if config['RENDER']:
                self.env.render()

            # Get learning parameters
            epsilon = self.get_epsilon()

            # Act based on current state s
            action = self.act(state, epsilon)
            state_, reward, done, _ = self.env.step(action)
            state_ = self.preprocess_state(state_)

            # Add to memory
            self.remember(done, state, action, reward, state_)

            # Train
            if len(self.replay_memory) > self.batch_size:
                self.train()

            # Set weights of target network to Q-network
            if self.step % config['UPDATE_EVERY'] == 0:
                self.update_target_network()

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

        logger.info(
            f'[Episode {self.episode}] - score: {score_e}, steps: {step_e}, mean score (100 ep.): {mean_score}.')

    def save_checkpoint(self, config):
        """

        :param config:
        :return:
        """

        # Save networks
        self.q_network.save(config['RECORD_DIR'] + 'q_network.h5')
        self.target_network.save(config['RECORD_DIR'] + 'target_network.h5')

        # Networks can't be pickled
        dummy_q_network = self.q_network
        dummy_target_network = self.target_network
        self.q_network = None
        self.target_network = None

        # Execute save function of base class
        super().save_checkpoint(config)

        # Put networks back
        self.q_network = dummy_q_network
        self.target_network = dummy_target_network
