"""
Contains the agents that can be used.
@author: Jesse Hagenaars
"""

# TODO: q-learning, deep q-learning (or any other continuous method)

from functools import reduce

import numpy as np
import pandas as pd


class Agent:
    """
    Base class for RL agents.
    """

    def __init__(self, state_space, action_space, episodes, config=None):
        """
        Agent initialization.
        :param state_space:
        :param action_space:
        :param episodes:
        :param config:
        """

        # Get from environment
        self.action_space = action_space

        # Seed
        np.random.seed(config['ENV_SEED'])

        # Defaults
        if config is None:
            learning_rate = (0, -1, 0.001, 0.001)
            discount_rate = (0, -1, 0.9, 0.9)
            e_greedy = (0, -1, 0.9, 0.9)
        else:
            learning_rate = config['LEARNING_RATE']
            discount_rate = config['DISCOUNT_RATE']
            e_greedy = config['E_GREEDY']

        # Flat part with starting value, then linear sloping part, then flat part with final value
        self.lr = np.concatenate([np.ones(learning_rate[0]) * learning_rate[2],
                                  np.linspace(learning_rate[2], learning_rate[3],
                                              num=(episodes if learning_rate[1] == -1 else learning_rate[1]) -
                                                  learning_rate[0]),
                                  np.ones(episodes - (episodes if learning_rate[1] == -1 else learning_rate[1])) *
                                  learning_rate[3]], axis=0)
        self.gamma = np.concatenate([np.ones(discount_rate[0]) * discount_rate[2],
                                     np.linspace(discount_rate[2], discount_rate[3],
                                                 num=(episodes if discount_rate[1] == -1 else discount_rate[1]) -
                                                     discount_rate[0]),
                                     np.ones(episodes - (episodes if discount_rate[1] == -1 else discount_rate[1])) *
                                     discount_rate[3]], axis=0)
        self.epsilon = np.concatenate([np.ones(e_greedy[0]) * e_greedy[2],
                                       np.linspace(e_greedy[2], e_greedy[3],
                                                   num=(episodes if e_greedy[1] == -1 else e_greedy[1]) -
                                                       e_greedy[0]),
                                       np.ones(episodes - (episodes if e_greedy[1] == -1 else e_greedy[1])) *
                                       e_greedy[3]], axis=0)

        # Discretize state space
        self.discretize_state(config['STATE_BOUNDS'], config['STATE_BINS'])

        # Filled with random, size depends on discretization of state
        n_states = reduce(lambda x, y: x * y, config['STATE_BINS'])
        self.q_table = np.random.uniform(low=-1.0, high=1.0, size=(n_states, action_space.n))

    def check_state_exist(self, state):
        """

        :param state: Tuple containing the state
        :return:
        """

        # Append state if it doesn't exist
        if state not in self.q_table.index:
            self.q_table = self.q_table.append(
                pd.DataFrame([np.zeros(self.action_space.n)], columns=range(self.action_space.n),
                             index=[state]))

    def act(self, state, episode):
        """
        Determine which action to take.
        :param state: List containing the state
        :param episode: Current episode
        :return: Integer representing action to take
        """

        # Select best/random action based on e-greedy policy
        if np.random.rand() < self.epsilon[episode]:
            # Flatten state to get list index
            flat_state = self.flatten_state(state)

            # Do best action
            action = np.argmax(self.q_table[flat_state, :])
        else:
            # Do random action
            action = self.action_space.sample()

        return action

    def learn(self, *args):
        pass

    def create_bins(self, between, n):
        """

        :param between:
        :param n:
        :return:
        """

        # Throw out the edges, so everything beyond the edge is the same as outer bin
        return pd.cut(between, bins=n, retbins=True)[1][1:-1]

    def discretize_state(self, state_bounds, n_bins):
        """
        Discretizes the state vector into a specified number of bins.
        :param state_bounds: State space bounds per state dimension
        :param n_bins: List containing the number of bins per state dimension
        :return: Tuple of discretized state bin indices
        """

        # Discretize
        self.discretized_state = [self.create_bins(between, n) for between, n in zip(state_bounds, n_bins)]

    def flatten_state(self, state):
        """
        Collapses state into bins, then flattens it into 1D.
        :param state:
        :return:
        """

        # Combine state values with bins per dimension
        pairs = list(zip(state, self.discretized_state))

        # Discretize state
        discrete_state = [np.digitize(*pair) for pair in pairs]

        # Multiply all bin indices + 1 to get the 1D index, then subtract 1 to account for 0-indexed lists
        flat_state = reduce(lambda x, y: x * y, [d + 1 for d in discrete_state]) - 1

        return flat_state


class RandomAgent(Agent):
    """
    Simplest agent possible!
    """

    def __init__(self, state_space, action_space, episodes):
        """

        :param state_space:
        :param action_space:
        :param episodes:
        """
        super().__init__(state_space, action_space, episodes)

    def act(self, state, episode):
        """

        :param state: Tuple containing the state
        :param episode: Current episode
        :return:
        """
        return self.action_space.sample()


class SarsaAgent(Agent):
    """
    Agent that makes use of Sarsa (on-policy TD control).
    """

    def __init__(self, state_space, action_space, episodes, config=None):
        """

        :param state_space:
        :param action_space:
        :param episodes:
        :param config:
        """
        super().__init__(state_space, action_space, episodes, config)

    def learn(self, episode, crashed, s, a, r_, s_, a_):
        """

        :param episode: Current episode
        :param crashed: Whether the lander has crashed or not
        :param s: Tuple containing current state
        :param a: Integer representing current action
        :param r_: Next reward
        :param s_: Tuple containing next state
        :param a_: Integer representing next action
        :return:
        """

        # Flatten current state s and next state s'
        flat_s = self.flatten_state(s)
        flat_s_ = self.flatten_state(s_)

        # Get current Q(s, a)
        q = self.q_table[flat_s, a]

        # Check if state is terminal, and get next Q(s', a')
        if not crashed:
            q_ = r_ + self.gamma[episode] * self.q_table[flat_s_, a_]
        else:
            q_ = r_

        # Update current Q(s, a)
        self.q_table[flat_s, a] += self.lr[episode] * (q_ - q)
