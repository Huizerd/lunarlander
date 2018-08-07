"""
Contains the agents that can be used.
@author: Jesse Hagenaars
"""

import numpy as np
import pandas as pd


# TODO: allow for changing e-greedy


class RandomAgent:
    """
    Simplest agent possible!
    """

    def __init__(self, action_space):
        """
        Agent initialization.
        :param action_space: List containing the action space.
        """
        self.action_space = action_space

    def act(self, observation, reward, done):
        """
        Sample a random action from the action space.
        :param observation:
        :param reward:
        :param done:
        :return: A random action from the available action space
        """
        return self.action_space.sample()


class Agent:
    """
    Base class for RL agents.
    """

    def __init__(self, action_space, learning_rate=0.01, discount_rate=0.9, e_greedy=0.9):
        """
        Agent initialization.
        :param action_space: List containing the action space
        :param learning_rate:
        :param discount_rate:
        :param e_greedy:
        """

        self.actions = action_space
        self.lr = learning_rate
        self.gamma = discount_rate
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

    def check_state_exist(self, state):
        """

        :param state:
        :return:
        """

        # Append state if it doesn't exist
        if state not in self.q_table.index:
            self.q_table = self.q_table.append(pd.Series([0] * len(self.actions), index=self.q_table.columns,
                                                         name=state))

    def choose_action(self, observation):
        """

        :param observation:
        :return:
        """

        # Check if state exists
        self.check_state_exist(observation)

        # Select best/random action based on e-greedy policy
        if np.random.rand() < self.epsilon:
            # Best
            state_action = self.q_table.loc[observation, :]
            state_action = state_action.reindex(np.random.permutation(
                state_action.index))  # some actions have same value
            action = state_action.idxmax()

        else:
            # Random
            action = np.random.choice(self.actions)

        return action

    def learn(self, *args):
        pass


class SarsaAgent(Agent):
    """
    Agent that makes use of Sarsa (on-policy TD control).
    """

    def __init__(self, actions, learning_rate=0.01, discount_rate=0.9, e_greedy=0.9):
        """

        :param actions:
        :param learning_rate:
        :param discount_rate:
        :param e_greedy:
        """
        super().__init__(actions, learning_rate, discount_rate, e_greedy)

    def learn(self, s, a, r_, s_, a_):
        """

        :param s: Current state
        :param a: Current action
        :param r_: Next reward
        :param s_: Next state
        :param a_: Next action
        :return:
        """

        # Check if next state s' exists, get current Q(s, a)
        self.check_state_exist(s_)
        q = self.q_table.loc[s, a]

        # Check if state is terminal, and get next Q(s', a')
        if s_ != 'terminal':
            q_ = r_ + self.gamma * self.q_table.loc[s_, a_]
        else:
            q_ = r_

        # Update Q-values
        self.q_table.loc[s, a] += self.lr * (q_ - q)
