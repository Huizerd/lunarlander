"""
Contains the agents that can be used.
@author: Jesse Hagenaars
"""

import numpy as np
import pandas as pd


# TODO: allow for changing e-greedy and plot it
# TODO: q-learning, deep q-learning (or any other continuous method)
# TODO: sort dataframe? --> performance warning about lexically sorting stuff
# TODO: parameters in config.yaml

class Agent:
    """
    Base class for RL agents.
    """

    def __init__(self, action_space, learning_rate=0.01, discount_rate=0.9, e_greedy=0.9):
        """
        Agent initialization.
        :param action_space:
        :param learning_rate:
        :param discount_rate:
        :param e_greedy:
        """

        self.action_space = action_space
        self.lr = learning_rate
        self.gamma = discount_rate
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame()

    def check_state_exist(self, state):
        """

        :param state: Tuple containing the state
        :return:
        """

        # Append state if it doesn't exist
        if not self.q_table.index.isin([state]).any():
            # State as index, actions as columns
            self.q_table = self.q_table.append(
                pd.DataFrame([np.zeros(self.action_space.n)], columns=range(self.action_space.n),
                             index=pd.MultiIndex.from_tuples([state])).astype(np.int8))

    def act(self, state):
        """
        Determine which action to take.
        :param state: Tuple containing the state
        :return: Integer representing action to take
        """

        # Check if state exists
        self.check_state_exist(state)

        # Select best/random action based on e-greedy policy
        if np.random.rand() < self.epsilon:
            # Best
            state_action = self.q_table.loc[state, :]
            state_action = state_action.sample(frac=1)  # pick random action in case of equal Q-values
            action = state_action.idxmax()

        else:
            # Random
            action = self.action_space.sample()

        return action

    def learn(self, *args):
        pass


class RandomAgent(Agent):
    """
    Simplest agent possible!
    """

    def __init__(self, action_space):
        """

        :param action_space:
        """
        super().__init__(action_space)

    def act(self, state):
        """

        :param state: Tuple containing the state
        :return:
        """
        return self.action_space.sample()


class SarsaAgent(Agent):
    """
    Agent that makes use of Sarsa (on-policy TD control).
    """

    def __init__(self, action_space, learning_rate=0.01, discount_rate=0.9, e_greedy=0.9):
        """

        :param action_space:
        :param learning_rate:
        :param discount_rate:
        :param e_greedy:
        """
        super().__init__(action_space, learning_rate, discount_rate, e_greedy)

    def learn(self, crashed, s, a, r_, s_, a_):
        """

        :param crashed: Whether the lander has crashed or not
        :param s: Tuple containing current state
        :param a: Integer representing current action
        :param r_: Next reward
        :param s_: Tuple containing next state
        :param a_: Integer representing next action
        :return:
        """

        # Check if next state s' exists, get current Q(s, a)
        self.check_state_exist(s_)
        q = self.q_table.loc[s, a]

        # Check if state is terminal, and get next Q(s', a')
        if not crashed:
            q_ = r_ + self.gamma * self.q_table.loc[s_, a_]
        else:
            q_ = r_

        # Update Q-values
        self.q_table.loc[s, a] += self.lr * (q_ - q)
