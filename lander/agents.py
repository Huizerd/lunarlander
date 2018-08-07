"""
Contains the agents that can be used.
@author: Jesse Hagenaars
"""


class RandomAgent:
    """
    Simplest agent possible!
    """

    def __init__(self, action_space):
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
