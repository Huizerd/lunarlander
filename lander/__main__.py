"""
Runs the lander agent.
@author: Jesse Hagenaars
"""

import argparse
import os

import gym
from gym import logger

from .agents import RandomAgent, SarsaAgent

# TODO: discretize state?
# TODO: fix render --> only record, not on screen? Also adjust docs then

# Global variables
SEED = 0
RECORD_DIR = 'record/'
ENV = 'LunarLander-v2'

# Create output directory if it doesn't exist
if not os.path.exists(RECORD_DIR):
    os.makedirs(RECORD_DIR)


def main(**kwargs):
    """
    Runs the lander agent.
    :param kwargs: Dict of keyword arguments from the parser
    :return:
    """

    # Initialize environment
    env = gym.make(ENV)
    # env = wrappers.Monitor(env, directory=RECORD_DIR + kwargs['agent'] + '/',
    #                        force=True)  # records only a sample of episodes, not all
    env.seed(SEED)
    episode_count = kwargs['episodes']

    # Select agent
    if kwargs['agent'] == 'random':
        agent = RandomAgent(env.action_space)
    elif kwargs['agent'] == 'sarsa':
        agent = SarsaAgent(env.action_space)
    else:
        raise ValueError('No valid agent given!')

    # Start
    for e in range(episode_count):

        # Initial values
        # State vector: x, y, Vx, Vy, angle, contact left, contact right (all between -1 and 1)
        state = tuple(env.reset())
        crashed = False
        score = 0
        t = 0

        # Initial action
        # Action vector: do nothing, fire left, fire main, fire right
        action = agent.act(state)

        # Continue while not crashed
        while not crashed:
            # Refresh environment
            if kwargs['render']:
                env.render()

            # Get next state and reward
            state_, reward_, crashed, _ = env.step(action)
            state_ = tuple(state_)

            # Act
            action_ = agent.act(state_)

            # Learn
            agent.learn(crashed, state, action, reward_, state_, action_)

            # Set next state and action to current
            state = state_
            action = action_
            score += reward_

            # Increment time for this episode
            t += 1

        print(f'Episode {e + 1} finished after {t + 1} timesteps with a score of {score}')

    env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-a', '--agent', type=str, default='random',
                        help='Choose the agent to use: random | sarsa | qlearning')
    parser.add_argument('-e', '--episodes', type=int, default=10, help='Set the number of episodes')
    parser.add_argument('-r', '--render', type=bool, default=True, help='Choose to render on-screen')
    kwargs = vars(parser.parse_args())

    # Determine the amount of info to receive
    logger.set_level(logger.WARN)

    main(**kwargs)
