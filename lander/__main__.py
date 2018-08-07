"""
Runs the lander agent.
@author: Jesse Hagenaars
"""

import argparse
import os

import gym
from gym import logger, wrappers

from .agents import RandomAgent

# Global variables
SEED = 0
TRAIN_DIR = 'train/'
TEST_DIR = 'test/'
RECORD_DIR = 'record/'

# Create output directory if it doesn't exist
for d in [TRAIN_DIR, TEST_DIR, RECORD_DIR]:
    if not os.path.exists(d):
        os.makedirs(d)


def main(**kwargs):
    """
    Runs the lander agent.
    :param kwargs: Dict of keyword arguments from the parser
    :return:
    """

    env = gym.make('LunarLander-v2')
    env = wrappers.Monitor(env, directory=RECORD_DIR, force=True)  # records only a sample of episodes, not all
    env.seed(SEED)

    if kwargs['agent'] == 'random':
        agent = RandomAgent(env.action_space)
    else:
        raise ValueError('No valid agent given!')

    episode_count = kwargs['episodes']
    reward = 0

    for e in range(episode_count):
        observation = env.reset()
        done = False
        t = 0
        while not done:
            env.render()
            # print(observation)
            action = agent.act(observation, reward, done)
            observation, reward, done, _ = env.step(action)
            t += 1

        print(f'Episode {e + 1} finished after {t + 1} timesteps')

    env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-a', '--agent', type=str, default='random',
                        help='Choose the agent to use: random | sarsa | qlearning')
    parser.add_argument('-e', '--episodes', type=int, default=10, help='Set the number of episodes')
    parser.add_argument('-m', '--mode', type=str, default='test', help='Choose between train and test')
    kwargs = vars(parser.parse_args())

    # Determine the amount of info to receive
    logger.set_level(logger.WARN)

    main(**kwargs)
