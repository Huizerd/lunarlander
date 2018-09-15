"""
Runs the lander agent.
@author: Jesse Hagenaars
"""

import argparse
import os

import numpy as np
import yaml
from gym import logger

from .agents import RandomAgent, SarsaAgent, QAgent, DoubleDQNAgent
from .utilities import prepare_plots, update_plots, restore_checkpoint


def main(config):
    """
    Runs the lander agent.
    :param config: Dict containing the specified configuration
    :return:
    """

    # Restore from checkpoint
    if config['CONTINUE']:
        agent = restore_checkpoint(config)
    else:
        # Select and configure agent
        if config['AGENT'] == 'random':
            agent = RandomAgent(config)
        elif config['AGENT'] == 'sarsa':
            agent = SarsaAgent(config)
        elif config['AGENT'] == 'qlearn':
            agent = QAgent(config)
        elif config['AGENT'] == 'doubledqn':
            agent = DoubleDQNAgent(config)
        else:
            raise ValueError('Invalid agent specified!')

    # Prepare plots
    if config['VERBOSE'] > 0:
        figure, axes, lines = prepare_plots()
    else:
        figure, axes, lines = None, None, None

    # Start
    while agent.episode < agent.episode_count:

        # Do episode
        agent.do_episode(config)

        # Update plots
        if config['VERBOSE'] > 0:
            figure = update_plots(figure, axes, lines, agent.episode, agent.score)

        # Save every nth episode
        if agent.episode % config['SAVE_EVERY'] == 0 and config['VERBOSE'] > 0:
            agent.save_checkpoint(config)
            figure.savefig(config['RECORD_DIR'] + 'score.pdf')

        # Break when goal of 100-score > 200 is reached
        if np.mean(agent.score_100) >= 200.0:
            if config['VERBOSE'] > 0:
                agent.save_checkpoint(config)
                figure.savefig(config['RECORD_DIR'] + 'score.pdf')
            logger.info('Goal reached!')
            break

    # Close
    agent.env.close()
    if config['AGENT'] == 'doubledqn':
        agent.sess.close()


if __name__ == '__main__':

    # Parse for configuration file
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-c', '--config', type=str, default='config/config.yaml.default',
                        help='Select the configuration file')
    args = vars(parser.parse_args())

    # Load configuration
    with open(args['config'], 'r') as config_file:
        config = yaml.load(config_file)

    # Create recording directory if it doesn't exist
    if not os.path.exists(config['RECORD_DIR']):
        os.makedirs(config['RECORD_DIR'])

    # Determine the amount of info to receive
    if config['VERBOSE'] > 0:
        logger.set_level(logger.INFO)
    else:
        logger.set_level(logger.DISABLED)

    # Run main
    main(config)
