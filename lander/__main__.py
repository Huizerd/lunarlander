"""
Runs the lander agent.
@author: Jesse Hagenaars
"""

import argparse
import os

import yaml
from gym import logger

from .agents import RandomAgent, SarsaAgent, QAgent, DeepQAgent
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
        elif config['AGENT'] == 'deepq':
            agent = DeepQAgent(config)
        else:
            raise ValueError('Invalid agent specified!')

    # Prepare plots
    figure, axes, lines = prepare_plots()

    # Start
    for e in range(agent.episode_start, agent.episode_count):

        # Do episode
        agent.do_episode(config, e)

        # Update plots
        figure = update_plots(figure, axes, lines, e, agent.score)

        # Save every nth episode
        if (e + 1) % config['SAVE_EVERY'] == 0:
            agent.save_checkpoint(config, e)
            figure.savefig(config['RECORD_DIR'] + 'score.pdf')

    # Close environment
    agent.env.close()


if __name__ == '__main__':

    # Parse for configuration file
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-c', '--config', type=str, default='config.yaml.default',
                        help='Select the configuration file')
    args = vars(parser.parse_args())

    # Load configuration
    with open(args['config'], 'r') as config_file:
        config = yaml.load(config_file)

    # Create recording directory if it doesn't exist
    if not os.path.exists(config['RECORD_DIR']):
        os.makedirs(config['RECORD_DIR'])

    # Determine the amount of info to receive
    logger.set_level(logger.INFO)

    # Run main
    main(config)
