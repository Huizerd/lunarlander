"""
Performs a parameter grid search.
"""

import argparse
import json
import multiprocessing as mp
import os
import time

import yaml
from agents import RandomAgent, SarsaAgent, QAgent, DoubleDQNAgent
from gym import logger
from sklearn.model_selection import ParameterGrid


def eval_single(args):
    """

    :param args:
    :return:
    """

    # Unpack arguments
    idx, params = args

    # Scores to save: highest score at certain episode
    scores = []

    # 10 runs and average
    for i in range(10):

        # Set seed based on run index
        params['ENV_SEED'] = i

        # Select and configure agent
        if params['AGENT'] == 'random':
            agent = RandomAgent(params)
        elif params['AGENT'] == 'sarsa':
            agent = SarsaAgent(params)
        elif params['AGENT'] == 'qlearn':
            agent = QAgent(params)
        elif params['AGENT'] == 'doubledqn':
            agent = DoubleDQNAgent(params)
        else:
            raise ValueError('Invalid agent specified!')

        # Start
        while agent.episode < agent.episode_count:
            # Do episode
            agent.do_episode(params)

        # Get best score
        scores.append(agent.get_best_score())

        # Close
        agent.env.close()
        if params['AGENT'] == 'doubledqn':
            agent.sess.close()

    # Average for episode and score
    score = (idx,) + tuple(map(lambda x: sum(x) / float(len(x)), zip(*scores)))

    return score


if __name__ == '__main__':

    # Parse for configuration file
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-c', '--config', type=str, default='config/config.yaml.default',
                        help='Select the configuration file')
    parser.add_argument('-g', '--grid', type=str, default='grid/grid_search.yaml.default',
                        help='Select the parameter grid')
    args = vars(parser.parse_args())

    # Load configuration
    with open(args['config'], 'r') as config_file:
        config = yaml.load(config_file)

    # Ensure verbose is 0
    config['VERBOSE'] = 0

    # Put all values in lists
    config = {key: [value] for key, value in config.items()}

    # Load parameter grid configuration
    with open(args['grid'], 'r') as grid_file:
        grid_config = yaml.load(grid_file)

    # Overwrite config
    # NOTE: get() makes use of a default value in case key is not in grid_config
    config = {key: grid_config.get(key, config[key]) for key in config}

    # Build parameter grid
    params = list(ParameterGrid(config))

    # Start clock
    start = time.time()
    print(f'About to evaluate {len(params)} parameter sets')

    # Disable logger
    logger.set_level(logger.DISABLED)

    # Multiprocessing pool
    pool = mp.Pool(processes=mp.cpu_count())

    # Run
    final_scores = pool.map(eval_single, list(enumerate(params)))

    # Close
    pool.close()
    pool.join()

    # Finished!
    print(f'Execution time: {(time.time() - start) / 3600:.2f} hours')

    # Create recording directory if it doesn't exist
    if not os.path.exists(config['RECORD_DIR'][0]):
        os.makedirs(config['RECORD_DIR'][0])

    # Save grid and scores
    dump = {'grid': params, 'scores': final_scores}
    with open(config['RECORD_DIR'][0] + 'grid_search.json', 'w') as result_file:
        json.dump(dump, result_file, sort_keys=True, indent=2)
