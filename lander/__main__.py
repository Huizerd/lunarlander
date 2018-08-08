"""
Runs the lander agent.
@author: Jesse Hagenaars
"""

# TODO: multi-processing
# TODO: sort dataframe? --> performance warning about lexically sorting stuff
# TODO: why more lexsort warnings?
# TODO: parameters in config.yaml and checkpoint
# TODO: distinct sampling per state dimension: x, y: 10, vel. x, y: 5, angle: 5, contact l, r: 2

import argparse
import os
import pickle

import gym
import matplotlib.pyplot as plt
import yaml
from bottleneck import move_mean
from gym import logger, wrappers

from .agents import RandomAgent, SarsaAgent
from .utilities import discretize_state


def main(config):
    """
    Runs the lander agent.
    :param config: Dict containing the specified configuration
    :return:
    """

    # Continue from checkpoint
    if config['CONTINUE']:
        checkpoint = pickle.load(open(config['CHECKPOINT_DIR'] + 'checkpoint.pickle', 'rb'))

        # Episode, score, state bins of previous run
        episode_start = checkpoint['episode'] + 1  # increment to indicate current episode
        score = checkpoint['score']
        state_bins = checkpoint['state_bins']
        run = checkpoint['run'] + 1  # increment with 1 to indicate current run

        # Initialize environment
        env_name = checkpoint['env_name']
        env_seed = checkpoint['env_seed']
        env = gym.make(env_name)
        env = wrappers.Monitor(env, directory=config['RECORD_DIR'] + f'run_{run}',
                               video_callable=lambda episode_id: (episode_id + 1) % config['SAVE_EVERY'] == 0,
                               force=True)  # record every nth episode, clear monitor files if present
        env.seed(env_seed)

        # Select agent
        agent = checkpoint['agent']

    else:
        # Start from scratch
        episode_start = 0
        score = []
        state_bins = config['STATE_BINS']
        run = 1

        # Initialize environment
        env_name = config['ENV_NAME']
        env_seed = config['ENV_SEED']
        env = gym.make(env_name)
        env = wrappers.Monitor(env, directory=config['RECORD_DIR'] + f'run_{run}',
                               video_callable=lambda episode_id: (episode_id + 1) % config['SAVE_EVERY'] == 0,
                               force=True)  # record every nth episode, clear monitor files if present
        env.seed(env_seed)

        # Select agent
        if config['AGENT'] == 'random':
            agent = RandomAgent(env.action_space)
        elif config['AGENT'] == 'sarsa':
            agent = SarsaAgent(env.action_space)
        else:
            raise ValueError('Invalid agent specified!')

    # Always get episode count from config --> you might want to lower it
    episode_count = config['EPISODES']

    # Prepare plot
    plt.ion()
    plt.style.use('fivethirtyeight')
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax1.set_ylabel('score')
    ax1.set_title('Score over epochs')
    ax2 = fig.add_subplot(212)
    ax2.set_xlabel('epoch')
    ax2.set_ylabel('score')
    ax2.set_title('Score moving average over epochs')

    line1, = ax1.plot([0])  # returns a tuple of line objects, thus the comma
    line2, = ax2.plot([0])

    # Start
    for e in range(episode_start, episode_count):

        # Initial values
        # State vector: x, y, Vx, Vy, angle, contact left, contact right (all between -1 and 1, discretized)
        state = discretize_state(env.reset(), state_bins)
        crashed = False
        score_e = 0
        t = 0

        # Initial action
        # Action vector: do nothing, fire left, fire main, fire right
        action = agent.act(state)

        # Continue while not crashed
        while not crashed:
            # Show on screen
            if config['RENDER']:
                env.render()

            # Get next state and reward (discretized again)
            state_, reward_, crashed, _ = env.step(action)
            state_ = discretize_state(state_, state_bins)

            # Act
            action_ = agent.act(state_)

            # Learn
            agent.learn(crashed, state, action, reward_, state_, action_)

            # Set next state and action to current
            state = state_
            action = action_
            score_e += reward_

            # Increment time for this episode
            t += 1

        # Print results
        logger.info(f'Episode {e + 1} finished after {t + 1} timesteps with a score of {score_e}')

        # Append score
        score.append(score_e)
        score_ma = move_mean(score, window=(100 if len(score) > 99 else len(score)), min_count=1)

        # Update plot
        line1.set_data(range(1, e + 2), score)
        line2.set_data(range(1, e + 2), score_ma)
        ax1.relim()
        ax2.relim()
        ax1.autoscale_view()
        ax2.autoscale_view()
        fig.tight_layout()
        fig.canvas.draw()
        fig.canvas.flush_events()

        # Save every nth episode
        if (e + 1) % config['SAVE_EVERY'] == 0:
            save = {'run': run, 'episode': e, 'env_name': env_name, 'env_seed': env_seed, 'state_bins': state_bins,
                    'agent': agent, 'score': score}
            pickle.dump(save, open(config['RECORD_DIR'] + 'checkpoint.pickle', 'wb'))
            fig.savefig(config['RECORD_DIR'] + 'score.pdf')

    env.close()


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
