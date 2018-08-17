"""
Contains utilities.
@author: Jesse Hagenaars
"""

import pickle

import gym
import matplotlib.pyplot as plt
from bottleneck import move_mean
from gym.wrappers import Monitor
from keras.models import load_model


def restore_checkpoint(config):
    """

    :param config:
    :return:
    """

    # Load agent
    with open(config['CHECKPOINT_DIR'] + 'checkpoint.pickle', 'rb') as p_file:
        agent = pickle.load(p_file)

    import pdb
    pdb.set_trace()

    # TODO: be able to restore random state of previous environment --> gym issue (pull request?)
    # TODO: ensure resume works properly (or bring runs back)
    # Get env based on env ID saved in agent
    agent.env = Monitor(gym.make(agent.env_id), directory=config['RECORD_DIR'],
                        video_callable=lambda episode_id: (episode_id + 1) % config['SAVE_EVERY'] == 0,
                        force=False, resume=True,
                        uid=config['AGENT'])  # record every nth episode, clear monitor files if present
    agent.env.seed(agent.env_seed)

    # Get networks for DoubleDQN
    if config['AGENT'] == 'doubledqn':
        agent.q_network = load_model(config['CHECKPOINT_DIR'] + 'q_network.h5')
        agent.target_network = load_model(config['CHECKPOINT_DIR'] + 'target_network.h5')

    return agent


def save():
    pass


def build_z_line():
    pass


def prepare_plots():
    """

    :return: Figure, axes, lines
    """

    # Interactive, style
    plt.ion()
    plt.style.use('fivethirtyeight')
    plt.rcParams['lines.linewidth'] = 2

    # Get figure and subplots
    figure = plt.figure()
    axes = [figure.add_subplot(211), figure.add_subplot(212)]

    # Configure subplots
    axes[0].set_ylabel('score')
    axes[0].set_title('Score over episodes')
    axes[1].set_ylabel('score')
    axes[1].set_title('Score moving average over episodes')
    axes[1].set_xlabel('episode')

    # Get lines
    lines = [axes[0].plot([0], color='#008fd5')[0],
             axes[1].plot([0], color='#008fd5')[0]]

    return figure, axes, lines


def update_plots(figure, axes, lines, episode, score):
    """

    :param figure:
    :param axes:
    :param lines:
    :param episode:
    :param score:
    :return:
    """

    # Moving average
    score_ma = move_mean(score, window=(100 if len(score) > 99 else len(score)), min_count=1)

    # Update plot
    lines[0].set_data(range(1, episode + 1), score)
    lines[1].set_data(range(1, episode + 1), score_ma)

    # Rescale axes
    for ax in axes:
        ax.relim()
        ax.autoscale_view()

    # Update figure
    figure.tight_layout()
    figure.canvas.draw()
    figure.canvas.flush_events()

    return figure
