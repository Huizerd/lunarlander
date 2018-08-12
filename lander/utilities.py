"""
Contains utilities.
@author: Jesse Hagenaars
"""

import pickle

import gym
import matplotlib.pyplot as plt
from bottleneck import move_mean
from gym.wrappers import Monitor


def restore_checkpoint(config):
    """

    :param config:
    :return:
    """

    # Load checkpoint
    checkpoint = pickle.load(open(config['CHECKPOINT_DIR'] + 'checkpoint.pickle', 'rb'))

    # Get agent
    agent = checkpoint['agent']

    # Get env based on env ID saved in agent
    agent.env = Monitor(gym.make(agent.env_id), directory=config['RECORD_DIR'] + f'run_{agent.run}',
                        video_callable=lambda episode_id: (episode_id + 1) % config['SAVE_EVERY'] == 0,
                        force=True)  # record every nth episode, clear monitor files if present
    agent.env.seed(agent.seed)

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
    lines[0].set_data(range(1, episode + 2), score)
    lines[1].set_data(range(1, episode + 2), score_ma)

    # Rescale axes
    for ax in axes:
        ax.relim()
        ax.autoscale_view()

    # Update figure
    figure.tight_layout()
    figure.canvas.draw()
    figure.canvas.flush_events()

    return figure
