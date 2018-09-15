"""
Contains utilities.
@author: Jesse Hagenaars
"""

import pickle

import gym
import matplotlib.pyplot as plt
from bottleneck import move_mean
from gym.wrappers import Monitor

# Matplotlib config
plt.style.use('ggplot')
# plt.rcParams['lines.linewidth'] = 2
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Ubuntu'
plt.rcParams['font.monospace'] = 'Ubuntu Mono'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['figure.titlesize'] = 14


# TODO: make restore working for double DQN (including TF random seed)

def restore_checkpoint(config):
    """

    :param config:
    :return:
    """

    # Load agent
    with open(config['CHECKPOINT_DIR'] + 'checkpoint.pickle', 'rb') as p_file:
        agent = pickle.load(p_file)

    # Increment run
    agent.run += 1

    # TODO: be able to restore random state of previous environment --> gym issue (pull request?)
    # Get env based on env ID saved in agent
    agent.env = Monitor(gym.make(agent.env_id), directory=config['RECORD_DIR'] + f'run_{agent.run}',
                        video_callable=lambda episode_id: (episode_id + 1) % config['SAVE_EVERY'] == 0,
                        force=True, uid=config['AGENT'])  # record every nth episode, clear monitor files if present
    agent.env.seed(agent.env_seed)

    return agent


def prepare_plots():
    """

    :return: Figure, axes, lines
    """

    # Interactive, style
    plt.ion()

    # Get figure and subplots
    figure = plt.figure(figsize=(16, 4))
    axes = [figure.add_subplot(121), figure.add_subplot(122)]

    # Configure subplots
    axes[0].set_xlabel('episode')
    axes[0].set_ylabel('score')
    axes[0].set_title('Score over episodes', fontstyle='italic')
    axes[1].set_xlabel('episode')
    axes[1].set_ylabel('score')
    axes[1].set_title('Score moving average over episodes', fontstyle='italic')

    # Get lines
    lines = [axes[0].plot([0])[0],
             axes[1].plot([0])[0]]

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
