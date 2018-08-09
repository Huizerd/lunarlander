"""
Contains utilities.
@author: Jesse Hagenaars
"""

import matplotlib.pyplot as plt
from bottleneck import move_mean


def restore_checkpoint():
    pass


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
    axes = [figure.add_subplot(311), figure.add_subplot(312), figure.add_subplot(313)]

    # Configure subplots
    axes[0].set_ylabel('score')
    axes[0].set_title('Score over episodes')
    axes[1].set_ylabel('score')
    axes[1].set_title('Score moving average over episodes')
    axes[2].set_xlabel('episode')
    axes[2].set_ylabel('epsilon')
    axes[2].set_title('Epsilon for e-greedy over episodes')

    # Get lines
    lines = [axes[0].plot([0], color='#008fd5')[0],
             axes[1].plot([0], color='#008fd5')[0],
             axes[2].plot([0], color='#fc4f30')[0]]

    return figure, axes, lines


def update_plots(figure, axes, lines, episode, score, epsilon):
    """

    :param figure:
    :param axes:
    :param lines:
    :param episode:
    :param score:
    :param epsilon:
    :return:
    """

    # Moving average
    score_ma = move_mean(score, window=(100 if len(score) > 99 else len(score)), min_count=1)

    # Update plot
    lines[0].set_data(range(1, episode + 2), score)
    lines[1].set_data(range(1, episode + 2), score_ma)
    lines[2].set_data(range(1, episode + 2), epsilon[:episode + 1])

    # Rescale axes
    for ax in axes:
        ax.relim()
        ax.autoscale_view()

    # Update figure
    figure.tight_layout()
    figure.canvas.draw()
    figure.canvas.flush_events()

    return figure
