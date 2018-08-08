"""
Contains utilities.
@author: Jesse Hagenaars
"""

# TODO: check -1 to 1 for bins (outside is all the same)

import numpy as np
import pandas as pd


def restore_checkpoint():
    pass


def build_state(features):
    return int("".join(map(lambda feature: str(int(feature)), features)))


def discretize_state(state, n_bins):
    """
    Discretizes the state vector into a specified number of bins.
    :param state: List containing the state vector
    :param n_bins: Number of bins
    :return: Tuple of discretized state bin indices
    """

    # Get bins
    # Cut off edges, so anything > 1.0 or < -1.0 is the same bin
    bins = pd.cut([-1.0, 1.0], bins=n_bins, retbins=True)[1][1:-1]

    # Bin indices are fine, since we won't be interpreting the state anyway
    return tuple(np.digitize(state, bins=bins))
