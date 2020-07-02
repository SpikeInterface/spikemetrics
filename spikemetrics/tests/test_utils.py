import numpy as np
import pytest

from spikemetrics.utils import get_spike_positions
from spikemetrics.tests.utils import simulated_pcs_for_one_spike


def test_get_spike_depths():

    peak_chan = 5

    pc_features, pc_feature_ind = simulated_pcs_for_one_spike(32, peak_chan)
    spike_clusters = np.ones((pc_features.shape[0],), dtype='int') * 0

    positions = get_spike_positions(spike_clusters, pc_features, pc_feature_ind, vertical_channel_spacing=1)
    depths = positions[:, 1]

    assert np.isclose(depths, peak_chan, rtol=0, atol=1e-5)
