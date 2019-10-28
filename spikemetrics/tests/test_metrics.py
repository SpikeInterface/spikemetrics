import numpy as np
import pytest

from spikemetrics import calculate_amplitude_cutoff, calculate_drift_metrics, calculate_firing_rate_and_spikes, \
    calculate_isi_violations, calculate_pc_metrics, calculate_silhouette_score, calculate_presence_ratio, \
    calculate_metrics

from spikemetrics.metrics import isi_violations, firing_rate

from spikemetrics.tests.utils import simulated_spike_train

@pytest.fixture
def simulated_spikes():
    max_time = 100
    train1 = simulated_spike_train(max_time, 10, 2)
    train2 = simulated_spike_train(max_time, 5, 4)
    train3 = simulated_spike_train(max_time, 5, 10)

    labels1 = np.ones((train1.shape), dtype='int') * 0   
    labels2 = np.ones((train2.shape), dtype='int') * 1
    labels3 = np.ones((train3.shape), dtype='int') * 2

    spike_times = np.concatenate((train1, train2, train3))
    spike_clusters = np.concatenate((labels1, labels2, labels3))

    order = np.argsort(spike_times)

    spike_times = spike_times[order]
    spike_clusters = spike_clusters[order]

    return {'spike_times' : spike_times, 'spike_clusters' : spike_clusters}


def test_calculate_amplitude_cutoff():
    pass


def test_calculate_drift_metrics():
    pass


def test_calculate_pc_metrics():
    pass


def test_calculate_silhouette_score():
    pass


def test_calculate_presence_ratio():
    pass


def test_calculate_metrics():
    
    pass


def test_calculate_firing_rate_and_spikes(simulated_spikes):

    firing_rates, spike_counts = calculate_firing_rate_and_spikes(simulated_spikes['spike_times'], 
                                    simulated_spikes['spike_clusters'], 
                                    3, verbose=False)

    print(firing_rates)

    assert np.allclose(firing_rates, np.array([10.03003003,  5.04504505,  5.10510511]))
    assert np.allclose(spike_counts, np.array([1002, 504, 510]))


def test_calculate_isi_violations(simulated_spikes):

    viol = calculate_isi_violations(simulated_spikes['spike_times'], 
                                    simulated_spikes['spike_clusters'], 
                                    3, 0.001, 0.0, verbose=False)

    assert np.allclose(viol, np.array([0.0995016 , 0.78656463, 1.92041522]))


def test_isi_violations():

    # 1. check value for fixed spike train parameters:

    train1 = simulated_spike_train(100, 10, 10)
    fpRate1, num_violations1 = isi_violations(train1, 0, np.max(train1), 0.001)

    assert np.isclose(fpRate1, 0.4901480247, rtol=0, atol=1e-5)
    assert num_violations1 == 10


    # 2. check that the value doesn't depend on recording duration:

    train2 = simulated_spike_train(200, 10, 20)
    fpRate2, num_violations2 = isi_violations(train2, 0, np.max(train2), 0.001)

    assert np.isclose(fpRate1, fpRate2, rtol=0, atol=1e-5)


    # 3. check that the value increases with the number of violations:

    train3 = simulated_spike_train(100, 10, 20)
    fpRate3, num_violations3 = isi_violations(train3, 0, np.max(train3), 0.001)

    assert fpRate3 > fpRate1


    # 4. check that the value decreases with a longer violation time:

    fpRate4, num_violations4 = isi_violations(train1, 0, np.max(train1), 0.002)

    assert fpRate4 < fpRate1


    # 5. check that the value decreases with firing rate:

    train4 = simulated_spike_train(100, 20, 10)
    fpRate5, num_violations5 = isi_violations(train4, 0, np.max(train4), 0.001)

    assert fpRate5 < fpRate1



def test_firing_rate():

    # 1. check that the output value is correct:

    train = simulated_spike_train(100, 10, 0)

    assert firing_rate(train, min_time=0, max_time=100) == 10.0

    # 2. check that widening the boundaries decreases the rate:

    assert firing_rate(train, min_time=0, max_time=100) > firing_rate(train, min_time=0, max_time=200)

