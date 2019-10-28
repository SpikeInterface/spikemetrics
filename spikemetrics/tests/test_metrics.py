import numpy as np

from spikemetrics import calculate_amplitude_cutoff, calculate_drift_metrics, calculate_firing_rate_and_spikes, \
    calculate_isi_violations, calculate_pc_metrics, calculate_silhouette_score, calculate_presence_ratio, \
    calculate_metrics

from spikemetrics.metrics import isi_violations, firing_rate

from spikemetrics.tests.utils import simulated_spike_train



def test_calculate_amplitude_cutoff():
    pass


def test_calculate_drift_metrics():
    pass


def test_calculate_firing_rate_and_spikes():
    pass


def test_calculate_isi_violations():

    pass


def test_calculate_pc_metrics():
    pass


def test_calculate_silhouette_score():
    pass


def test_calculate_presence_ratio():
    pass


def test_calculate_metrics():
    
    pass


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

    assert firing_rate(train, min_time=0, max_time=100) > firing_rate(train, min_time=0, max_time =200)

