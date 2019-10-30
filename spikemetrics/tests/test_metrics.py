import numpy as np
import pytest

from spikemetrics import calculate_amplitude_cutoff, calculate_drift_metrics, calculate_firing_rate_and_spikes, \
    calculate_isi_violations, calculate_pc_metrics, calculate_silhouette_score, calculate_presence_ratio, \
    calculate_metrics

from spikemetrics.metrics import isi_violations, firing_rate, presence_ratio, amplitude_cutoff

from spikemetrics.tests.utils import (simulated_spike_train, 
                                      simulated_spike_amplitudes,
                                      simulated_pcs_for_one_unit)

@pytest.fixture
def simulated_spikes():

    max_time = 100

    trains = [simulated_spike_train(max_time, 10, 2),
              simulated_spike_train(max_time, 5, 4),
              simulated_spike_train(max_time, 5, 10)]

    labels = [np.ones((len(trains[i]),), dtype='int') * i for i in range(len(trains))]

    spike_times = np.concatenate(trains)
    spike_clusters = np.concatenate(labels)

    order = np.argsort(spike_times)

    spike_times = spike_times[order]
    spike_clusters = spike_clusters[order]

    return {'spike_times' : spike_times, 'spike_clusters' : spike_clusters}

@pytest.fixture
def simulated_amplitudes():

    num_spikes = 5000

    amps = [simulated_spike_amplitudes(num_spikes, center, 30) for center in [100, 50, 20, 5]]

    labels = [np.ones((len(amps[i]),), dtype='int') * i for i in range(len(amps))]

    spike_amplitudes = np.concatenate(amps)
    spike_clusters = np.concatenate(labels)

    return {'spike_amplitudes' : spike_amplitudes, 'spike_clusters' : spike_clusters}


@pytest.fixture
def simulated_drift_pcs():

    num_spikes = 5000

    pc_features = [simulated_pcs_for_one_unit(num_spikes, 32, 5, end_channel) for end_channel in [5, 10, 15, 20]]

    labels = [np.ones((pc_features[i].shape[0],), dtype='int') * i for i in range(len(pc_features))]
    times = [np.linspace(0, 100, pc_features[i].shape[0],) for i in range(len(pc_features))]
    
    pc_feature_ind = np.tile(np.arange(32, dtype='int'), (len(pc_features),1))

    pc_features = np.concatenate(pc_features, axis=0)
    spike_clusters = np.concatenate(labels)
    spike_times = np.concatenate(times)

    return {'pc_features' : pc_features, 
            'pc_feature_ind' : pc_feature_ind,
            'spike_clusters' : spike_clusters,
            'spike_times' : spike_times}


def test_calculate_metrics():
    
    pass


def test_calculate_pc_metrics():
    pass


def test_calculate_silhouette_score():
    pass


def test_calculate_drift_metrics(simulated_drift_pcs):

    max_drift, cumulative_drift = calculate_drift_metrics(simulated_drift_pcs['spike_times'],
                            simulated_drift_pcs['spike_clusters'],
                            4,
                            simulated_drift_pcs['pc_features'],
                            simulated_drift_pcs['pc_feature_ind'],
                            10,1,1)

    assert np.allclose(max_drift, np.array([0, 4.5, 9, 13.5]))
    assert np.allclose(cumulative_drift, np.array([0, 4.5, 9, 13.5]))


def test_calculate_amplitude_cutoff(simulated_amplitudes):


    amp_cuts = calculate_amplitude_cutoff(simulated_amplitudes['spike_clusters'], 
                                          simulated_amplitudes['spike_amplitudes'], 
                                          4, 
                                          verbose=False)

    assert np.allclose(amp_cuts, np.array([0.0015467, 0.03828989, 0.36565474, 0.5]), rtol=0, atol=1e-5)


def test_amplitude_cutoff(): 

    amplitudes = simulated_spike_amplitudes(5000, 100, 30)

    amp_cut = amplitude_cutoff(amplitudes)

    assert np.isclose(amp_cut, 0.001546, rtol=0, atol=1e-5)



def test_calculate_presence_ratio(simulated_spikes):

    ratios = calculate_presence_ratio(simulated_spikes['spike_times'], 
                                    simulated_spikes['spike_clusters'], 
                                    3,
                                    verbose=False)

    assert np.allclose(ratios, np.array([1.0, 1.0, 1.0]))


@pytest.mark.parametrize(
    "overall_duration,expected_value",
    [
        [100, 1.0], [150, 0.67], [200, 0.5], [600, 0.17]
    ],
)
def test_presence_ratio(overall_duration, expected_value):

    spike_times = simulated_spike_train(100, 20, 0)

    assert presence_ratio(spike_times, 0, overall_duration) == expected_value




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


    # 4. check that the value decreases with a longer violation window:

    fpRate4, num_violations4 = isi_violations(train1, 0, np.max(train1), 0.002)

    assert fpRate4 < fpRate1


    # 5. check that the value decreases with firing rate:

    train4 = simulated_spike_train(100, 20, 10)
    fpRate5, num_violations5 = isi_violations(train4, 0, np.max(train4), 0.001)

    assert fpRate5 < fpRate1



def test_calculate_firing_rate_and_spikes(simulated_spikes):

    firing_rates, spike_counts = calculate_firing_rate_and_spikes(simulated_spikes['spike_times'], 
                                    simulated_spikes['spike_clusters'], 
                                    3, verbose=False)

    print(firing_rates)

    assert np.allclose(firing_rates, np.array([10.03003003,  5.04504505,  5.10510511]))
    assert np.allclose(spike_counts, np.array([1002, 504, 510]))


def test_firing_rate():

    # 1. check that the output value is correct:

    max_time = 100
    simulated_firing_rate = 10.0

    train = simulated_spike_train(max_time, simulated_firing_rate, 0)

    assert firing_rate(train, min_time=0, max_time=max_time) == simulated_firing_rate

    # 2. check that widening the boundaries decreases the rate:

    assert firing_rate(train, min_time=0, max_time=100) > firing_rate(train, min_time=0, max_time=200)

