import numpy as np
import pytest

from spikemetrics import (calculate_amplitude_cutoff,
                          calculate_drift_metrics,
                          calculate_firing_rates,
                          calculate_isi_violations,
                          calculate_pc_metrics,
                          calculate_silhouette_score,
                          calculate_presence_ratio,
                          calculate_metrics,
                          calculate_num_spikes)

from spikemetrics.metrics import (isi_violations,
                                  firing_rate,
                                  presence_ratio,
                                  amplitude_cutoff,
                                  mahalanobis_metrics,
                                  lda_metrics,
                                  nearest_neighbors_metrics,
                                  make_index_mask,
                                  make_channel_mask,
                                  get_unit_pcs)

from spikemetrics.tests.utils import (simulated_spike_train,
                                      simulated_spike_amplitudes,
                                      simulated_pcs_for_one_unit,
                                      create_ground_truth_pc_distributions)


@pytest.fixture
def simulated_spike_times():
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

    return {'spike_times': spike_times, 'spike_clusters': spike_clusters}


@pytest.fixture
def simulated_amplitudes():
    num_spikes = 5000

    amps = [simulated_spike_amplitudes(num_spikes, center, 30) for center in [100, 50, 20, 5]]

    labels = [np.ones((len(amps[i]),), dtype='int') * i for i in range(len(amps))]

    spike_amplitudes = np.concatenate(amps)
    spike_clusters = np.concatenate(labels)

    return {'spike_amplitudes': spike_amplitudes, 'spike_clusters': spike_clusters}


@pytest.fixture
def simulated_drift_pcs():
    num_spikes = 5000

    pc_features = [simulated_pcs_for_one_unit(num_spikes, 32, 5, end_channel) for end_channel in [5, 10, 15, 20]]

    labels = [np.ones((pc_features[i].shape[0],), dtype='int') * i for i in range(len(pc_features))]
    times = [np.linspace(0, 100, pc_features[i].shape[0], ) for i in range(len(pc_features))]

    pc_feature_ind = np.tile(np.arange(32, dtype='int'), (len(pc_features), 1))

    pc_features = np.concatenate(pc_features, axis=0)
    spike_clusters = np.concatenate(labels)
    spike_times = np.concatenate(times)

    return {'pc_features': pc_features,
            'pc_feature_ind': pc_feature_ind,
            'spike_clusters': spike_clusters,
            'spike_times': spike_times}


def test_calculate_metrics():
    pass


def test_calculate_silhouette_score():
    pc_features, spike_clusters = create_ground_truth_pc_distributions([1, -1, 10, 20], [1000, 1000, 500, 20])
    pc_feature_ind = np.zeros((4, 1), dtype='int')
    total_units = 4
    pc_features = np.expand_dims(pc_features, axis=2)

    ss = calculate_silhouette_score(spike_clusters,
                                    total_units,
                                    pc_features,
                                    pc_feature_ind,
                                    1000,
                                    verbose=False)

    assert np.sum(np.isnan(ss)) == 0


def test_calculate_pc_metrics():
    pc_features, spike_clusters = create_ground_truth_pc_distributions([1, -1, 10, 20], [1000, 1000, 500, 20])
    pc_feature_ind = np.zeros((4, 1), dtype='int')
    total_units = 4
    pc_features = np.expand_dims(pc_features, axis=2)

    isolation_distances, l_ratios, d_primes, nn_hit_rates, nn_miss_rates = \
        calculate_pc_metrics(spike_clusters,
                             total_units,
                             pc_features,
                             pc_feature_ind,
                             1,
                             500,
                             1000,
                             3,
                             verbose=False)

    assert np.sum(np.isnan(isolation_distances)) == 0
    assert np.sum(np.isnan(l_ratios)) == 0
    assert np.sum(np.isnan(d_primes)) == 0
    assert np.sum(np.isnan(nn_hit_rates)) == 0
    assert np.sum(np.isnan(nn_miss_rates)) == 0


def test_mahalanobis_metrics():
    all_pcs1, all_labels1 = create_ground_truth_pc_distributions([1, -1], [1000, 1000])
    all_pcs2, all_labels2 = create_ground_truth_pc_distributions([1, -2],
                                                                 [1000, 1000])  # increase distance between clusters

    isolation_distance1, l_ratio1 = mahalanobis_metrics(all_pcs1, all_labels1, 0)
    isolation_distance2, l_ratio2 = mahalanobis_metrics(all_pcs2, all_labels2, 0)

    assert isolation_distance1 < isolation_distance2
    assert l_ratio1 > l_ratio2


def test_lda_metrics():
    all_pcs1, all_labels1 = create_ground_truth_pc_distributions([1, -1], [1000, 1000])
    all_pcs2, all_labels2 = create_ground_truth_pc_distributions([1, -2],
                                                                 [1000, 1000])  # increase distance between clusters

    d_prime1 = lda_metrics(all_pcs1, all_labels1, 0)
    d_prime2 = lda_metrics(all_pcs2, all_labels2, 0)

    assert d_prime1 < d_prime2


def test_nearest_neighbors_metrics():
    all_pcs1, all_labels1 = create_ground_truth_pc_distributions([1, -1], [1000, 1000])
    all_pcs2, all_labels2 = create_ground_truth_pc_distributions([1, -2],
                                                                 [1000, 1000])  # increase distance between clusters

    hit_rate1, miss_rate1 = nearest_neighbors_metrics(all_pcs1, all_labels1, 0, 1000, 3)
    hit_rate2, miss_rate2 = nearest_neighbors_metrics(all_pcs2, all_labels2, 0, 1000, 3)

    assert hit_rate1 < hit_rate2
    assert miss_rate1 > miss_rate2


@pytest.mark.parametrize(
    "num_total_spikes,num_selected_spikes",
    [
        [100, 0], [500, 500], [1000, 500]
    ],
)
def test_make_index_mask(num_total_spikes, num_selected_spikes):
    spike_clusters = np.ones((num_total_spikes,), dtype='int')
    unit_id = 1

    index_mask = make_index_mask(spike_clusters, unit_id, 200, 500, seed=0)

    assert np.sum(index_mask) == num_selected_spikes


def test_make_channel_mask():
    pc_feature_ind = np.tile(np.arange(32), (4, 1))
    pc_feature_ind[0, :] = pc_feature_ind[0, np.random.permutation(32)]
    unit_id = 0
    channels_to_use = np.arange(10)

    channel_mask = make_channel_mask(unit_id, pc_feature_ind, channels_to_use)

    sorted_channel_mask = np.sort(channel_mask)
    expected_result = np.where(np.isin(pc_feature_ind[unit_id, :], channels_to_use))[0]

    assert np.array_equal(sorted_channel_mask, expected_result)


def test_get_unit_pcs():
    original_spike_count = 1000
    masked_spike_count = 100

    original_channel_count = 32
    masked_channel_count = 10

    num_pc_features = 3

    these_pc_features = np.ones((original_spike_count, num_pc_features, original_channel_count))

    channel_mask = np.arange(masked_channel_count)
    index_mask = np.zeros((original_spike_count,), dtype='bool')
    index_mask[:masked_spike_count] = True

    unit_PCs = get_unit_pcs(these_pc_features, index_mask, channel_mask)

    assert unit_PCs.shape == (masked_spike_count, num_pc_features, masked_channel_count)


def test_calculate_drift_metrics(simulated_drift_pcs):
    max_drift, cumulative_drift = calculate_drift_metrics(simulated_drift_pcs['spike_times'],
                                                          simulated_drift_pcs['spike_clusters'],
                                                          4,
                                                          simulated_drift_pcs['pc_features'],
                                                          simulated_drift_pcs['pc_feature_ind'],
                                                          interval_length=10,
                                                          min_spikes_per_interval=1, vertical_channel_spacing=1,
                                                          channel_locations=None)

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


def test_calculate_presence_ratio(simulated_spike_times):
    ratios = calculate_presence_ratio(simulated_spike_times['spike_times'],
                                      simulated_spike_times['spike_clusters'],
                                      3,
                                      duration=simulated_spike_times['spike_times'][-1],
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

    assert presence_ratio(spike_times, overall_duration) == expected_value


def test_calculate_isi_violations(simulated_spike_times):
    viol = calculate_isi_violations(simulated_spike_times['spike_times'],
                                    simulated_spike_times['spike_clusters'],
                                    3, 0.001, 0.0, duration=simulated_spike_times['spike_times'][-1], verbose=False)
    assert np.allclose(viol, np.array([0.0996012, 0.78735198, 1.92233756]))


def test_isi_violations():
    # 1. check value for fixed spike train parameters:

    train1 = simulated_spike_train(100, 10, 10)
    fpRate1, num_violations1 = isi_violations(train1, np.max(train1), 0.001)

    assert np.isclose(fpRate1, 0.4901480247, rtol=0, atol=1e-5)
    assert num_violations1 == 10

    # 2. check that the value doesn't depend on recording duration:

    train2 = simulated_spike_train(200, 10, 20)
    fpRate2, num_violations2 = isi_violations(train2, np.max(train2), 0.001)

    assert np.isclose(fpRate1, fpRate2, rtol=0, atol=1e-5)

    # 3. check that the value increases with the number of violations:

    train3 = simulated_spike_train(100, 10, 20)
    fpRate3, num_violations3 = isi_violations(train3, np.max(train3), 0.001)

    assert fpRate3 > fpRate1

    # 4. check that the value decreases with a longer violation window:

    fpRate4, num_violations4 = isi_violations(train1, np.max(train1), 0.002)

    assert fpRate4 < fpRate1

    # 5. check that the value decreases with firing rate:

    train4 = simulated_spike_train(100, 20, 10)
    fpRate5, num_violations5 = isi_violations(train4, np.max(train4), 0.001)

    assert fpRate5 < fpRate1


def test_calculate_firing_rate_num_spikes(simulated_spike_times):
    firing_rates = calculate_firing_rates(simulated_spike_times['spike_times'],
                                          simulated_spike_times['spike_clusters'],
                                          3, duration=simulated_spike_times['spike_times'][-1], verbose=False)
    num_spikes = calculate_num_spikes(simulated_spike_times['spike_times'],
                                      simulated_spike_times['spike_clusters'],
                                      3, verbose=False)
    assert np.allclose(firing_rates, np.array([10.02, 5.04, 5.1]))
    assert np.allclose(num_spikes, np.array([1002, 504, 510]))


def test_firing_rate():
    # 1. check that the output value is correct:

    max_time = 100
    simulated_firing_rate = 10.0

    train = simulated_spike_train(max_time, simulated_firing_rate, 0)

    assert firing_rate(train, duration=max_time) == simulated_firing_rate

    # 2. check that widening the boundaries decreases the rate:

    assert firing_rate(train, duration=100) > firing_rate(train, duration=200)
