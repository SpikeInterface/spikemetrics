# Copyright © 2019. Allen Institute.  All rights reserved.

import numpy as np
import pandas as pd
from collections import OrderedDict
import math
import warnings

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score

from scipy.spatial.distance import cdist
from scipy.stats import chi2
from scipy.ndimage.filters import gaussian_filter1d

from .utils import Epoch
from .utils import printProgressBar, get_spike_positions


def calculate_metrics(spike_times, spike_clusters, amplitudes, pc_features, pc_feature_ind, params,
                      duration, cluster_ids=None, epochs=None, seed=None, verbose=True):
    """ Calculate metrics for all units on one probe

    Inputs:
    ------
    spike_times : numpy.ndarray (num_spikes x 0)
        Spike times in seconds (same timebase as epochs)
    spike_clusters : numpy.ndarray (num_spikes x 0)
        Cluster IDs for each spike time
    pc_features : numpy.ndarray (num_spikes x num_pcs x num_channels)
        Pre-computed PCs for blocks of channels around each spike
    pc_feature_ind : numpy.ndarray (num_units x num_channels)
        Channel indices of PCs for each unit
    epochs : list of Epoch objects
        contains information on Epoch start and stop times
    duration : length of recording (seconds)
    params : dict of parameters
        'isi_threshold' : minimum time for isi violations
        'min_isi'
        'num_channels_to_compare'
        'max_spikes_for_unit'
        'max_spikes_for_nn'
        'n_neighbors'
        'drift_metrics_interval_s'
        'drift_metrics_min_spikes_per_interval'


    Outputs:
    --------
    metrics : pandas.DataFrame
        one column for each metric
        one row per unit per epoch

    """

    metrics = pd.DataFrame()

    if epochs is None:
        epochs = [Epoch('complete_session', 0, np.inf)]

    total_units = np.max(spike_clusters) + 1
    total_epochs = len(epochs)

    for epoch in epochs:
        in_epoch = np.logical_and(spike_times >= epoch.start_time, spike_times < epoch.end_time)
        spikes_in_epoch = np.sum(in_epoch)
        spikes_for_nn = min(spikes_in_epoch, params['max_spikes_for_nn'])
        spikes_for_silhouette = min(spikes_in_epoch, params['n_silhouette'])

        print("Calculating isi violations")
        isi_viol = calculate_isi_violations(spike_times[in_epoch], spike_clusters[in_epoch], total_units,
                                            duration, params['isi_threshold'], params['min_isi'], verbose=verbose)

        print("Calculating presence ratio")
        presence_ratio = calculate_presence_ratio(spike_times[in_epoch], spike_clusters[in_epoch], total_units,
                                                  duration, verbose=verbose)

        print("Calculating firing rate")
        firing_rate = calculate_firing_rates(spike_times[in_epoch], spike_clusters[in_epoch],
                                             duration, total_units, verbose=verbose)

        print("Calculating amplitude cutoff")
        amplitude_cutoff = calculate_amplitude_cutoff(spike_clusters[in_epoch], amplitudes[in_epoch], total_units,
                                                      verbose=verbose)

        print("Calculating PC-based metrics")
        isolation_distance, l_ratio, d_prime, nn_hit_rate, nn_miss_rate = calculate_pc_metrics(spike_clusters[in_epoch],
                                                                                               total_units,
                                                                                               pc_features[in_epoch, :,
                                                                                               :],
                                                                                               pc_feature_ind,
                                                                                               params[
                                                                                                   'num_channels_to_compare'],
                                                                                               params[
                                                                                                   'max_spikes_for_unit'],
                                                                                               spikes_for_nn,
                                                                                               params['n_neighbors'],
                                                                                               seed=seed,
                                                                                               verbose=verbose)

        print("Calculating silhouette score")
        silhouette_score = calculate_silhouette_score(spike_clusters[in_epoch],
                                                      total_units,
                                                      pc_features[in_epoch, :, :],
                                                      pc_feature_ind,
                                                      spikes_for_silhouette,
                                                      seed=seed, verbose=verbose)

        print("Calculating drift metrics")
        max_drift, cumulative_drift = calculate_drift_metrics(spike_times[in_epoch],
                                                              spike_clusters[in_epoch],
                                                              total_units,
                                                              pc_features[in_epoch, :, :],
                                                              pc_feature_ind,
                                                              params['drift_metrics_interval_s'],
                                                              params['drift_metrics_min_spikes_per_interval'],
                                                              verbose=verbose)
        if cluster_ids is None:
            cluster_ids_out = np.arange(total_units)
        else:
            cluster_ids_out = cluster_ids
        epoch_name = [epoch.name] * len(cluster_ids_out)

        metrics = pd.concat((metrics, pd.DataFrame(data=OrderedDict((('cluster_id', cluster_ids_out),
                                                                     ('firing_rate', firing_rate),
                                                                     ('presence_ratio', presence_ratio),
                                                                     ('isi_violation', isi_viol),
                                                                     ('amplitude_cutoff', amplitude_cutoff),
                                                                     ('isolation_distance', isolation_distance),
                                                                     ('l_ratio', l_ratio),
                                                                     ('d_prime', d_prime),
                                                                     ('nn_hit_rate', nn_hit_rate),
                                                                     ('nn_miss_rate', nn_miss_rate),
                                                                     ('silhouette_score', silhouette_score),
                                                                     ('max_drift', max_drift),
                                                                     ('cumulative_drift', cumulative_drift),
                                                                     ('epoch_name', epoch_name),
                                                                     )))))

    return metrics


# ===============================================================

# HELPER FUNCTIONS TO LOOP THROUGH CLUSTERS:

# ===============================================================

def calculate_isi_violations(spike_times, spike_clusters, total_units, isi_threshold, min_isi, duration, spike_cluster_subset=None,  verbose=True):
    if spike_cluster_subset is not None:
        cluster_ids = spike_cluster_subset
    else: 
        cluster_ids = np.unique(spike_clusters)

    viol_rates = np.zeros((total_units,))

    for idx, cluster_id in enumerate(cluster_ids):

        if verbose:
            printProgressBar(cluster_id + 1, total_units)

        for_this_cluster = (spike_clusters == cluster_id)
        viol_rates[cluster_id], num_violations = isi_violations(spike_times[for_this_cluster],
                                                                duration=duration,
                                                                isi_threshold=isi_threshold,
                                                                min_isi=min_isi)

    return viol_rates


def calculate_presence_ratio(spike_times, spike_clusters, total_units, duration, spike_cluster_subset=None, verbose=True):
    if spike_cluster_subset is not None:
        cluster_ids = spike_cluster_subset
    else: 
        cluster_ids = np.unique(spike_clusters)

    ratios = np.zeros((total_units,))

    for idx, cluster_id in enumerate(cluster_ids):

        if verbose:
            printProgressBar(cluster_id + 1, total_units)

        for_this_cluster = (spike_clusters == cluster_id)
        ratios[cluster_id] = presence_ratio(spike_times[for_this_cluster],
                                            duration=duration)

    return ratios



def calculate_num_spikes(spike_times, spike_clusters, total_units, spike_cluster_subset=None, verbose=True):
    num_spikes = np.zeros((total_units,))
    if spike_cluster_subset is not None:
        cluster_ids = spike_cluster_subset
    else: 
        cluster_ids = np.unique(spike_clusters)

    for idx, cluster_id in enumerate(cluster_ids):

        if verbose:
            printProgressBar(cluster_id + 1, total_units)

        for_this_cluster = (spike_clusters == cluster_id)
        num_spikes[cluster_id] = len(spike_times[for_this_cluster])

    return num_spikes


def calculate_firing_rates(spike_times, spike_clusters, total_units, duration, spike_cluster_subset=None, verbose=True):
    if spike_cluster_subset is not None:
        cluster_ids = spike_cluster_subset
    else: 
        cluster_ids = np.unique(spike_clusters)
    
    firing_rates = np.zeros((total_units,))

    for idx, cluster_id in enumerate(cluster_ids):

        if verbose:
            printProgressBar(cluster_id + 1, total_units)

        for_this_cluster = (spike_clusters == cluster_id)
        firing_rates[cluster_id] = firing_rate(spike_times[for_this_cluster],
                                               duration=duration)

    return firing_rates


def calculate_amplitude_cutoff(spike_clusters, amplitudes, total_units, spike_cluster_subset=None,  verbose=True):
    if spike_cluster_subset is not None:
        cluster_ids = spike_cluster_subset
    else: 
        cluster_ids = np.unique(spike_clusters)

    amplitude_cutoffs = np.zeros((total_units,))

    for idx, cluster_id in enumerate(cluster_ids):

        if verbose:
            printProgressBar(cluster_id + 1, total_units)

        for_this_cluster = (spike_clusters == cluster_id)
        amplitude_cutoffs[cluster_id] = amplitude_cutoff(amplitudes[for_this_cluster])

    return amplitude_cutoffs


def calculate_pc_metrics(spike_clusters,
                         total_units,
                         pc_features,
                         pc_feature_ind,
                         num_channels_to_compare,
                         max_spikes_for_cluster,
                         spikes_for_nn,
                         n_neighbors,
                         min_num_pcs=10,
                         metric_names=None,
                         seed=None, 
                         spike_cluster_subset=None,  
                         verbose=True):
    assert (num_channels_to_compare % 2 == 1)
    half_spread = int((num_channels_to_compare - 1) / 2)

    if metric_names is None:
        metric_names = ['isolation_distance', 'l_ratio', 'd_prime', 'nearest_neighbor']

    all_cluster_ids = np.unique(spike_clusters)
    if spike_cluster_subset is not None:
        cluster_ids = spike_cluster_subset
    else: 
        cluster_ids = all_cluster_ids

    peak_channels = np.zeros((total_units,), dtype='uint16')
    isolation_distances = np.zeros((total_units,))
    l_ratios = np.zeros((total_units,))
    d_primes = np.zeros((total_units,))
    nn_hit_rates = np.zeros((total_units,))
    nn_miss_rates = np.zeros((total_units,))

    for idx, cluster_id in enumerate(all_cluster_ids):
        for_unit = np.squeeze(spike_clusters == cluster_id)
        pc_max = np.argmax(np.mean(pc_features[for_unit, 0, :], 0))
        peak_channels[cluster_id] = pc_feature_ind[cluster_id, pc_max]

    for idx, cluster_id in enumerate(cluster_ids):

        if verbose:
            printProgressBar(cluster_id + 1, total_units)

        peak_channel = peak_channels[cluster_id]

        half_spread_down = peak_channel \
            if peak_channel < half_spread \
            else half_spread

        half_spread_up = np.max(pc_feature_ind) - peak_channel \
            if peak_channel + half_spread > np.max(pc_feature_ind) \
            else half_spread

        units_for_channel, channel_index = np.unravel_index(np.where(pc_feature_ind.flatten() == peak_channel)[0],
                                                            pc_feature_ind.shape)

        units_in_range = (peak_channels[units_for_channel] >= peak_channel - half_spread_down) * \
                         (peak_channels[units_for_channel] <= peak_channel + half_spread_up)

        units_for_channel = units_for_channel[units_in_range]
        channel_index = channel_index[units_in_range]

        channels_to_use = np.arange(peak_channel - half_spread_down, peak_channel + half_spread_up + 1)

        spike_counts = np.zeros(units_for_channel.shape)

        for idx2, cluster_id2 in enumerate(units_for_channel):
            spike_counts[idx2] = np.sum(spike_clusters == cluster_id2)

        this_unit_idx = np.where(units_for_channel == cluster_id)[0]

        if spike_counts[this_unit_idx] > max_spikes_for_cluster:
            relative_counts = spike_counts / spike_counts[this_unit_idx] * max_spikes_for_cluster
        else:
            relative_counts = spike_counts

        all_pcs = np.zeros((0, pc_features.shape[1], channels_to_use.size))
        all_labels = np.zeros((0,))

        for idx2, cluster_id2 in enumerate(units_for_channel):

            try:
                channel_mask = make_channel_mask(cluster_id2, pc_feature_ind, channels_to_use)
            except IndexError:
                # Occurs when pc_feature_ind does not contain all channels of interest
                # In that case, we will exclude this unit for the calculation
                pass
            else:
                subsample = int(relative_counts[idx2])
                index_mask = make_index_mask(spike_clusters, cluster_id2, min_num=0, max_num=subsample, seed=seed)
                pcs = get_unit_pcs(pc_features, index_mask, channel_mask)
                labels = np.ones((pcs.shape[0],)) * cluster_id2

                all_pcs = np.concatenate((all_pcs, pcs), 0)
                all_labels = np.concatenate((all_labels, labels), 0)

        all_pcs = np.reshape(all_pcs, (all_pcs.shape[0], pc_features.shape[1] * channels_to_use.size))

        if all_pcs.shape[0] > min_num_pcs:

            if 'isolation_distance' in metric_names or 'l_ratio' in metric_names:
                isolation_distances[cluster_id], l_ratios[cluster_id] = mahalanobis_metrics(all_pcs, all_labels,
                                                                                            cluster_id)
            else:
                isolation_distances[cluster_id] = np.nan
                l_ratios[cluster_id] = np.nan

            if 'd_prime' in metric_names:
                d_primes[cluster_id] = lda_metrics(all_pcs, all_labels, cluster_id)
            else:
                d_primes[cluster_id] = np.nan

            if 'nearest_neighbor' in metric_names:
                nn_hit_rates[cluster_id], nn_miss_rates[cluster_id] = nearest_neighbors_metrics(all_pcs, all_labels,
                                                                                                cluster_id,
                                                                                                spikes_for_nn,
                                                                                                n_neighbors)
            else:
                nn_hit_rates[cluster_id] = np.nan
                nn_miss_rates[cluster_id] = np.nan
        else:

            isolation_distances[cluster_id] = np.nan
            l_ratios[cluster_id] = np.nan
            d_primes[cluster_id] = np.nan
            nn_hit_rates[cluster_id] = np.nan
            nn_miss_rates[cluster_id] = np.nan

    return isolation_distances, l_ratios, d_primes, nn_hit_rates, nn_miss_rates


def calculate_silhouette_score(spike_clusters,
                               total_units,
                               pc_features,
                               pc_feature_ind,
                               spikes_for_silhouette,
                               seed=None, 
                               spike_cluster_subset=None, 
                               verbose=True):
    random_spike_inds = np.random.RandomState(seed=seed).permutation(spike_clusters.size)
    random_spike_inds = random_spike_inds[:spikes_for_silhouette]
    num_pc_features = pc_features.shape[1]
    num_channels = np.max(pc_feature_ind) + 1

    all_pcs = np.zeros((spikes_for_silhouette, num_channels * num_pc_features))

    for idx, i in enumerate(random_spike_inds):

        unit_id = spike_clusters[i]
        channels = pc_feature_ind[unit_id, :]

        for j in range(0, num_pc_features):
            all_pcs[idx, channels + num_channels * j] = pc_features[i,j,:]

    cluster_labels = spike_clusters[random_spike_inds]

    all_cluster_ids = np.unique(spike_clusters)
    if spike_cluster_subset is not None:
        cluster_ids = spike_cluster_subset
    else: 
        cluster_ids = all_cluster_ids

    SS = np.empty((total_units, total_units))
    SS[:] = np.nan

    seen_unit_pairs = set()
    for idx1, i in enumerate(cluster_ids):

        if verbose:
            printProgressBar(idx1 + 1, len(cluster_ids))

        for idx2, j in enumerate(all_cluster_ids):
            if (i,j) not in seen_unit_pairs and (j,i) not in seen_unit_pairs and i != j:
                inds = np.in1d(cluster_labels, np.array([i, j]))
                X = all_pcs[inds, :]
                labels = cluster_labels[inds]
                if len(labels) > 2:
                    SS[i, j] = silhouette_score(X, labels, random_state=seed)
                seen_unit_pairs.add((i,j))

    with warnings.catch_warnings():
      warnings.simplefilter("ignore")
      a = np.nanmin(SS, 0)
      b = np.nanmin(SS, 1)

    return np.array([np.nanmin([a,b]) for a, b in zip(a,b)])


def calculate_drift_metrics(spike_times,
                            spike_clusters,
                            total_units,
                            pc_features,
                            pc_feature_ind,
                            interval_length,
                            min_spikes_per_interval,
                            vertical_channel_spacing=10,
                            channel_locations=None,
                            spike_cluster_subset=None,
                            verbose=True):

    max_drift = np.zeros((total_units,))
    cumulative_drift = np.zeros((total_units,))

    positions = get_spike_positions(spike_clusters, pc_features, pc_feature_ind, channel_locations,
                                    vertical_channel_spacing)
    interval_starts = np.arange(np.min(spike_times), np.max(spike_times), interval_length)
    interval_ends = interval_starts + interval_length

    if spike_cluster_subset is not None:
        cluster_ids = spike_cluster_subset
    else: 
        cluster_ids = np.unique(spike_clusters)

    for idx, cluster_id in enumerate(cluster_ids):

        if verbose:
            printProgressBar(cluster_id + 1, len(cluster_ids))

        in_cluster = spike_clusters == cluster_id
        times_for_cluster = spike_times[in_cluster]
        positions_for_cluster = positions[in_cluster]

        median_positions = []

        for t1, t2 in zip(interval_starts, interval_ends):
            in_range = (times_for_cluster > t1) * (times_for_cluster < t2)

            if np.sum(in_range) >= min_spikes_per_interval:
                median_positions.append(np.median(positions_for_cluster[in_range], 0))
            else:
                median_positions.append([np.nan, np.nan])

        median_positions = np.array(median_positions)

        # Extract emi-matrix of shifts in positions (used to extract max_drift and cum_drift)
        position_diffs = np.zeros((len(median_positions), len(median_positions)))
        for i, pos_i in enumerate(median_positions):
            for j, pos_j in enumerate(median_positions):
                if j > i:
                    if not np.isnan(pos_i[0]) and not np.isnan(pos_j[0]):
                        position_diffs[i, j] = np.linalg.norm(pos_i - pos_j)
                    else:
                        position_diffs[i, j] = 0

        # Maximum drift among all periods
        if np.any(position_diffs > 0):
            max_drift[cluster_id] = np.around(np.max(position_diffs[position_diffs > 0]), 2)
            # The +1 diagonal contains the step-by-step drifts between intervals.
            # Summing them up we obtain cumulative drift
            cumulative_drift[cluster_id] = np.around(np.sum(np.diag(position_diffs, 1)), 2)
        else:
            # not enough spikes
            max_drift[cluster_id] = 0
            cumulative_drift[cluster_id] = 0

    return max_drift, cumulative_drift


# ==========================================================

# IMPLEMENTATION OF ACTUAL METRICS:

# ==========================================================


def isi_violations(spike_train, duration, isi_threshold, min_isi=0):
    """Calculate Inter-Spike Interval (ISI) violations for a spike train.

    Based on metric described in Hill et al. (2011) J Neurosci 31: 8699-8705

    Originally written in Matlab by Nick Steinmetz (https://github.com/cortex-lab/sortingQuality)
    Converted to Python by Daniel Denman

    Inputs:
    -------
    spike_train : array of monotonically increasing spike times (in seconds) [t1, t2, t3, ...]
    duration : length of recording (seconds)
    isi_threshold : threshold for classifying adjacent spikes as an ISI violation
      - this is the biophysical refractory period
    min_isi : minimum possible inter-spike interval (default = 0)
      - this is the artificial refractory period enforced by the data acquisition system
        or post-processing algorithms

    Outputs:
    --------
    fpRate : rate of contaminating spikes as a fraction of overall rate
      - higher values indicate more contamination
    num_violations : total number of violations detected

    """
    isis_initial = np.diff(spike_train)

    if min_isi > 0:
        duplicate_spikes = np.where(isis_initial <= min_isi)[0]
        spike_train = np.delete(spike_train, duplicate_spikes + 1)

    isis = np.diff(spike_train)
    num_spikes = len(spike_train)
    num_violations = sum(isis < isi_threshold)
    violation_time = 2 * num_spikes * (isi_threshold - min_isi)
    total_rate = firing_rate(spike_train, duration)
    violation_rate = num_violations / violation_time
    fpRate = violation_rate / total_rate

    return fpRate, num_violations


def presence_ratio(spike_train, duration, num_bin_edges=101):
    """Calculate fraction of time the unit is present within an epoch.

    Inputs:
    -------
    spike_train : array of spike times
    duration : length of recording (seconds)
    num_bin_edges : number of bin edges for histogram
      - total bins = num_bin_edges - 1

    Outputs:
    --------
    presence_ratio : fraction of time bins in which this unit is spiking

    """

    h, b = np.histogram(spike_train, np.linspace(0, duration, num_bin_edges))

    return np.sum(h > 0) / (num_bin_edges - 1)


def firing_rate(spike_train, duration):
    """Calculate firing rate for a spike train.

    If either temporal bound is not specified, the first and last spike time are used by default.

    Inputs:
    -------
    spike_train : array of spike times (in seconds)
    duration : length of recording (in seconds)

    Outputs:
    --------
    fr : float
        Firing rate in Hz

    """
    fr = spike_train.size / duration

    return fr


def amplitude_cutoff(amplitudes, num_histogram_bins=500, histogram_smoothing_value=3):
    """ Calculate approximate fraction of spikes missing from a distribution of amplitudes

    Assumes the amplitude histogram is symmetric (not valid in the presence of drift)

    Inspired by metric described in Hill et al. (2011) J Neurosci 31: 8699-8705

    Input:
    ------
    amplitudes : numpy.ndarray
        Array of amplitudes (don't need to be in physical units)
    num_histogram_bins : int
        Number of bins for calculating amplitude histogram
    histogram_smoothing_value : float
        Gaussian filter window for smoothing amplitude histogram

    Output:
    -------
    fraction_missing : float
        Fraction of missing spikes (ranges between 0 and 0.5)
        If more than 50% of spikes are missing, an accurate estimate isn't possible

    """

    h, b = np.histogram(amplitudes, num_histogram_bins, density=True)

    pdf = gaussian_filter1d(h, histogram_smoothing_value)
    support = b[:-1]

    peak_index = np.argmax(pdf)
    G = np.argmin(np.abs(pdf[peak_index:] - pdf[0])) + peak_index

    bin_size = np.mean(np.diff(support))
    fraction_missing = np.sum(pdf[G:]) * bin_size

    fraction_missing = np.min([fraction_missing, 0.5])

    return fraction_missing


def mahalanobis_metrics(all_pcs, all_labels, this_unit_id):
    """ Calculates isolation distance and L-ratio (metrics computed from Mahalanobis distance)

    Based on metrics described in Schmitzer-Torbert et al. (2005) Neurosci 131: 1-11

    Inputs:
    -------
    all_pcs : numpy.ndarray (num_spikes x PCs)
        2D array of PCs for all spikes
    all_labels : numpy.ndarray (num_spikes x 0)
        1D array of cluster labels for all spikes
    this_unit_id : Int
        number corresponding to unit for which these metrics will be calculated

    Outputs:
    --------
    isolation_distance : float
        Isolation distance of this unit
    l_ratio : float
        L-ratio for this unit

    """

    pcs_for_this_unit = all_pcs[all_labels == this_unit_id, :]
    pcs_for_other_units = all_pcs[all_labels != this_unit_id, :]

    mean_value = np.expand_dims(np.mean(pcs_for_this_unit, 0), 0)

    try:
        VI = np.linalg.inv(np.cov(pcs_for_this_unit.T))
    except np.linalg.linalg.LinAlgError:  # case of singular matrix
        return np.nan, np.nan

    mahalanobis_other = np.sort(cdist(mean_value,
                                      pcs_for_other_units,
                                      'mahalanobis', VI=VI)[0])

    mahalanobis_self = np.sort(cdist(mean_value,
                                     pcs_for_this_unit,
                                     'mahalanobis', VI=VI)[0])

    n = np.min([pcs_for_this_unit.shape[0], pcs_for_other_units.shape[0]])  # number of spikes

    if n >= 2:
        dof = pcs_for_this_unit.shape[1]  # number of features
        l_ratio = np.sum(1 - chi2.cdf(pow(mahalanobis_other, 2), dof)) / mahalanobis_other.shape[0]
        isolation_distance = pow(mahalanobis_other[n - 1], 2)
        # if math.isnan(l_ratio):
        #     print("NaN detected", mahalanobis_other, VI)
    else:
        l_ratio = np.nan
        isolation_distance = np.nan

    return isolation_distance, l_ratio


def lda_metrics(all_pcs, all_labels, this_unit_id):
    """ Calculates d-prime based on Linear Discriminant Analysis

    Based on metric described in Hill et al. (2011) J Neurosci 31: 8699-8705

    Inputs:
    -------
    all_pcs : numpy.ndarray (num_spikes x PCs)
        2D array of PCs for all spikes
    all_labels : numpy.ndarray (num_spikes x 0)
        1D array of cluster labels for all spikes
    this_unit_id : Int
        number corresponding to unit for which these metrics will be calculated

    Outputs:
    --------
    d_prime : float
        Isolation distance of this unit
    l_ratio : float
        L-ratio for this unit

    """

    X = all_pcs

    y = np.zeros((X.shape[0],), dtype='bool')
    y[all_labels == this_unit_id] = True

    lda = LDA(n_components=1)

    X_flda = lda.fit_transform(X, y)

    flda_this_cluster = X_flda[np.where(y)[0]]
    flda_other_cluster = X_flda[np.where(np.invert(y))[0]]

    d_prime = (np.mean(flda_this_cluster) - np.mean(flda_other_cluster)) / np.sqrt(
        0.5 * (np.std(flda_this_cluster) ** 2 + np.std(flda_other_cluster) ** 2))

    return d_prime


def nearest_neighbors_metrics(all_pcs, all_labels, this_unit_id, spikes_for_nn, n_neighbors):
    """ Calculates unit contamination based on NearestNeighbors search in PCA space

    Based on metrics described in Chung, Magland et al. (2017) Neuron 95: 1381-1394

    Inputs:
    -------
    all_pcs : numpy.ndarray (num_spikes x PCs)
        2D array of PCs for all spikes
    all_labels : numpy.ndarray (num_spikes x 0)
        1D array of cluster labels for all spikes
    this_unit_id : Int
        number corresponding to unit for which these metrics will be calculated
    spikes_for_nn : Int
        number of spikes to use (calculation can be very slow when this number is >20000)
    n_neighbors : Int
        number of neighbors to use

    Outputs:
    --------
    hit_rate : float
        Fraction of neighbors for target cluster that are also in target cluster
    miss_rate : float
        Fraction of neighbors outside target cluster that are in target cluster

    """

    total_spikes = all_pcs.shape[0]
    ratio = spikes_for_nn / total_spikes
    this_unit = all_labels == this_unit_id

    X = np.concatenate((all_pcs[this_unit, :], all_pcs[np.invert(this_unit), :]), 0)

    n = np.sum(this_unit)

    if ratio < 1:
        inds = np.arange(0, X.shape[0] - 1, 1 / ratio).astype('int')
        X = X[inds, :]
        n = int(n * ratio)

    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree').fit(X)
    distances, indices = nbrs.kneighbors(X)

    this_cluster_inds = np.arange(n)

    this_cluster_nearest = indices[:n, 1:].flatten()
    other_cluster_nearest = indices[n:, 1:].flatten()

    hit_rate = np.mean(this_cluster_nearest < n)
    miss_rate = np.mean(other_cluster_nearest < n)

    return hit_rate, miss_rate


# ==========================================================

# HELPER FUNCTIONS:

# ==========================================================

def make_index_mask(spike_clusters, unit_id, min_num, max_num, seed=None):
    """ Create a mask for the spike index dimensions of the pc_features array

    Inputs:
    -------
    spike_clusters : numpy.ndarray (num_spikes x 0)
        Contains cluster IDs for all spikes in pc_features array
    unit_id : Int
        ID for this unit
    min_num : Int
        Minimum number of spikes to return; if there are not enough spikes for this unit, return all False
    max_num : Int
        Maximum number of spikes to return; if too many spikes for this unit, return a random subsample
    seed: int
        Random seed for reproducibility

    Output:
    -------
    index_mask : numpy.ndarray (boolean)
        Mask of spike indices for pc_features array

    """

    index_mask = spike_clusters == unit_id

    inds = np.where(index_mask)[0]

    if len(inds) < min_num:
        index_mask = np.zeros((spike_clusters.size,), dtype='bool')
    else:
        index_mask = np.zeros((spike_clusters.size,), dtype='bool')
        order = np.random.RandomState(seed=seed).permutation(inds.size)
        index_mask[inds[order[:max_num]]] = True

    return index_mask


def make_channel_mask(unit_id, pc_feature_ind, channels_to_use):
    """ Create a mask for the channel dimension of the pc_features array
    Inputs:
    -------
    unit_id : Int
        ID for this unit
    pc_feature_ind : np.ndarray
        Channels used for PC calculation for each unit
    channels_to_use : np.ndarray
        Channels to use for calculating metrics
    Output:
    -------
    channel_mask : numpy.ndarray
        Channel indices to extract from pc_features array

    """

    these_inds = pc_feature_ind[unit_id, :]
    channel_mask = [np.argwhere(these_inds == i)[0][0] for i in channels_to_use]

    return np.array(channel_mask)


def get_unit_pcs(these_pc_features, index_mask, channel_mask):
    """ Use the index_mask and channel_mask to return PC features for one unit

    Inputs:
    -------
    these_pc_features : numpy.ndarray (float)
        Array of pre-computed PC features (num_spikes x num_PCs x num_channels)
    index_mask : numpy.ndarray (boolean)
        Mask for spike index dimension of pc_features array
    channel_mask : numpy.ndarray (boolean)
        Mask for channel index dimension of pc_features array

    Output:
    -------
    unit_PCs : numpy.ndarray (float)
        PCs for one unit (num_spikes x num_PCs x num_channels)

    """

    unit_PCs = these_pc_features[index_mask, :, :]

    unit_PCs = unit_PCs[:, :, channel_mask]

    return unit_PCs
