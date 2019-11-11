import numpy as np

from scipy.stats import norm, multivariate_normal

def create_ground_truth_pc_distributions(center_locations, total_points):
    """ Simulate PCs as multivariate Gaussians, for testing PC-based quality metrics

    Values are created for only one channel and vary along one dimension

    Input:
    ------
    separation : distance between distribution means
    total_points : array indicating number of points in each distribution

    Output:
    -------
    all_pcs : N x 3 matrix of simulated PCs
        N = np.sum(total_points)
    all_labels : array of cluster IDs

    """

    np.random.seed(0)

    distributions = [multivariate_normal.rvs(mean=[center, 0.0, 0.0],
                                             cov=[1.0, 1.0, 1.0],
                                             size=size) 
                    for center, size in zip(center_locations, total_points)]

    all_labels = np.concatenate([np.ones((distributions[i].shape[0],),dtype='int')*i  for i in range(len(distributions))])

    all_pcs = np.concatenate(distributions, axis=0)

    return all_pcs, all_labels
    


def simulated_pcs_for_one_spike(total_channels, peak_channel):
    """ Simulate the top principal components across channels for one spike

    Used for testing drift metrics

    Input:
    ------
    total_channels : number of channels
    peak_channel : location of the peak channel

    Output:
    -------
    pc_features : 1 x 3 x total_channels matrix of PC features
    pc_feature_ind : channel indices for 3rd dimension of pc_features

    """

    pc_feature_ind = np.arange(total_channels)
    feat = np.sqrt(norm.pdf(pc_feature_ind, loc=peak_channel, scale=1))
    pc_features = np.zeros((1,3,32), dtype='float64')
    pc_features[:,0,:] = feat.T

    return pc_features, np.expand_dims(pc_feature_ind,axis=1).T


def simulated_pcs_for_one_unit(num_spikes, total_channels, start_channel, end_channel):
    """ Simulate the top principal components across channels for one unit

    Used for testing drift metrics

    Input:
    ------
    num_spikes : total number of spikes
    total_channels : number of channels
    start_channel : initial peak channel
    end_channel : final peak channel

    Output:
    -------
    spike_train : array of monotonically increasing spike times

    """

    pc_features = []

    peak_channels = np.linspace(start_channel, end_channel, num_spikes)

    for i, peak_chan in enumerate(peak_channels):
        pc_feat, pc_feature_ind = simulated_pcs_for_one_spike(total_channels, peak_chan)
        pc_features.append(pc_feat)

    return np.concatenate(pc_features, axis=0)


def simulated_spike_train(duration, baseline_rate, num_violations, violation_delta=1e-5):
    """ Create a spike train for testing ISI violations

    Has uniform inter-spike intervals, except where violations occur

    Input:
    ------
    duration : length of simulated recording (in seconds)
    baseline_rate : firing rate for 'true' spikes
    num_violations : number of contaminating spikes
    violation_delta : temporal offset of contaminating spikes (in seconds)

    Output:
    -------
    spike_train : array of monotonically increasing spike times

    """

    isis = np.ones((int(duration*baseline_rate),)) / baseline_rate
    spike_train = np.cumsum(isis)
    viol_times = spike_train[:int(num_violations)] + violation_delta
    spike_train = np.sort(np.concatenate((spike_train, viol_times)))
    
    return spike_train
    

def simulated_spike_amplitudes(num_spikes, mean_amplitude, amplitude_std):
    """ Create spike amplitudes for testing amplitude cutoff

    Has a Gaussian distribution, but may be truncated at the low end

    Input:
    ------
    num_spikes : total_number of spikes
    mean_amplitude : center of the amplitude histogram
    amplitude_std : standard deviation of the amplitude histogram

    Output:
    -------
    spike_amplitudes : array of amplitudes (arbitrary units)

    """

    np.random.seed(1)
    r = norm.rvs(size=num_spikes, loc=mean_amplitude, scale=amplitude_std)
    spike_amplitudes = np.delete(r, np.where(r < 0)[0])
    
    return spike_amplitudes
    