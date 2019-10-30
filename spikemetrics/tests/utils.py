import numpy as np

from scipy.stats import norm

def create_ground_truth_pc_distributions():
    # HINT: start from Guassians in PC space and stereotyped waveforms and build dataset.
    pass


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
    