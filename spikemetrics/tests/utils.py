import numpy as np


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
    