# SpikeMetrics
Computes quality metrics for sorted units. This module can calculate metrics separately for individual epochs. If no epochs are specified, metrics are computed for the entire recording.

The base code (and portions of the README images/description) was ported from: https://github.com/AllenInstitute/ecephys_spike_sorting/tree/master/ecephys_spike_sorting/modules/quality_metrics

Copyright 2019. Allen Institute.  All rights reserved.

## Included Metrics

| Metric             | Icon                     | Description                                        |    Reference     |
| ------------------ |:------------------------:| -------------------------------------------------- | -----------------|
| Spike Count       |                          | Spike count in an epoch                        |                  |
| Firing rate        |                          | Mean spike rate in an epoch                        |                  |
| Presence ratio     |                          | Fraction of epoch in which spikes are present      |                  |
| ISI violations     |![](images/isi_viol.png)  | Rate of refractory-period violations               |                  |
| Isolation distance |![](images/isol_dist.png) | The Mahalanobis distance from a specified unit within as many spikes belong to the specified unit as to other units   | Harris et al. Neuron 32.1 (2001): 141-149. |
| L-ratio            |                          | The Mahalanobis distance and chi-squared inverse cdf (given the assumption that the spikes in the cluster distribute normally in each dimension) are used to find the probability of cluster membership for each spike. |        Schmitzer-Torbert and Redish. _J Neurophy_  91.5 (2004): 2259-2272.         |
| _d'_               |![](images/d_prime.png)   | The classification accuracy between units based on linear discriminant analysis (LDA).               | Hill et al. (2011) _J Neurosci_ **31**, 8699-9705 |
| Nearest-neighbors  |![](images/nn_overlap.png)| Non-parametric estimate of unit contamination using nearest-neighbor classification.      | Chung et al. (2017) _Neuron_ **95**, 1381-1394 |
| Silhouette score  |                           | A standard metric for quantifying cluster overlap      |         |
| Maximum drift     |                           | Maximum change in spike depth throughout recording    |         |
| Cumulative drift  |                           | Cumulative change in spike depth throughout recording |         |

### A Note on Calculations

For metrics based on waveform principal components (isolation distance, L-ratio, _d'_, and nearest neighbors hit rate and false alarm rate), it is typical to compute the metrics for all pairs of units and report the "worst-case" value. We have found that this tends to under- or over-estimate the degree of contamination when there are large firing rate differences between pairs of units that are being compared. Instead, we compute metrics by sub-selecting spikes from _all_ other units on the same set of channels, which seems to give a more accurate picture of isolation quality. We would appreciate feedback on whether this approach makes sense.
