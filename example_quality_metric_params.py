#Quality metric params from https://github.com/AllenInstitute/ecephys_spike_sorting/blob/master/ecephys_spike_sorting/scripts/create_input_json.py

quality_metrics_params = {
                          "isi_threshold" : 0.0015,
                          "min_isi" : 0.000166,
                          "num_channels_to_compare" : 13,
                          "max_spikes_for_unit" : 500,
                          "max_spikes_for_nn" : 10000,
                          "n_neighbors" : 4,
                          'n_silhouette' : 10000,
                          "quality_metrics_output_file" : "metrics.csv",
                          "drift_metrics_interval_s" : 51,
                          "drift_metrics_min_spikes_per_interval" : 10
                         }
