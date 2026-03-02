### CROCOpy - CRitical Oscillations and COnnectivity in Python

*crocopy* is a small Python toolbox designed to process electrophysiological recordings and compute criticality and functional connectivity observables in application to neuroscience. 

The observables include:

|             Category            |                Construct                |                       Observables                       |
|:-------------------------------:|:---------------------------------------:|:-------------------------------------------------------:|
|           Oscillations          |          Phase autocorrelation          |                        pAC value                        |
|                                 |                                         |                      pAC lifetimes                      |
|           Criticality           |                Avalanches               |                      Avalanche size                     |
|                                 |                                         |                    Avalanche duration                   |
|                                 |                  LRTCs                  |                   DFA scaling exponent                  |
|                                 | Functional  excitation-inhibition ratio |                           fE/I                          |
|                                 |               Bistability               |                    Bistability index                    |
| Oscillation-based  connectivity |             Phase Synchrony             |                   Phase locking value (PLV)             |
|                                 |                                         |         Imag. component of  PLV                         |
|                                 |                                         |                 Weighted phase lag index (wPLI)         |
|                                 |            Amplitude Coupling           |                  Amplitude correlation                  |
|                                 |                                         |         Amplitude correlation  (orthogonalized)         |
|                                 |        Cross-frequency  Coupling        | Cross-frequency phase coupling                          |
|                                 |                                         |                 Phase-amplitude coupling                |
|                                 |                                         |             Filtered amplitude correlations             |

### Installation

One can install the toolbox directly from github using pip

    pip install git+https://github.com/palvalab/crocopy.git

Or clone the repository at first

    git clone https://github.com/palvalab/crocopy.git
    cd crocopy
    pip install .

### Quick start

Please refer to the [Tutorials](https://github.com/palvalab/crocopy/tree/main/tutorials) for comprehensive overview of features and how to use them.

    data_broadband = ... # load your data here e.g. with MNE
    data_sfreq = data_obj_ec.info['sfreq']

    frequencies_of_interest = np.geomspace(1, 50, 30)

    n_chans, n_ts = data_broadband.shape

    dfa_as_frequency = np.zeros((len(frequencies_of_interest), n_chans))
    bis_as_frequency = np.zeros((len(frequencies_of_interest), n_chans))
    wpli_as_frequency = np.zeros((len(frequencies_of_interest), n_chans, n_chans))
    occ_as_frequency = np.zeros((len(frequencies_of_interest), n_chans, n_chans))

    for frequency_idx, frequency in enumerate(frequencies_of_interest):
        # for DFA we need to specify window sizes, typycally we recommend the smallest window to start at ~10 cycles of narrow-band central frequency
        # and larger windows to be roughly 25% of signal length to have enough data for robust mean estimation.
        samples_per_cycle = data_sfreq/frequency
        dfa_window_size_for_frequency = np.geomspace(10*samples_per_cycle, n_ts//4)

        # we pass n_jobs='cuda' to enforce filtering on GPU to make it faster and avoid copying data from CPU to GPU back and forth.
        data_filt = crocopy.preprocessing.signal.filter_data(data_broadband, sfreq=data_sfreq, frequency=frequency, omega=5.0, n_jobs='cuda')
        data_envelope = np.abs(data_filt)

        # DFA is computed using envelope of a narrow-band signal while oCC and wPLI require complex data.
        # note:  observables functions utilize CPU/GPU based on type of the data. e.g. if your signal is stored on CPU, the observables will be computed using it.
        # since we filtered the data using GPU and our data is stored on it, the functions will automatically use it too.
        dfa_res = crocopy.observables.criticality.lrtc.compute_dfa(data_envelope, dfa_window_size_for_frequency)
        bis_res = crocopy.observables.criticality.bistability.compute_BiS(data_envelope, method='em')
        wpli_res = crocopy.observables.connectivity.synchrony.compute_wpli(data_filt, debias=False)
        occ_res = crocopy.observables.connectivity.amplitude_correlations.compute_occ(data_filt)
        
        # since we use GPU to compute our observables, wPLI and oCC functions return GPU arrays and we need to get them back to CPU.
        # note: DFA uses statsmodels to fit a robust linear regression, thus it always returns CPU data.
        dfa_as_frequency[frequency_idx] = dfa_res.dfa_values
        wpli_as_frequency[frequency_idx] = np.abs(wpli_res.get())
        occ_as_frequency[frequency_idx] = np.abs(occ_res.get())
        bis_as_frequency[frequency_idx] = bis_res.get()