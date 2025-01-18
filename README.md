# EEG-Neuro-Toolbox

This toolbox is designed for analyzing electroencephalography (EEG) data. The toolbox utilizes the MNE-Python library and other advanced signal processing methods to explore brain dynamics and connectivity patterns.


## Requirements

This toolbox requires the following Python packages:

- `numpy`
- `pandas`
- `matplotlib`
- `mne`
- `mne_connectivity`
- `tensorpac`



## Usage

### 1. **Preprocessing and Filtering**
   The EEG data is first filtered to remove noise and unwanted frequencies. High-pass and low-pass filters are applied to retain the relevant frequency bands.

### 2. **Epoching the Data**
   Events are extracted from the stimulus channel, and the data is segmented into epochs based on these events. Epochs are typically time-locked to specific stimuli, such as auditory stimuli.

### 3. **Power Spectral Density (PSD)**
   The PSD is calculated for each epoch and across all EEG channels to observe the power distribution across different frequencies (e.g., Delta, Theta, Alpha, Beta, Gamma).

### 4. **Coherence**
   Coherence is calculated for all electrode pairs, quantifying the degree of synchrony between signals from different brain regions in specific frequency bands.

### 5. **Phase-Locking Value (PLV)**
   PLV is computed for pairs of electrodes to evaluate how consistently their phases are locked together, indicating functional connectivity between brain regions.

### 6. **Phase-Amplitude Coupling (PAC)**
   PAC measures the coupling between the phase of lower frequencies (e.g., Theta) and the amplitude of higher frequencies (e.g., Gamma), which is thought to play a role in cognitive processes like attention.

## Data Format

The toolbox is designed to work with EEG data in the [FIF](https://mne.tools/stable/overview.html) file format, which can be loaded directly using the MNE-Python library.


## Contributions

Feel free to fork the repository and submit issues and pull requests if you have improvements or find bugs.

