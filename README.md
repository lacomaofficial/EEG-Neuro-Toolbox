# EEG-Neuro-Toolbox

This toolbox is designed for analyzing electroencephalography (EEG) data, performing various spectral analyses, including Power Spectral Density (PSD), coherence, Phase-Locking Value (PLV), and Phase-Amplitude Coupling (PAC). The toolbox utilizes the MNE-Python library and other advanced signal processing methods to explore brain dynamics and connectivity patterns.

## Features

- **Preprocessing**: High-pass and low-pass filtering of EEG data.
- **Epoching**: Epoching data based on stimulus events.
- **Power Spectral Density (PSD)**: Calculation of power across frequency bands to study the energy distribution in different frequency ranges.
- **Coherence**: Measure the synchronization between electrode pairs in the brain.
- **Phase-Locking Value (PLV)**: Analyze the consistency of phase differences between electrode pairs over time.
- **Phase-Amplitude Coupling (PAC)**: Explore the relationship between low-frequency phase and high-frequency amplitude.

## Requirements

This toolbox requires the following Python packages:

- `numpy`
- `pandas`
- `matplotlib`
- `mne`
- `mne_connectivity`
- `tensorpac`

You can install the necessary dependencies by running:

```bash
pip install -r requirements.txt
```

## Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/EEG-Neuro-Toolbox.git
```

Navigate into the project directory:

```bash
cd EEG-Neuro-Toolbox
```

Install the package:

```bash
pip install .
```

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

## Results

The results are stored in dataframes containing key spectral and connectivity measures for each electrode and electrode pair:

- **PSD**: Power spectral density for each electrode and frequency band.
- **Coherence**: Connectivity measure between electrode pairs.
- **PLV**: Phase-locking value for electrode pairs.
- **PAC**: Phase-amplitude coupling for each electrode.

These results can be merged into a combined dataframe for further analysis or visualization.

## Example Output

- **Power Spectral Density (PSD)**:
  The average PSD across channels, with specific frequency bands identified.
  
- **Coherence**:
  The coherence values between electrode pairs, quantifying the strength of synchrony.

- **PLV**:
  The PLV values, showing phase locking between pairs of electrodes in specific frequency bands.

- **PAC**:
  The PAC values, quantifying the coupling between low-frequency phase and high-frequency amplitude.

## Contributions

Feel free to fork the repository and submit issues and pull requests if you have improvements or find bugs.

