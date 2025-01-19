# EEG-Neuro-Toolbox

This toolbox is designed for analyzing electroencephalography (EEG) data. The toolbox utilizes the MNE-Python library and other advanced signal processing methods to explore brain dynamics and connectivity patterns.

## Overview

The project includes the following stages:

1. **Data Loading and Preprocessing**  
   - Load EEG data from the MNE sample dataset.
   - Apply high-pass and low-pass filtering to remove noise and focus on relevant frequency ranges.
   
2. **Event Extraction and Epoching**  
   - Extract event information from stimulus markers.
   - Create epochs based on specific event types (auditory and visual stimuli).
   
3. **Epoch Visualization and Inspection**  
   - Visualize the PSD of the raw data.
   - Inspect individual epochs to identify any bad epochs and remove them as necessary.
   
4. **Evoked Responses**  
   - Compute average evoked responses for different stimuli (auditory and visual).
   - Visualize these responses to observe spatial and temporal dynamics across electrodes.
   
5. **Power Spectral Density (PSD) Analysis**  
   - Calculate and visualize the PSD for each event type.
   - Compare the PSD across different stimulus conditions to explore spectral power differences.
   
6. **Frequency Band Analysis**  
   - Define standard frequency bands (Delta, Theta, Alpha, Beta, Gamma).
   - Compute the average power within each frequency band and visualize the results.
   
7. **Phase-Amplitude Coupling (PAC) Analysis**  
   - Compute PAC between the phase of low-frequency oscillations (e.g., Theta band) and the amplitude of high-frequency oscillations (e.g., Gamma band).
   - Visualize PAC values for different stimuli to observe coupling effects.

## Libraries Used

- **MNE**: A comprehensive library for processing and analyzing EEG and MEG data. It is used for data loading, filtering, event extraction, epoching, and spectral analysis.
- **Tensorpac**: A library for analyzing phase-amplitude coupling (PAC) in EEG data. It is used to compute PAC between different frequency bands.
- **Matplotlib**: Used for plotting the results of the analysis (e.g., PSD, evoked responses, PAC).
- **Pandas**: Used for organizing and handling data, particularly when computing and visualizing frequency band power.

## Key Functions

- **Data Preprocessing**: High-pass filtering, epoch extraction, and event handling.
- **Evoked Response**: Computing and plotting the average evoked response for each stimulus condition.
- **Power Spectral Density (PSD)**: Calculation of PSD using Welch's method and comparison across event types.
- **Frequency Band Analysis**: Breakdown of PSD into predefined frequency bands and visualization of power in each band.
- **Phase-Amplitude Coupling (PAC)**: Estimation of PAC between low and high-frequency oscillations and visualization of coupling strength across events.

## Results

The analysis provides insights into:
- The spectral power distribution across different stimulus conditions (auditory and visual).
- The presence of phase-amplitude coupling between specific low and high-frequency bands.
- The differences in evoked responses between auditory and visual stimuli, including topographic maps.

The final output consists of visual plots and tables summarizing key findings such as the power within frequency bands and PAC values across electrodes and events.

## Data

The dataset used in this analysis is a sample EEG dataset provided by MNE. It consists of EEG recordings from an auditory and visual stimulation experiment. The data includes measurements from multiple EEG channels (magnetometers, gradiometers, and EEG) as well as event markers indicating the onset of auditory and visual stimuli.

## Usage

To run the analysis:
1. Install the required libraries: MNE, Tensorpac, Matplotlib, and Pandas.
2. Load the data using MNE's sample dataset functionality.
3. Follow the steps outlined in the notebook or script to preprocess the data, extract events, and perform the analysis.

## Conclusion

This project demonstrates a complete workflow for analyzing EEG data, from preprocessing to spectral analysis and PAC computation. The results provide a deeper understanding of brain oscillations and their coupling mechanisms in response to auditory and visual stimuli.

