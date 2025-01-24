# EEG-Neuro-Toolbox

This toolbox is designed to analyze electroencephalography (EEG) data. The toolbox utilizes the MNE-Python library and other advanced signal-processing methods to explore brain dynamics and connectivity patterns.

This repository contains multiple tests and frameworks for EEG data analysis, feature extraction, and machine learning classification. The tests cover both **intrasubject** (single subject) and **intersubject** (multiple subjects) analysis, focusing on tasks such as EEG segmentation, feature extraction, and classification using machine learning models like **XGBoost** and **Common Spatial Pattern (CSP)**.



## Table of Contents

1. [EEG Epoch Processing Framework](#eeg-epoch-processing-framework)
2. [EEG Feature Engineering Framework](#eeg-feature-engineering-framework)
3. [Test1: Data Exploration and Visualization](#test1-data-exploration-and-visualization)
4. [Test2: Auditory vs. Visual Stimuli Classification](#test2-auditory-vs-visual-stimuli-classification)
5. [Test3: Brain-Computer Interface (BCI) with CSP and XGBoost](#test3-brain-computer-interface-bci-with-csp-and-xgboost)
6. [Test4: Wavelet Denoising for Event-Related Epochs](#test4-wavelet-denoising-for-event-related-epochs)
7. [Test5: Intracranial EEG (iEEG) Seizure Classification](#test5-intracranial-eeg-ieeg-seizure-classification)
8. [Test6: Spectral Analysis and Classification with XGBoost](#test6-spectral-analysis-and-classification-with-xgboost)

---

## EEG Epoch Processing Framework

This framework is designed for **intersubject** analysis of EEG data, processing raw EEG recordings from multiple subjects, segmenting them into epochs based on experimental events, and saving them for further analysis.

### Key Features:
- **Data Loading**: Supports formats such as `.edf` for raw EEG data.
- **Preprocessing**: Implements band-pass and notch filtering, with optional wavelet denoising.
- **Event Extraction**: Identifies experiment-specific triggers (e.g., stimulus onsets).
- **Epoch Creation**: Segments continuous EEG data into epochs based on event timings.
- **Data Saving**: Saves processed epochs as `.fif` files for future use.

### Usage:
- Configure settings (e.g., filtering ranges) in the `CONFIG` dictionary.
- Run the `main(CONFIG)` function to process data for specified subjects and sessions.
- Use the `load_epochs()` function to retrieve the processed epochs for analysis.

---

## EEG Feature Engineering Framework

This framework is used for **intersubject** feature extraction from preprocessed EEG epochs, preparing the data for further analysis or machine learning tasks.

### Key Features:
- **Data Loading**: Loads preprocessed EEG epochs from `.fif` files.
- **Power Spectral Density (PSD)**: Computes frequency-domain representations of EEG data.
- **Frequency Band Analysis**: Divides the PSD into standard frequency bands (Delta, Theta, Alpha, Beta, Gamma).
- **Event-Specific Feature Extraction**: Derives features based on specific experimental conditions (e.g., auditory vs. visual stimuli).
- **Data Reshaping**: Organizes features into a table format suitable for machine learning.
- **Data Saving**: Stores the extracted features for future use.

---

## Test1: Data Exploration and Visualization

This test is focused on **intrasubject** data exploration and visualization. The primary goal is to segment raw EEG data into epochs based on event markers and perform initial exploration through visualizations.

### Key Focus:
- **Data Segmentation and Cleaning**: Segment raw EEG data into epochs, clean the data by removing artifacts, and explore the data visually.
- **Visualization**: Plot the Power Spectral Density (PSD) of EEG data to understand signal characteristics and identify any potential artifacts.
- **Data Preprocessing**: Filter and clean the EEG signal for further analysis.

---

## Test2: Auditory vs. Visual Stimuli Classification

In this **intrasubject** test, the focus is on classifying **auditory vs. visual stimuli** based on preprocessed EEG epochs.

### Key Focus:
- **Machine Learning**: Train a classifier to distinguish between auditory and visual stimuli using **epoch-based features**.
- **Event Conditions**: Utilize specific experimental event markers for auditory vs. visual stimuli.
- **Classification**: Implement machine learning models (e.g., **XGBoost**) to classify the stimuli based on the features extracted from the EEG data.

---

## Test3: Brain-Computer Interface (BCI) with CSP and XGBoost

This test focuses on **intrasubject** Brain-Computer Interface (BCI) tasks. The data is processed using **Common Spatial Pattern (CSP)** for feature extraction, followed by **XGBoost** for classification.

### Key Focus:
- **Feature Extraction with CSP**: Apply CSP to extract spatial features from EEG data.
- **Classification with XGBoost**: Use the extracted features to classify tasks such as **left vs. right motor imagery** or other BCI tasks.
- **Machine Learning**: Use **XGBoost** to classify different brain states based on the features derived through CSP.

---

## Test4: Wavelet Denoising for Event-Related Epochs

This **intrasubject** test applies **wavelet denoising** to EEG data to clean up noise before performing further analysis on event-related epochs.

### Key Focus:
- **Wavelet Denoising**: Reduce noise in the EEG data using wavelet-based techniques.
- **Event Conditions**: Focus on conditions like `auditory/left`, `auditory/right`, etc., for event-related potential (ERP) analysis.
- **Data Cleaning**: Use wavelet denoising as a preprocessing step to improve signal quality before extracting features for classification.

---

## Test5: Intracranial EEG (iEEG) Seizure Classification

This test focuses on **intrasubject** **intracranial EEG (iEEG)** data and its classification for **epilepsy-related** events such as seizures.

### Key Focus:
- **Intracranial EEG (iEEG)**: Analyze EEG data from a single subject with epilepsy.
- **Event Conditions**: Use specific event conditions related to seizures (e.g., `Trigger-401`, `Trigger-402`, `Trigger-501`).
- **Seizure Classification**: Apply machine learning models to classify **seizure** vs **non-seizure** events.
- **Context**: This test is highly specific to **epilepsy** and seizure detection tasks.

---

## Test6: Spectral Analysis and Classification with XGBoost

This **intrasubject** test applies spectral analysis to **intracranial EEG (iEEG)** data, specifically using **stereo-electroencephalography (sEEG)**, followed by classification using **XGBoost** based on frequency-domain features.

### Key Focus:
- **Spectral Analysis**: Perform spectral analysis on **iEEG** data to extract features in different frequency bands (Delta, Theta, Alpha, Beta, Gamma).
- **Classification with XGBoost**: Classify EEG events based on spectral features extracted from the frequency-domain analysis.
- **Frequency Bands**: Focus on **frequency bands** like Delta, Theta, Alpha, etc., to capture the dynamics of neurological activity, such as seizures.
- **Stereo-EEG Data**: This test specifically uses **stereo-EEG** data, which involves intracranial electrodes for high-resolution monitoring of brain activity, especially useful in epilepsy detection.


---



## Libraries Used

- **MNE**: A comprehensive library for processing and analyzing EEG and MEG data. It is used for data loading, filtering, event extraction, epoching, and spectral analysis.
- **Tensorpac**: A library for analyzing phase-amplitude coupling (PAC) in EEG data. It is used to compute PAC between different frequency bands.
- **Matplotlib**: Used for plotting the results of the analysis (e.g., PSD, evoked responses, PAC).
- **Pandas**: Used for organizing and handling data, particularly when computing and visualizing frequency band power.


