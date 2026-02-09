# **MinimalEEG:** A Minimalist Pipeline for High-Density EEG Preprocessing

A robust, two-stage preprocessing pipeline for dense-array EEG (e.g., 280-channel systems) that:
- Preserves full sensor topology (no permanent channel dropping)
- Handles real-world artifacts in short-duration trials (40–60 s)
- Outputs BIDS-compatible derivatives ready for ICA, source imaging, or machine learning
[Library Documentation](https://nbara.github.io/python-meegkit/index.html)

## Pipeline Overview

### Stage 1: Sensor-Level Cleaning (`1_meegkit.ipynb`)
- **Bad channel detection**: Identifies flat and persistently noisy channels using robust statistics (MAD + amplitude)
- **MEEGKit cleaning**: Applies ASR → STAR → SNS on good channels only
- **Full-sensor reconstruction**: Interpolates bad channels **after** cleaning to avoid artifact spreading
- **Output**: `*_meegkit_cleaned_eeg.fif` (280 channels, cleaned, interpolated)

### Stage 2: Component-Level Refinement (`2_icalabel.ipynb`)
- **ICA decomposition**: Uses Picard algorithm on cleaned data
- **Automatic labeling**: ICLabel classifies components (eye blink, muscle, heart, etc.)
- **Artifact rejection**: Removes components above confidence thresholds
- **Final interpolation**: Ensures full sensor set for downstream analysis
- **Output**: `*_meegkit_icalabel_cleaned_eeg.fif`

### Stage 3: Quality Control (`3_check_quality.ipynb`)
- Interactive comparison of raw vs. cleaned vs. ICA-cleaned data
- Visual inspection of time series and topographies

## Key Features
- ✅ **Minimal**: Only essential steps — no over-engineering
- ✅ **Robust**: Works on highly contaminated, short trials
- ✅ **Full-topography preserved**: All 280 channels retained via safe interpolation
- ✅ **Reproducible**: Script-based, no manual intervention
- ✅ **MNE-native**: Fully compatible with MNE-Python ecosystem

## Dependencies
- Python ≥3.9
- `mne >=1.5`
- `meegkit`
- `mne-icalabel`
- `scipy`, `numpy`, `matplotlib`

## Usage
1. Configure paths in `1_meegkit.ipynb`
2. Run notebooks in order: `1` → `2` → `3`
3. Inspect outputs in `derivatives/eeg/meegkit/`

> **Note**: Designed for high-density EEG (e.g., EGI 256+). Adjust thresholds for other systems.

## Links
-  Meegkit: [Library Documentation](https://nbara.github.io/python-meegkit/index.html)

