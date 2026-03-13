# STN Validation Library

A unified Python library for validating source-localized LCMV activity against invasive iEEG recordings in the subthalamic nucleus (STN). Compare how well two methods track beta power changes during motor tasks.

---

## 📋 Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Core Concepts](#core-concepts)
- [Usage Examples](#usage-examples)
- [Configuration](#configuration)
- [Output Interpretation](#output-interpretation)


---

## Overview

This library validates whether **LCMV source-localized signals** from MEG/EEG accurately capture the same **beta ERD (event-related desynchronization)** patterns as invasive **iEEG recordings** (ground truth) in the STN during motor tasks.

### Key Features
- **Two analysis methods**: Power change (P_task - P_baseline) and ERSP (10*log10(P_task/P_baseline))
- **Flexible configuration**: Customize beta bands, subjects, preprocessing
- **Comprehensive statistics**: Spearman correlation with bootstrap confidence intervals
- **Rich visualization**: Individual and group-level plots
- **HTML reporting**: Self-contained reports for sharing

### What we're measuring
- **Beta band** (13-30 Hz): Dominant frequency in motor circuits
- **ERD** (negative values): Power decrease during movement (expected)
- **ERS** (positive values): Power increase during movement
- **STN**: Key node in basal ganglia motor circuit

---

## Installation

```bash
# Clone or download the stn_validation.py file
# Place it in your project directory

# Required dependencies
pip install numpy scipy matplotlib ipython
```

---

## Quick Start

```python
from stn_validation import *
from pathlib import Path

# Run ERSP analysis (classic method)
results, matches, correlations = run_ersp_analysis(
    ieeg_path=Path('data/consolidated_ieeg.npz'),
    lcmv_path=Path('data/consolidated_lcmv.npz'),
    output_dir=Path('results'),
    analysis_type='voxel'
)

# View correlation results
print_correlation_summary(correlations['c'], baseline='c')
```

---

## Core Concepts

### Mathematical Framework

**Power Spectral Density** (Welch's method):
- 2-second Hann windows, 50% overlap
- Power integrated over frequency bands

**Band Power in dB**:
```
P_dB = 10 * log10(P_linear)
```

**Two Analysis Approaches**:

1. **Power Change** (subtraction, needs alignment)
   ```
   ΔP = P_task_dB - P_baseline_dB
   ```

2. **ERSP** (division, scale-invariant)
   ```
   ERSP = 10 * log10(P_task / P_baseline)
   ```

**Match Criteria**:
- **Match**: Same sign (both positive or both negative)
- **Suppression**: Both negative (ERD in both methods)

---

## Usage Examples

### 1. Basic ERSP Analysis (Script 2 style)

```python
from stn_validation import *
from pathlib import Path

# Run ERSP with default settings
ersp_results, ersp_matches, ersp_corr = run_ersp_analysis(
    ieeg_path=Path('/data/consolidated_ieeg.npz'),
    lcmv_path=Path('/data/consolidated_lcmv.npz'),
    output_dir=Path('./results_ersp'),
    analysis_type='voxel',           # or 'atlas'
    baselines=['c'],                  # Eyes Closed only
    trim_seconds=5.0,                 # Remove 5s from edges
    do_statistics=True,                # Run correlations
    make_plots=True,                   # Show plots
    save_plots=False                    # Don't save to disk
)

# View correlation results
print_correlation_summary(ersp_corr['c'], baseline='c')

# Plot correlation summary
plot_correlation_summary(
    ersp_corr['c'],
    baseline='c',
    save_path=None
)
```

### 2. Basic Power Change Analysis (Script 1 style)

```python
from stn_validation import *
from pathlib import Path

# Run power change analysis
power_results, power_matches = run_power_change_analysis(
    ieeg_path=Path('/data/consolidated_ieeg.npz'),
    lcmv_path=Path('/data/consolidated_lcmv.npz'),
    output_dir=Path('./results_power'),
    analysis_type='voxel',
    baselines=['c', 'o'],              # Both Eyes Closed and Open
    make_plots=True,
    save_plots=False
)

# Add correlation analysis
power_corr = analyze_correlations(power_results, baseline='c')
print_correlation_summary(power_corr, baseline='c')
```

### 3. Custom Configuration

```python
from stn_validation import *
from pathlib import Path

# Create custom configuration
config = create_custom_config(
    # Custom frequency bands
    beta_bands={
        'low_beta': (13, 20),
        'high_beta': (20, 30),
        'gamma': (30, 45)
    },
    # Analyze specific subjects
    valid_subjects=['sub-01', 'sub-05', 'sub-10'],
    # Custom preprocessing
    default_trim_seconds=3.0,
    default_n_bootstrap=1000
)

# Run with custom config
results, matches, corr = run_ersp_analysis(
    ieeg_path=Path('/data/consolidated_ieeg.npz'),
    lcmv_path=Path('/data/consolidated_lcmv.npz'),
    output_dir=Path('./results_custom'),
    config=config,
    analysis_type='voxel'
)

print_correlation_summary(corr['c'], baseline='c')
```

### 4. Access Individual Results

```python
from stn_validation import *
from pathlib import Path

# Run analysis
ersp_results, ersp_matches, ersp_corr = run_ersp_analysis(
    ieeg_path=Path('/data/consolidated_ieeg.npz'),
    lcmv_path=Path('/data/consolidated_lcmv.npz'),
    output_dir=Path('./results'),
    analysis_type='voxel'
)

# Explore results for a specific subject
subject = 'sub-01'
region = 'L_STN'

if subject in ersp_results and region in ersp_results[subject]:
    data = ersp_results[subject][region]
    
    print(f"\n{subject} - {region} results:")
    for band, values in data.items():
        lh_ieeg = values['left_hand'].get('ieeg_c', 'N/A')
        lh_lcmv = values['left_hand'].get('lcmv_c', 'N/A')
        rh_ieeg = values['right_hand'].get('ieeg_c', 'N/A')
        rh_lcmv = values['right_hand'].get('lcmv_c', 'N/A')
        
        print(f"\n  {band}:")
        print(f"    LH: iEEG={lh_ieeg:+.2f}, LCMV={lh_lcmv:+.2f}")
        print(f"    RH: iEEG={rh_ieeg:+.2f}, LCMV={rh_lcmv:+.2f}")

# Access correlation for specific band
band = 'beta_19_22'
condition = 'left_hand_c'

if band in ersp_corr['c'] and condition in ersp_corr['c'][band]:
    corr = ersp_corr['c'][band][condition]
    print(f"\n{band} - Left Hand:")
    print(f"  Spearman ρ: {corr['spearman_rho']:.3f}")
    print(f"  p-value: {corr['p_value']:.4f}")
    print(f"  95% CI: [{corr['ci_95_lower']:.3f}, {corr['ci_95_upper']:.3f}]")
    print(f"  Directional agreement: {corr['directional_agreement_pct']:.1f}%")
```

### 5. Compare Both Methods

```python
from stn_validation import *
from pathlib import Path

# Run both analyses
ersp_results, ersp_matches, ersp_corr = run_ersp_analysis(
    ieeg_path=Path('/data/consolidated_ieeg.npz'),
    lcmv_path=Path('/data/consolidated_lcmv.npz'),
    output_dir=Path('./compare/ersp'),
    analysis_type='voxel',
    make_plots=False
)

power_results, power_matches = run_power_change_analysis(
    ieeg_path=Path('/data/consolidated_ieeg.npz'),
    lcmv_path=Path('/data/consolidated_lcmv.npz'),
    output_dir=Path('./compare/power'),
    analysis_type='voxel',
    make_plots=False
)

# Get correlations for both
power_corr = analyze_correlations(power_results, baseline='c')

# Compare
print("\n" + "="*60)
print("ERSP vs POWER CHANGE COMPARISON")
print("="*60)

for band in ersp_corr['c'].keys():
    if band in power_corr:
        print(f"\n{band}:")
        lh_ersp = ersp_corr['c'][band]['left_hand_c']['spearman_rho']
        lh_power = power_corr[band]['left_hand_c']['spearman_rho']
        print(f"  LH - ERSP: ρ={lh_ersp:.3f}, Power: ρ={lh_power:.3f}")
        
        rh_ersp = ersp_corr['c'][band]['right_hand_c']['spearman_rho']
        rh_power = power_corr[band]['right_hand_c']['spearman_rho']
        print(f"  RH - ERSP: ρ={rh_ersp:.3f}, Power: ρ={rh_power:.3f}")
```

### 6. Export to DataFrame

```python
from stn_validation import *
from pathlib import Path

# Run analysis
_, _, corr = run_ersp_analysis(
    ieeg_path=Path('/data/consolidated_ieeg.npz'),
    lcmv_path=Path('/data/consolidated_lcmv.npz'),
    output_dir=Path('./results'),
    analysis_type='voxel'
)

# Export to pandas DataFrame
try:
    import pandas as pd
    df = create_statistics_table(corr['c'])
    if df is not None:
        print(df)
        df.to_csv('correlation_results.csv', index=False)
        print("\nSaved to correlation_results.csv")
except ImportError:
    print("pandas not available - skipping DataFrame export")
```

### 7. Generate HTML Report

```python
from stn_validation import *
from pathlib import Path

# Run analysis
_, matches, _ = run_ersp_analysis(
    ieeg_path=Path('/data/consolidated_ieeg.npz'),
    lcmv_path=Path('/data/consolidated_lcmv.npz'),
    output_dir=Path('./results'),
    analysis_type='voxel'
)

# Generate report (already done by run_ersp_analysis)
# Reports are saved to output_dir/validation_report_*.html
```

### 8. Custom Plotting

```python
from stn_validation import *
from pathlib import Path

# Load data manually
ieeg_data, lcmv_data = load_consolidated_data(
    Path('/data/consolidated_ieeg.npz'),
    Path('/data/consolidated_lcmv.npz')
)

# Collect with custom parameters
collected = collect_all_data(
    ieeg_data, lcmv_data,
    analysis_type='voxel',
    subjects=['sub-01', 'sub-05'],
    trim_seconds=3.0,
    apply_filter=True
)

# Compute ERSP
ersp_results = compute_ersp(collected)

# Plot individual subject
for subject in collected.keys():
    for region in collected[subject].keys():
        if region in ersp_results[subject]:
            plot_power_spectra(
                collected[subject][region],
                subject=subject,
                region=region,
                runs=[('c', 'Closed'), ('l', 'Left'), ('r', 'Right')]
            )
            
            plot_subject_summary(
                ersp_results[subject],
                subject=subject,
                region=region,
                baseline='c',
                analysis_name='ERSP'
            )
            break  # Just show one
```

---

## Configuration

### Available Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `beta_bands` | Frequency bands to analyze | 13-16, 16-19, 19-22, 22-25, 25-28, 28-31 Hz |
| `valid_subjects` | Subjects to include | ['sub-01', 'sub-05', ...] |
| `trim_seconds` | Seconds to trim from edges | 5.0 |
| `filter_cutoff` | High-pass filter cutoff | 0.5 Hz |
| `window_seconds` | Welch window length | 2.0 s |
| `n_bootstrap` | Bootstrap samples | 2000 |
| `ci_level` | Confidence interval | 95 |

### Create Custom Config

```python
config = create_custom_config(
    beta_bands={'alpha': (8, 12), 'beta': (13, 30)},
    valid_subjects=['sub-01', 'sub-02'],
    default_trim_seconds=2.0,
    default_n_bootstrap=1000
)
```

---

## Output Interpretation

### Power Change / ERSP Values
- **Negative** = ERD (power decreased) - expected during movement
- **Positive** = ERS (power increased)
- **Near zero** = No change

### Match Results
- **✓✓** = Both methods show ERD (strong agreement)
- **✓** = Same sign but not both negative
- **✗** = Opposite signs
- **?** = Missing data

### Correlation Results

```
SPEARMAN CORRELATION SUMMARY (baseline=c)
----------------------------------------------------------------------
Band        Cond   N    Corr     P-val    CI_low   CI_high  Agree%  
----------------------------------------------------------------------
13-16       LH     12   0.72    0.003**  0.45     0.89     83.3    
13-16       RH     12   0.68    0.011*   0.38     0.85     75.0    
16-19       LH     12   0.81    0.001*** 0.62     0.93     91.7    
----------------------------------------------------------------------
Corr = Spearman's ρ | P-val = p-value | CI = 95% Bootstrap CI
Significance: *** p<0.001, ** p<0.01, * p<0.05
```

- **Corr > 0.7**: Strong relationship
- **Corr 0.3-0.7**: Moderate relationship
- **Corr < 0.3**: Weak relationship
- **p < 0.05**: Statistically significant

### Directional Agreement
- **>80%**: Excellent agreement on direction
- **60-80%**: Good agreement
- **<60%**: Poor agreement

### Subject Classification
- **Excellent**: ≥70% matches across all bands/conditions
- **Good**: 50-70% matches
- **Poor**: <50% matches

---

## API Reference

### Main Pipelines

| Function | Returns | Description |
|----------|---------|-------------|
| `run_ersp_analysis()` | (results, matches, correlations) | Full ERSP pipeline |
| `run_power_change_analysis()` | (results, matches) | Power change pipeline |

### Analysis Functions

| Function | Description |
|----------|-------------|
| `compute_ersp()` | ERSP = 10*log10(P_task/P_baseline) |
| `compute_power_changes()` | ΔP = P_task_dB - P_baseline_dB |
| `get_ersp_match_results()` | Convert ERSP to matches |
| `get_power_match_results()` | Convert power changes to matches |

### Statistics

| Function | Description |
|----------|-------------|
| `analyze_correlations()` | Full correlation analysis |
| `print_correlation_summary()` | Formatted correlation table |
| `calculate_spearman_rho()` | Spearman correlation |
| `bootstrap_confidence_interval()` | Bootstrap CI |

### Plotting

| Function | Description |
|----------|-------------|
| `plot_power_spectra()` | Individual power spectra |
| `plot_subject_summary()` | Single subject results |
| `plot_group_summary()` | Group-level results |
| `plot_correlation_summary()` | Correlation plots |

### Configuration

| Function | Description |
|----------|-------------|
| `create_custom_config()` | Create custom STNConfig |

---



## Support

For questions or issues, please open an issue on GitHub or contact the maintainer.
