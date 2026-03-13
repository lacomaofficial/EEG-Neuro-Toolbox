"""
STN Validation Unified Library
Combines power change and ERSP analysis approaches for comparing iEEG and LCMV source-localized STN activity

Features:
- Power change analysis: P_task - P_baseline (dB)
- ERSP analysis: 10 * log10(P_task / P_baseline)
- Configurable beta bands, subjects, and preprocessing
- Statistical analysis with Spearman correlation and bootstrap CI
- Comprehensive plotting and reporting
"""

import numpy as np
import scipy.signal as signal
from scipy.integrate import trapezoid
from scipy.stats import rankdata, t
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.gridspec as gridspec
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Literal, Union, Any, Callable
import warnings
import logging
from IPython.display import display, Markdown, HTML
import datetime
from dataclasses import dataclass, field
from enum import Enum

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION CLASSES
# =============================================================================

class AnalysisMethod(Enum):
    """Available analysis methods"""
    POWER_CHANGE = "power_change"  # P_task - P_baseline (dB)
    ERSP = "ersp"  # 10 * log10(P_task / P_baseline)


@dataclass
class STNConfig:
    """Main configuration for STN validation analysis"""
    
    # Beta bands (can be modified after creation)
    beta_bands: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        'beta_13_16': (13, 16),
        'beta_16_19': (16, 19),
        'beta_19_22': (19, 22),
        'beta_22_25': (22, 25),
        'beta_25_28': (25, 28),
        'beta_28_31': (28, 31),
    })
    
    # Colors for plots
    band_colors: Dict[str, str] = field(default_factory=lambda: {
        'beta_13_16': '#0000FF',
        'beta_16_19': '#4169E1',
        'beta_19_22': '#9370DB',
        'beta_22_25': '#BA55D3',
        'beta_25_28': '#FF69B4',
        'beta_28_31': '#FF1493',
    })
    
    # Baseline names
    baseline_names: Dict[str, str] = field(default_factory=lambda: {
        'c': 'Eyes Closed',
        'o': 'Eyes Open'
    })
    
    # Valid subjects
    valid_subjects: List[str] = field(default_factory=lambda: [
        'sub-01', 'sub-05', 'sub-07', 'sub-10', 'sub-12', 'sub-14'
    ])
    
    # All available subjects
    all_subjects: List[str] = field(default_factory=lambda: [
        'sub-01', 'sub-05', 'sub-06', 'sub-07', 'sub-10', 'sub-11', 'sub-12', 'sub-14'
    ])
    
    # Reference frequency range for 0 dB alignment
    ref_freq_range: Tuple[float, float] = (1, 3)
    
    # ROI patterns for different analysis types
    roi_patterns: Dict[str, Dict[str, List[str]]] = field(default_factory=lambda: {
        'atlas': {
            'L_STN': ['STN_atlas'],
            'R_STN': ['STN_atlas']
        },
        'voxel': {
            'L_STN': ['L_STN_voxel'],
            'R_STN': ['R_STN_voxel']
        }
    })
    
    # Default sampling frequencies
    default_sfreq: Dict[str, float] = field(default_factory=lambda: {
        'ieeg': 250.0,
        'lcmv': 500.0
    })
    
    # Preprocessing defaults
    default_trim_seconds: float = 5.0
    default_filter_cutoff: float = 0.5
    default_spectrum_method: str = 'welch'
    default_window_seconds: float = 2.0
    
    # Analysis defaults
    default_analysis_method: AnalysisMethod = AnalysisMethod.ERSP
    default_baseline: str = 'c'
    
    # Statistics defaults
    default_n_bootstrap: int = 2000
    default_ci_level: float = 95
    
    def get_band_list(self) -> List[str]:
        """Get list of band names"""
        return list(self.beta_bands.keys())
    
    def get_band_range(self, band_name: str) -> Tuple[float, float]:
        """Get frequency range for a band"""
        return self.beta_bands.get(band_name, (13, 30))
    
    def get_band_color(self, band_name: str) -> str:
        """Get color for a band"""
        return self.band_colors.get(band_name, '#808080')
    
    def update_beta_bands(self, new_bands: Dict[str, Tuple[float, float]]):
        """Update beta bands configuration"""
        self.beta_bands.update(new_bands)
        # Auto-generate colors if needed
        for band in new_bands:
            if band not in self.band_colors:
                # Generate a color from the spectrum
                import matplotlib.cm as cm
                idx = len(self.band_colors)
                self.band_colors[band] = cm.tab10(idx % 10)
    
    def set_valid_subjects(self, subjects: List[str]):
        """Set valid subjects for analysis"""
        self.valid_subjects = subjects


# Global default configuration
DEFAULT_CONFIG = STNConfig()


# =============================================================================
# TYPE ALIASES
# =============================================================================

AnalysisType = Literal['atlas', 'voxel']
BaselineType = Literal['c', 'o']
ModalityType = Literal['ieeg', 'lcmv']
RunType = Literal['c', 'o', 'l', 'r']
SpectrumMethod = Literal['welch']


# =============================================================================
# SIGNAL PROCESSING FUNCTIONS
# =============================================================================

def highpass_filter(ts: np.ndarray, sfreq: float, cutoff: float = 0.5) -> np.ndarray:
    """
    Apply 4th-order zero-phase Butterworth high-pass filter
    
    Args:
        ts: Time series data
        sfreq: Sampling frequency in Hz
        cutoff: Cutoff frequency in Hz
    
    Returns:
        Filtered time series
    """
    nyq = sfreq * 0.5
    b, a = signal.butter(4, cutoff / nyq, btype='high')
    return signal.filtfilt(b, a, ts)


def trim_trial_edges(time_series: np.ndarray, sfreq: float, trim_seconds: float) -> np.ndarray:
    """
    Trim specified number of seconds from start and end of each trial
    
    Args:
        time_series: Time series data
        sfreq: Sampling frequency in Hz
        trim_seconds: Seconds to trim from start and end
    
    Returns:
        Trimmed time series
    """
    trim_samples = int(trim_seconds * sfreq)
    
    if len(time_series) <= 2 * trim_samples:
        logger.warning(f"Trial too short ({len(time_series)/sfreq:.1f}s) for {trim_seconds}s trim")
        return time_series
    
    trimmed = time_series[trim_samples:-trim_samples]
    return trimmed


def get_spectrum(
    time_series: np.ndarray,
    sfreq: float,
    method: str = 'welch',
    window_seconds: float = 2.0,
    apply_filter: bool = False,
    filter_cutoff: float = 0.5
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Compute power spectrum
    
    Args:
        time_series: Time series data
        sfreq: Sampling frequency in Hz
        method: Spectrum estimation method ('welch')
        window_seconds: Window length in seconds
        apply_filter: Whether to apply high-pass filter first
        filter_cutoff: High-pass filter cutoff if apply_filter is True
    
    Returns:
        Tuple of (frequencies, power spectral density)
    """
    ts = np.real(time_series).astype(np.float64)
    
    if apply_filter:
        ts = highpass_filter(ts, sfreq, filter_cutoff)
    
    nperseg = int(window_seconds * sfreq)
    if len(ts) < nperseg:
        logger.warning(f"Time series too short: {len(ts)} < {nperseg}")
        return None, None
    
    if method == 'welch':
        freqs, psd = signal.welch(
            ts, fs=sfreq, window='hann',
            nperseg=nperseg, noverlap=nperseg // 2,
            nfft=nperseg, scaling='density'
        )
    else:
        raise ValueError(f"Unknown spectrum method: {method}")
    
    return freqs, psd


def get_band_power(
    time_series: np.ndarray,
    sfreq: float,
    band: Tuple[float, float],
    **spectrum_kwargs
) -> float:
    """
    Get integrated power in a frequency band (linear scale)
    
    Args:
        time_series: Time series data
        sfreq: Sampling frequency in Hz
        band: (low_freq, high_freq) tuple in Hz
        **spectrum_kwargs: Additional arguments for get_spectrum
    
    Returns:
        Integrated power in the band
    """
    freqs, psd = get_spectrum(time_series, sfreq, **spectrum_kwargs)
    if freqs is None or psd is None:
        return np.nan
    
    band_mask = (freqs >= band[0]) & (freqs <= band[1])
    if not np.any(band_mask):
        return np.nan
    
    return float(trapezoid(psd[band_mask], freqs[band_mask]))


def get_band_power_db(
    time_series: np.ndarray,
    sfreq: float,
    band: Tuple[float, float],
    **spectrum_kwargs
) -> float:
    """
    Get band power in dB
    
    Args:
        time_series: Time series data
        sfreq: Sampling frequency in Hz
        band: (low_freq, high_freq) tuple in Hz
        **spectrum_kwargs: Additional arguments for get_spectrum
    
    Returns:
        Band power in dB
    """
    power = get_band_power(time_series, sfreq, band, **spectrum_kwargs)
    return 10.0 * np.log10(power + 1e-30) if not np.isnan(power) else np.nan


def align_spectrum_to_zero(
    time_series: np.ndarray,
    sfreq: float,
    ref_range: Tuple[float, float] = (1, 3),
    **spectrum_kwargs
) -> np.ndarray:
    """
    Scale time series so that low-frequency power is 1 (0 dB after log transform)
    
    Args:
        time_series: Time series data
        sfreq: Sampling frequency in Hz
        ref_range: Reference frequency range for alignment
        **spectrum_kwargs: Additional arguments for get_spectrum
    
    Returns:
        Scaled time series
    """
    freqs, psd = get_spectrum(time_series, sfreq, **spectrum_kwargs)
    if freqs is None or psd is None:
        return time_series
    
    ref_mask = (freqs >= ref_range[0]) & (freqs <= ref_range[1])
    if not np.any(ref_mask):
        ref_mask = np.zeros_like(freqs, dtype=bool)
        ref_mask[:min(5, len(freqs))] = True
    
    ref_power = np.mean(psd[ref_mask])
    
    if ref_power > 0:
        scale_factor = 1.0 / ref_power
        return time_series * np.sqrt(scale_factor)
    
    return time_series


# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

def load_consolidated_data(ieeg_path: Path, lcmv_path: Path) -> Tuple[Dict, Dict]:
    """Load iEEG and LCMV data from consolidated npz files"""
    for path, name in [(ieeg_path, 'iEEG'), (lcmv_path, 'LCMV')]:
        if not path.exists():
            raise FileNotFoundError(f"{name} data not found: {path}")
    
    logger.info("Loading iEEG and LCMV data...")
    
    ieeg_file = np.load(ieeg_path, allow_pickle=True)
    lcmv_file = np.load(lcmv_path, allow_pickle=True)
    
    ieeg_data = ieeg_file['data'].item()
    lcmv_data = lcmv_file['data'].item()
    
    logger.info(f"✓ Loaded {len(ieeg_data)} iEEG subjects, {len(lcmv_data)} LCMV subjects")
    
    return ieeg_data, lcmv_data


def get_subject_data(
    ieeg_data: Dict,
    lcmv_data: Dict,
    subject: str,
    analysis_type: AnalysisType,
    config: STNConfig = DEFAULT_CONFIG,
    runs: Optional[List[RunType]] = None,
    trim_seconds: Optional[float] = None,
    apply_filter: bool = True,
    align_to_zero: bool = False
) -> Optional[Dict]:
    """
    Extract and prepare data for a single subject.
    
    Args:
        ieeg_data: iEEG data dictionary
        lcmv_data: LCMV data dictionary
        subject: Subject ID
        analysis_type: 'atlas' or 'voxel'
        config: Configuration object
        runs: List of runs to extract (default: all runs)
        trim_seconds: Seconds to trim from start/end (None = no trimming)
        apply_filter: Whether to apply high-pass filter
        align_to_zero: Whether to align spectrum to 0 dB
    
    Returns:
        Dictionary with subject data or None if missing required data
    """
    if subject not in ieeg_data or subject not in lcmv_data:
        return None
    
    if runs is None:
        runs = ['c', 'o', 'l', 'r']
    
    subj_ieeg = ieeg_data[subject]
    subj_lcmv = lcmv_data[subject]
    
    ieeg_sfreq = subj_ieeg.get('meta', {}).get('sfreq', config.default_sfreq['ieeg'])
    lcmv_sfreq = subj_lcmv.get('meta', {}).get('sfreq', config.default_sfreq['lcmv'])
    
    roi_patterns = config.roi_patterns[analysis_type]
    
    result = {}
    
    for region in ['L_STN', 'R_STN']:
        # Get iEEG data
        ieeg_runs = {}
        for run in runs:
            if run in subj_ieeg and region in subj_ieeg[run]:
                ts = subj_ieeg[run][region].copy()
                if ts.ndim > 1:
                    ts = ts.flatten()
                
                # Apply preprocessing
                ts_processed = ts
                if trim_seconds is not None:
                    ts_processed = trim_trial_edges(ts_processed, ieeg_sfreq, trim_seconds)
                if apply_filter:
                    ts_processed = highpass_filter(ts_processed, ieeg_sfreq, config.default_filter_cutoff)
                if align_to_zero:
                    ts_processed = align_spectrum_to_zero(
                        ts_processed, ieeg_sfreq, config.ref_freq_range,
                        window_seconds=config.default_window_seconds
                    )
                
                ieeg_runs[run] = ts_processed
        
        # Get LCMV data
        lcmv_runs = {}
        for run in runs:
            if run in subj_lcmv:
                for pattern in roi_patterns[region]:
                    if pattern in subj_lcmv[run]:
                        ts = subj_lcmv[run][pattern].copy()
                        if ts.ndim > 1:
                            ts = ts.flatten()
                        
                        # Apply preprocessing
                        ts_processed = ts
                        if trim_seconds is not None:
                            ts_processed = trim_trial_edges(ts_processed, lcmv_sfreq, trim_seconds)
                        if apply_filter:
                            ts_processed = highpass_filter(ts_processed, lcmv_sfreq, config.default_filter_cutoff)
                        if align_to_zero:
                            ts_processed = align_spectrum_to_zero(
                                ts_processed, lcmv_sfreq, config.ref_freq_range,
                                window_seconds=config.default_window_seconds
                            )
                        
                        lcmv_runs[run] = ts_processed
                        break
        
        # Only include if we have both iEEG and LCMV data for at least some runs
        if ieeg_runs and lcmv_runs:
            result[region] = {
                'ieeg': ieeg_runs,
                'lcmv': lcmv_runs,
                'ieeg_sfreq': ieeg_sfreq,
                'lcmv_sfreq': lcmv_sfreq
            }
    
    return result if result else None


def collect_all_data(
    ieeg_data: Dict,
    lcmv_data: Dict,
    analysis_type: AnalysisType,
    config: STNConfig = DEFAULT_CONFIG,
    subjects: Optional[List[str]] = None,
    runs: Optional[List[RunType]] = None,
    trim_seconds: Optional[float] = None,
    apply_filter: bool = True,
    align_to_zero: bool = False
) -> Dict:
    """
    Collect data for all valid subjects
    
    Args:
        ieeg_data: iEEG data dictionary
        lcmv_data: LCMV data dictionary
        analysis_type: 'atlas' or 'voxel'
        config: Configuration object
        subjects: List of subject IDs to process (default: config.valid_subjects)
        runs: List of runs to extract
        trim_seconds: Seconds to trim from start/end (None = no trimming)
        apply_filter: Whether to apply high-pass filter
        align_to_zero: Whether to align spectrum to 0 dB
    
    Returns:
        Dictionary with collected data for all valid subjects
    """
    if subjects is None:
        subjects = config.valid_subjects
    
    collected = {}
    
    for subj in subjects:
        subj_data = get_subject_data(
            ieeg_data, lcmv_data, subj, analysis_type, config,
            runs=runs, trim_seconds=trim_seconds, 
            apply_filter=apply_filter, align_to_zero=align_to_zero
        )
        if subj_data:
            collected[subj] = subj_data
            logger.info(f"✓ Added {subj}: {list(subj_data.keys())}")
    
    logger.info(f"\nCollected data for {len(collected)} subjects")
    return collected


# =============================================================================
# ANALYSIS FUNCTIONS - POWER CHANGE (Script 1)
# =============================================================================

def compute_power_changes(
    subject_data: Dict,
    config: STNConfig = DEFAULT_CONFIG,
    bands: Optional[List[str]] = None,
    **spectrum_kwargs
) -> Dict:
    """
    Compute power changes for all bands and conditions.
    Power change = P_task - P_baseline (in dB)
    
    Args:
        subject_data: Subject data from get_subject_data
        config: Configuration object
        bands: List of band names to analyze (default: all bands)
        **spectrum_kwargs: Additional arguments for get_spectrum
    
    Returns:
        Nested dict with results for each region and band
    """
    if bands is None:
        bands = config.get_band_list()
    
    results = {}
    
    for region, data in subject_data.items():
        region_results = {}
        
        for band_name in bands:
            if band_name not in config.beta_bands:
                continue
                
            band_limits = config.beta_bands[band_name]
            
            # Get powers in dB
            powers = {}
            for modality in ['ieeg', 'lcmv']:
                sfreq_key = f'{modality}_sfreq'
                for run in ['c', 'o', 'l', 'r']:
                    if run in data[modality]:
                        key = f'{modality}_{run}'
                        powers[key] = get_band_power_db(
                            data[modality][run], data[sfreq_key], band_limits, **spectrum_kwargs)
                    else:
                        powers[f'{modality}_{run}'] = np.nan
            
            # Compute changes relative to baselines
            changes = {
                'left_hand': {
                    'ieeg_c': powers['ieeg_l'] - powers['ieeg_c'] if not np.isnan(powers['ieeg_l']) and not np.isnan(powers['ieeg_c']) else np.nan,
                    'lcmv_c': powers['lcmv_l'] - powers['lcmv_c'] if not np.isnan(powers['lcmv_l']) and not np.isnan(powers['lcmv_c']) else np.nan,
                    'ieeg_o': powers['ieeg_l'] - powers['ieeg_o'] if not np.isnan(powers['ieeg_l']) and not np.isnan(powers['ieeg_o']) else np.nan,
                    'lcmv_o': powers['lcmv_l'] - powers['lcmv_o'] if not np.isnan(powers['lcmv_l']) and not np.isnan(powers['lcmv_o']) else np.nan,
                },
                'right_hand': {
                    'ieeg_c': powers['ieeg_r'] - powers['ieeg_c'] if not np.isnan(powers['ieeg_r']) and not np.isnan(powers['ieeg_c']) else np.nan,
                    'lcmv_c': powers['lcmv_r'] - powers['lcmv_c'] if not np.isnan(powers['lcmv_r']) and not np.isnan(powers['lcmv_c']) else np.nan,
                    'ieeg_o': powers['ieeg_r'] - powers['ieeg_o'] if not np.isnan(powers['ieeg_r']) and not np.isnan(powers['ieeg_o']) else np.nan,
                    'lcmv_o': powers['lcmv_r'] - powers['lcmv_o'] if not np.isnan(powers['lcmv_r']) and not np.isnan(powers['lcmv_o']) else np.nan,
                }
            }
            
            region_results[band_name] = changes
        
        results[region] = region_results
    
    return results


def get_power_match_results(
    computed_results: Dict,
    baseline: str = 'c'
) -> Dict:
    """
    Convert computed power changes to match/suppression results.
    
    Args:
        computed_results: Results from compute_power_changes
        baseline: Baseline type ('c' or 'o')
    
    Returns:
        Dictionary with match/suppression results
    """
    results = {'L_STN': [], 'R_STN': []}
    
    for subject, regions in computed_results.items():
        for region, bands in regions.items():
            for band_name, changes in bands.items():
                lh_match = not np.isnan(changes['left_hand'][f'ieeg_{baseline}']) and \
                           not np.isnan(changes['left_hand'][f'lcmv_{baseline}']) and \
                           changes['left_hand'][f'ieeg_{baseline}'] * changes['left_hand'][f'lcmv_{baseline}'] > 0
                
                rh_match = not np.isnan(changes['right_hand'][f'ieeg_{baseline}']) and \
                           not np.isnan(changes['right_hand'][f'lcmv_{baseline}']) and \
                           changes['right_hand'][f'ieeg_{baseline}'] * changes['right_hand'][f'lcmv_{baseline}'] > 0
                
                lh_suppress = not np.isnan(changes['left_hand'][f'ieeg_{baseline}']) and \
                              not np.isnan(changes['left_hand'][f'lcmv_{baseline}']) and \
                              changes['left_hand'][f'ieeg_{baseline}'] < 0 and changes['left_hand'][f'lcmv_{baseline}'] < 0
                
                rh_suppress = not np.isnan(changes['right_hand'][f'ieeg_{baseline}']) and \
                              not np.isnan(changes['right_hand'][f'lcmv_{baseline}']) and \
                              changes['right_hand'][f'ieeg_{baseline}'] < 0 and changes['right_hand'][f'lcmv_{baseline}'] < 0
                
                results[region].append({
                    'subject': subject,
                    'band': band_name,
                    'lh_match': lh_match,
                    'rh_match': rh_match,
                    'lh_suppress': lh_suppress,
                    'rh_suppress': rh_suppress
                })
    
    return results


# =============================================================================
# ANALYSIS FUNCTIONS - ERSP (Script 2)
# =============================================================================

def compute_ersp(
    subject_data: Dict,
    config: STNConfig = DEFAULT_CONFIG,
    bands: Optional[List[str]] = None,
    baselines: Optional[List[str]] = None,
    **spectrum_kwargs
) -> Dict:
    """
    Compute ERSP for all conditions.
    ERSP (dB) = 10 * log10(P_task / P_baseline)
    
    Args:
        subject_data: Subject data from get_subject_data
        config: Configuration object
        bands: List of band names to analyze (default: all bands)
        baselines: List of baseline types to use (default: ['c'])
        **spectrum_kwargs: Additional arguments for get_spectrum
    
    Returns:
        Nested dict with results for each region and band
    """
    if bands is None:
        bands = config.get_band_list()
    
    if baselines is None:
        baselines = ['c']
    
    results = {}
    
    for region, data in subject_data.items():
        region_results = {}
        
        for band_name in bands:
            if band_name not in config.beta_bands:
                continue
                
            band_limits = config.beta_bands[band_name]
            band_results = {
                'left_hand': {},
                'right_hand': {}
            }
            
            for baseline in baselines:
                if baseline not in data['ieeg'] or baseline not in data['lcmv']:
                    continue
                
                # Compute ERSP for each task
                for task in ['l', 'r']:
                    task_name = 'left_hand' if task == 'l' else 'right_hand'
                    
                    if task not in data['ieeg'] or task not in data['lcmv']:
                        continue
                    
                    # Compute for iEEG
                    p_task_ieeg = get_band_power(
                        data['ieeg'][task], data['ieeg_sfreq'], 
                        band_limits, **spectrum_kwargs
                    )
                    p_baseline_ieeg = get_band_power(
                        data['ieeg'][baseline], data['ieeg_sfreq'], 
                        band_limits, **spectrum_kwargs
                    )
                    
                    if not np.isnan(p_task_ieeg) and not np.isnan(p_baseline_ieeg) and p_baseline_ieeg > 0:
                        ersp_ieeg = 10.0 * np.log10(p_task_ieeg / p_baseline_ieeg)
                    else:
                        ersp_ieeg = np.nan
                    
                    # Compute for LCMV
                    p_task_lcmv = get_band_power(
                        data['lcmv'][task], data['lcmv_sfreq'], 
                        band_limits, **spectrum_kwargs
                    )
                    p_baseline_lcmv = get_band_power(
                        data['lcmv'][baseline], data['lcmv_sfreq'], 
                        band_limits, **spectrum_kwargs
                    )
                    
                    if not np.isnan(p_task_lcmv) and not np.isnan(p_baseline_lcmv) and p_baseline_lcmv > 0:
                        ersp_lcmv = 10.0 * np.log10(p_task_lcmv / p_baseline_lcmv)
                    else:
                        ersp_lcmv = np.nan
                    
                    band_results[task_name][f'ieeg_{baseline}'] = ersp_ieeg
                    band_results[task_name][f'lcmv_{baseline}'] = ersp_lcmv
            
            region_results[band_name] = band_results
        
        results[region] = region_results
    
    return results


def get_ersp_match_results(
    computed_results: Dict,
    baseline: str = 'c'
) -> Dict:
    """
    Convert computed ERSP to match/suppression results.
    
    Args:
        computed_results: Results from compute_ersp
        baseline: Baseline type ('c' or 'o')
    
    Returns:
        Dictionary with match/suppression results
    """
    results = {'L_STN': [], 'R_STN': []}
    
    for subject, regions in computed_results.items():
        for region, bands in regions.items():
            for band_name, changes in bands.items():
                lh_match = not np.isnan(changes['left_hand'].get(f'ieeg_{baseline}', np.nan)) and \
                           not np.isnan(changes['left_hand'].get(f'lcmv_{baseline}', np.nan)) and \
                           changes['left_hand'][f'ieeg_{baseline}'] * changes['left_hand'][f'lcmv_{baseline}'] > 0
                
                rh_match = not np.isnan(changes['right_hand'].get(f'ieeg_{baseline}', np.nan)) and \
                           not np.isnan(changes['right_hand'].get(f'lcmv_{baseline}', np.nan)) and \
                           changes['right_hand'][f'ieeg_{baseline}'] * changes['right_hand'][f'lcmv_{baseline}'] > 0
                
                lh_suppress = not np.isnan(changes['left_hand'].get(f'ieeg_{baseline}', np.nan)) and \
                              not np.isnan(changes['left_hand'].get(f'lcmv_{baseline}', np.nan)) and \
                              changes['left_hand'][f'ieeg_{baseline}'] < 0 and changes['left_hand'][f'lcmv_{baseline}'] < 0
                
                rh_suppress = not np.isnan(changes['right_hand'].get(f'ieeg_{baseline}', np.nan)) and \
                              not np.isnan(changes['right_hand'].get(f'lcmv_{baseline}', np.nan)) and \
                              changes['right_hand'][f'ieeg_{baseline}'] < 0 and changes['right_hand'][f'lcmv_{baseline}'] < 0
                
                results[region].append({
                    'subject': subject,
                    'band': band_name,
                    'lh_match': lh_match,
                    'rh_match': rh_match,
                    'lh_suppress': lh_suppress,
                    'rh_suppress': rh_suppress
                })
    
    return results


# =============================================================================
# STATISTICAL ANALYSIS FUNCTIONS
# =============================================================================

def calculate_spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    """Calculate Spearman's rank correlation coefficient"""
    mask = ~(np.isnan(x) | np.isnan(y))
    x_clean = x[mask]
    y_clean = y[mask]
    
    if len(x_clean) < 3:
        return np.nan
    
    x_rank = rankdata(x_clean)
    y_rank = rankdata(y_clean)
    
    x_rank_centered = x_rank - np.mean(x_rank)
    y_rank_centered = y_rank - np.mean(y_rank)
    
    numerator = np.sum(x_rank_centered * y_rank_centered)
    denominator = np.sqrt(np.sum(x_rank_centered**2) * np.sum(y_rank_centered**2))
    
    if denominator == 0:
        return np.nan
    
    return numerator / denominator


def spearman_p_value(rho: float, n: int) -> float:
    """Calculate p-value for Spearman's rho using t-distribution approximation"""
    if np.isnan(rho) or n < 3:
        return np.nan
    
    if abs(rho) >= 1.0:
        return 0.0
    
    t_stat = rho * np.sqrt((n - 2) / (1 - rho**2))
    p_value = 2 * (1 - t.cdf(abs(t_stat), df=n-2))
    
    return p_value


def bootstrap_confidence_interval(
    x: np.ndarray,
    y: np.ndarray,
    n_bootstrap: int = 2000,
    ci_level: float = 95
) -> Tuple[float, float, float, np.ndarray]:
    """Calculate bootstrap confidence interval for Spearman's rho"""
    mask = ~(np.isnan(x) | np.isnan(y))
    x_clean = x[mask]
    y_clean = y[mask]
    
    if len(x_clean) < 3:
        return np.nan, np.nan, np.nan, np.array([])
    
    n = len(x_clean)
    rho_original = calculate_spearman_rho(x_clean, y_clean)
    
    bootstrap_rhos = []
    indices = np.arange(n)
    
    for _ in range(n_bootstrap):
        bootstrap_idx = np.random.choice(indices, size=n, replace=True)
        x_boot = x_clean[bootstrap_idx]
        y_boot = y_clean[bootstrap_idx]
        
        rho_boot = calculate_spearman_rho(x_boot, y_boot)
        if not np.isnan(rho_boot):
            bootstrap_rhos.append(rho_boot)
    
    bootstrap_rhos = np.array(bootstrap_rhos)
    
    if len(bootstrap_rhos) < 100:
        return rho_original, np.nan, np.nan, bootstrap_rhos
    
    alpha = (100 - ci_level) / 2
    ci_lower = np.percentile(bootstrap_rhos, alpha)
    ci_upper = np.percentile(bootstrap_rhos, 100 - alpha)
    
    return rho_original, ci_lower, ci_upper, bootstrap_rhos


def calculate_directional_agreement(
    ieeg_values: np.ndarray,
    lcmv_values: np.ndarray
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Calculate percentage where iEEG and LCMV agree on sign"""
    mask = ~(np.isnan(ieeg_values) | np.isnan(lcmv_values))
    ieeg_clean = ieeg_values[mask]
    lcmv_clean = lcmv_values[mask]
    
    if len(ieeg_clean) == 0:
        return np.nan, np.array([]), np.array([]), np.array([])
    
    sign_ieeg = np.sign(ieeg_clean)
    sign_lcmv = np.sign(lcmv_clean)
    
    agreement_mask = (sign_ieeg == sign_lcmv)
    agreement_pct = np.mean(agreement_mask) * 100
    
    return agreement_pct, sign_ieeg, sign_lcmv, agreement_mask


def analyze_correlations(
    computed_results: Dict,
    config: STNConfig = DEFAULT_CONFIG,
    baseline: str = 'c',
    bands: Optional[List[str]] = None,
    conditions: Optional[List[str]] = None
) -> Dict:
    """
    Perform comprehensive correlation analysis on computed results
    
    Args:
        computed_results: Results from compute_ersp or compute_power_changes
        config: Configuration object
        baseline: Baseline type ('c' or 'o')
        bands: List of bands to analyze (default: all)
        conditions: List of conditions to analyze (default: ['left_hand', 'right_hand'])
    
    Returns:
        Dictionary with correlation results for each band and condition
    """
    logger.info("\n" + "="*60)
    logger.info(f"STATISTICAL ANALYSIS: SPEARMAN CORRELATION WITH BOOTSTRAP CI")
    logger.info("="*60)
    
    if bands is None:
        bands = config.get_band_list()
    
    if conditions is None:
        conditions = ['left_hand', 'right_hand']
    
    correlation_results = {}
    
    for band in bands:
        band_results = {}
        
        for condition in conditions:
            key = f"{condition}_{baseline}"
            
            # Collect all subject data for this band/condition
            ieeg_values = []
            lcmv_values = []
            subject_labels = []
            
            for subject, regions in computed_results.items():
                for region in ['L_STN', 'R_STN']:
                    if region in regions and band in regions[region]:
                        data = regions[region][band].get(condition, {})
                        
                        ieeg_val = data.get(f'ieeg_{baseline}', np.nan)
                        lcmv_val = data.get(f'lcmv_{baseline}', np.nan)
                        
                        if not np.isnan(ieeg_val) and not np.isnan(lcmv_val):
                            ieeg_values.append(ieeg_val)
                            lcmv_values.append(lcmv_val)
                            subject_labels.append(f"{subject}_{region}")
            
            ieeg_array = np.array(ieeg_values)
            lcmv_array = np.array(lcmv_values)
            
            n_valid = len(ieeg_array)
            
            if n_valid >= 3:
                # Spearman's rho
                rho = calculate_spearman_rho(ieeg_array, lcmv_array)
                
                # P-value
                p_value = spearman_p_value(rho, n_valid)
                
                # Bootstrap CI
                rho_orig, ci_low, ci_high, bootstrap_dist = bootstrap_confidence_interval(
                    ieeg_array, lcmv_array, 
                    n_bootstrap=config.default_n_bootstrap,
                    ci_level=config.default_ci_level
                )
                
                # Directional agreement
                agreement_pct, sign_i, sign_l, agree_mask = calculate_directional_agreement(
                    ieeg_array, lcmv_array
                )
                
                band_results[key] = {
                    'n_subjects': n_valid,
                    'n_pairs': n_valid,
                    'spearman_rho': rho,
                    'p_value': p_value,
                    'ci_95_lower': ci_low,
                    'ci_95_upper': ci_high,
                    'directional_agreement_pct': agreement_pct,
                    'ieeg_values': ieeg_array.tolist(),
                    'lcmv_values': lcmv_array.tolist(),
                    'subject_labels': subject_labels,
                    'agreement_mask': agree_mask.tolist() if len(agree_mask) > 0 else []
                }
            else:
                band_results[key] = {
                    'n_subjects': n_valid,
                    'n_pairs': n_valid,
                    'spearman_rho': np.nan,
                    'p_value': np.nan,
                    'ci_95_lower': np.nan,
                    'ci_95_upper': np.nan,
                    'directional_agreement_pct': np.nan,
                    'ieeg_values': [],
                    'lcmv_values': [],
                    'subject_labels': [],
                    'agreement_mask': []
                }
            
            # Log results
            if n_valid >= 3:
                sig_star = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
                logger.info(f"\n{band} - {condition} (baseline={baseline}):")
                logger.info(f"  N={n_valid}, ρ={rho:.3f}{sig_star}, p={p_value:.4f}")
                logger.info(f"  95% CI: [{ci_low:.3f}, {ci_high:.3f}]")
                logger.info(f"  Directional agreement: {agreement_pct:.1f}%")
        
        correlation_results[band] = band_results
    
    return correlation_results


def print_correlation_summary(correlation_results: Dict, baseline: str = 'c'):
    """Print a formatted summary of correlation results"""
    logger.info("\n" + "="*70)
    logger.info(f"SPEARMAN CORRELATION SUMMARY (baseline={baseline})")
    logger.info("="*70)
    
    header = (f"{'Band':<12} {'Cond':<6} {'N':<4} "
              f"{'Corr':<8} {'P-val':<8} {'CI_low':<8} {'CI_high':<8} {'Agree%':<8}")
    logger.info(header)
    logger.info("-" * 70)
    
    for band, band_results in correlation_results.items():
        band_short = band.replace('beta_', '')
        
        for condition_key, results in band_results.items():
            if results['n_subjects'] >= 3:
                rho = results['spearman_rho']
                p = results['p_value']
                ci_low = results['ci_95_lower']
                ci_high = results['ci_95_upper']
                agree = results['directional_agreement_pct']
                
                # Format p-value with stars
                if p < 0.001:
                    p_str = f"{p:.1e}***"
                elif p < 0.01:
                    p_str = f"{p:.3f}**"
                elif p < 0.05:
                    p_str = f"{p:.3f}*"
                else:
                    p_str = f"{p:.3f}"
                
                # Format condition
                if 'left_hand' in condition_key:
                    cond_short = 'LH'
                else:
                    cond_short = 'RH'
                
                logger.info(f"{band_short:<12} {cond_short:<6} {results['n_subjects']:<4} "
                           f"{rho:<8.3f} {p_str:<8} {ci_low:<8.3f} {ci_high:<8.3f} {agree:<8.1f}")
    
    logger.info("-" * 70)
    logger.info("Corr = Spearman's ρ | P-val = p-value | CI = 95% Bootstrap CI | Agree% = Directional agreement")
    logger.info("Significance: *** p<0.001, ** p<0.01, * p<0.05")

# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================

def plot_power_spectra(
    subject_data: Dict,
    subject: str,
    region: str,
    config: STNConfig = DEFAULT_CONFIG,
    runs: Optional[List[Tuple[str, str]]] = None,
    save_path: Optional[Path] = None,
    show: bool = True,
    **spectrum_kwargs
):
    """Plot power spectra for specified conditions"""
    if runs is None:
        runs = [('c', 'Eyes Closed'), ('o', 'Eyes Open'),
                ('l', 'Left Hand'), ('r', 'Right Hand')]
    
    colors = {'c': 'blue', 'o': 'cyan', 'l': 'red', 'r': 'green'}
    
    n_conditions = len(runs)
    fig, axes = plt.subplots(2, n_conditions, figsize=(5*n_conditions, 10))
    
    # Collect data for y-limits
    all_psd_values = []
    for modality in ['ieeg', 'lcmv']:
        for run, _ in runs:
            if run in subject_data[modality]:
                freqs, psd = get_spectrum(
                    subject_data[modality][run], 
                    subject_data[f'{modality}_sfreq'],
                    **spectrum_kwargs
                )
                if freqs is not None:
                    psd_db = 10 * np.log10(psd + 1e-30)
                    beta_mask = (freqs >= 13) & (freqs <= 35)
                    if np.any(beta_mask):
                        all_psd_values.extend(psd_db[beta_mask])
    
    # Set y-limits
    if all_psd_values:
        p5, p95 = np.percentile(all_psd_values, [5, 95])
        y_max = max(abs(p5), abs(p95)) * 1.2
        y_lim = [-y_max, y_max]
    else:
        y_lim = [-5, 5]
    
    # Plot
    for row, modality in enumerate(['ieeg', 'lcmv']):
        for col, (run, label) in enumerate(runs):
            ax = axes[row, col]
            
            if run not in subject_data[modality]:
                ax.text(0.5, 0.5, f'No {run} data', ha='center', va='center')
                continue
            
            freqs, psd = get_spectrum(
                subject_data[modality][run],
                subject_data[f'{modality}_sfreq'],
                **spectrum_kwargs
            )
            
            if freqs is not None:
                psd_db = 10 * np.log10(psd + 1e-30)
                
                ax.plot(freqs, psd_db, color=colors[run], linewidth=1.5, label=label)
                
                # Highlight beta bands
                for band_name, (fmin, fmax) in config.beta_bands.items():
                    color = config.get_band_color(band_name)
                    ax.axvspan(fmin, fmax, color=color, alpha=0.15)
                
                ax.axhline(y=0, color='black', linestyle='--', alpha=0.3, linewidth=0.5)
                ax.set_xlabel('Frequency (Hz)')
                ax.set_ylabel('Power (dB)')
                ax.set_title(f'{modality.upper()} - {label}')
                ax.set_xlim([1, 35])
                ax.set_ylim(y_lim)
                ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'{subject} - {region}: Power Spectra', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if show:
        plt.show()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close(fig)


def plot_subject_summary(
    computed_results: Dict,
    subject: str,
    region: str,
    config: STNConfig = DEFAULT_CONFIG,
    baseline: str = 'c',
    analysis_name: str = 'Power Change',
    save_path: Optional[Path] = None
):
    """Plot summary for a single subject"""
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    baseline_name = config.baseline_names.get(baseline, baseline)
    baseline_key = f'ieeg_{baseline}'
    lcmv_key = f'lcmv_{baseline}'
    
    fig.suptitle(f'Subject {subject} - {region}: Beta Subband Analysis\n'
                 f'Baseline: {baseline_name} ({analysis_name})',
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Bar plot
    ax1 = fig.add_subplot(gs[0, :])
    subbands = config.get_band_list()
    x = np.arange(len(subbands))
    width = 0.35
    
    ieeg_means, lcmv_means = [], []
    ieeg_stds, lcmv_stds = [], []
    all_values = []
    
    for band in subbands:
        if band in computed_results[region]:
            data = computed_results[region][band]
            ieeg_vals = [data['left_hand'].get(baseline_key, np.nan), 
                        data['right_hand'].get(baseline_key, np.nan)]
            lcmv_vals = [data['left_hand'].get(lcmv_key, np.nan), 
                        data['right_hand'].get(lcmv_key, np.nan)]
            
            ieeg_means.append(np.nanmean(ieeg_vals))
            lcmv_means.append(np.nanmean(lcmv_vals))
            ieeg_stds.append(np.nanstd(ieeg_vals))
            lcmv_stds.append(np.nanstd(lcmv_vals))
            all_values.extend([v for v in ieeg_vals + lcmv_vals if not np.isnan(v)])
        else:
            ieeg_means.append(np.nan)
            lcmv_means.append(np.nan)
            ieeg_stds.append(0)
            lcmv_stds.append(0)
    
    if all_values:
        y_max = max(abs(min(all_values)), abs(max(all_values))) * 1.2
        bar_ylim = [-y_max, y_max]
        if bar_ylim[1] - bar_ylim[0] < 4:
            bar_ylim = [-2, 2]
    else:
        bar_ylim = [-5, 5]
    
    ax1.bar(x - width/2, ieeg_means, width, yerr=ieeg_stds,
            label='iEEG', color='blue', alpha=0.7, edgecolor='black',
            capsize=3, error_kw={'linewidth': 1})
    ax1.bar(x + width/2, lcmv_means, width, yerr=lcmv_stds,
            label='LCMV', color='red', alpha=0.7, edgecolor='black',
            capsize=3, error_kw={'linewidth': 1}, hatch='//')
    
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{int(config.beta_bands[b][0])}-{int(config.beta_bands[b][1])}" 
                         for b in subbands])
    ax1.set_ylabel(f'{analysis_name} (dB)')
    ax1.set_title(f'Beta Subbands (Average LH/RH) - Baseline: {baseline_name}')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim(bar_ylim)
    
    # Scatter plots for key bands (middle bands)
    n_scatter = min(3, len(subbands))
    if n_scatter > 0:
        middle_idx = len(subbands) // 2
        key_bands = subbands[max(0, middle_idx-1):min(len(subbands), middle_idx+2)]
        axes_scatter = [fig.add_subplot(gs[1, i]) for i in range(n_scatter)]
        
        for idx, (band_name, ax) in enumerate(zip(key_bands, axes_scatter)):
            if band_name in computed_results[region]:
                data = computed_results[region][band_name]
                ieeg_vals = [data['left_hand'].get(baseline_key, np.nan), 
                            data['right_hand'].get(baseline_key, np.nan)]
                lcmv_vals = [data['left_hand'].get(lcmv_key, np.nan), 
                            data['right_hand'].get(lcmv_key, np.nan)]
                
                valid_vals = [v for v in ieeg_vals + lcmv_vals if not np.isnan(v)]
                if valid_vals:
                    scatter_max = max(abs(min(valid_vals)), abs(max(valid_vals))) * 1.2
                    scatter_lim = [-scatter_max, scatter_max]
                else:
                    scatter_lim = [-5, 5]
                
                valid = [(i, l) for i, l in zip(ieeg_vals, lcmv_vals)
                         if not np.isnan(i) and not np.isnan(l)]
                
                if valid:
                    ieeg_plot, lcmv_plot = zip(*valid)
                    
                    colors = ['darkgreen' if i < 0 and l < 0 else 'green' if i * l > 0 else 'red'
                             for i, l in valid]
                    
                    ax.scatter(ieeg_plot, lcmv_plot, c=colors, s=150,
                              alpha=0.7, edgecolors='black', linewidth=2)
                    
                    for i, (ieeg, lcmv) in enumerate(valid):
                        ax.annotate(['LH', 'RH'][i], (ieeg, lcmv), fontsize=10,
                                    xytext=(5, 5), textcoords='offset points',
                                    fontweight='bold')
                
                ax.axhspan(scatter_lim[0], 0, xmin=0, xmax=1, alpha=0.1, color='green')
                ax.axvspan(scatter_lim[0], 0, ymin=0, ymax=1, alpha=0.1, color='green')
                ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
                ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
                ax.set_xlabel(f'iEEG {analysis_name} (dB)')
                ax.set_ylabel(f'LCMV {analysis_name} (dB)')
                band_range = config.beta_bands[band_name]
                ax.set_title(f'{band_range[0]:.0f}-{band_range[1]:.0f} Hz')
                ax.grid(True, alpha=0.3)
                ax.set_xlim(scatter_lim)
                ax.set_ylim(scatter_lim)
                ax.set_aspect('equal')
    
    # Summary table
    ax4 = fig.add_subplot(gs[2, :])
    ax4.axis('off')
    
    table_data = [['Band', 'Range (Hz)', 'LH iEEG', 'LH LCMV', 'Match', 'RH iEEG', 'RH LCMV', 'Match']]
    
    for band_name in subbands:
        if band_name in computed_results[region]:
            data = computed_results[region][band_name]
            band_range = f"{config.beta_bands[band_name][0]:.0f}-{config.beta_bands[band_name][1]:.0f}"
            
            lh_ieeg = data['left_hand'].get(baseline_key, np.nan)
            lh_lcmv = data['left_hand'].get(lcmv_key, np.nan)
            rh_ieeg = data['right_hand'].get(baseline_key, np.nan)
            rh_lcmv = data['right_hand'].get(lcmv_key, np.nan)
            
            def get_match(ieeg, lcmv):
                if np.isnan(ieeg) or np.isnan(lcmv):
                    return '?'
                if ieeg < 0 and lcmv < 0:
                    return '✓✓'
                elif ieeg * lcmv > 0:
                    return '✓'
                return '✗'
            
            table_data.append([
                band_name,
                band_range,
                f'{lh_ieeg:+.2f}' if not np.isnan(lh_ieeg) else 'N/A',
                f'{lh_lcmv:+.2f}' if not np.isnan(lh_lcmv) else 'N/A',
                get_match(lh_ieeg, lh_lcmv),
                f'{rh_ieeg:+.2f}' if not np.isnan(rh_ieeg) else 'N/A',
                f'{rh_lcmv:+.2f}' if not np.isnan(rh_lcmv) else 'N/A',
                get_match(rh_ieeg, rh_lcmv)
            ])
    
    table = ax4.table(cellText=table_data, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    
    plt.tight_layout()
    plt.show()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close(fig)


def plot_group_summary(
    match_results: Dict,
    config: STNConfig = DEFAULT_CONFIG,
    baseline: str = 'c',
    analysis_name: str = 'Power Change',
    title_extra: str = '',
    save_path: Optional[Path] = None
):
    """Plot group summary for specified baseline"""
    
    if not any(match_results.values()):
        return
        
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    baseline_name = config.baseline_names.get(baseline, baseline)
    
    fig.suptitle(f'STN Validation: Group Summary - {baseline_name} Baseline\n'
                f'Valid Subjects: {", ".join(config.valid_subjects)}{title_extra}',
                fontsize=16, fontweight='bold', y=0.98)
    
    subbands = config.get_band_list()
    subband_labels = [f"{int(config.beta_bands[b][0])}-{int(config.beta_bands[b][1])}" 
                     for b in subbands]
    
    # Agreement by region
    ax1 = fig.add_subplot(gs[0, :2])
    x = np.arange(len(subbands))
    width = 0.35
    
    l_stn_agree, r_stn_agree = [], []
    l_stn_suppress, r_stn_suppress = [], []
    
    for band in subbands:
        for region, agree_list, suppress_list in [
            ('L_STN', l_stn_agree, l_stn_suppress),
            ('R_STN', r_stn_agree, r_stn_suppress)
        ]:
            if region in match_results:
                band_data = [r for r in match_results[region] if r['band'] == band]
                if band_data:
                    matches = sum([(1 if r['lh_match'] else 0) + (1 if r['rh_match'] else 0) 
                                  for r in band_data])
                    suppresses = sum([(1 if r['lh_suppress'] else 0) + (1 if r['rh_suppress'] else 0)
                                     for r in band_data])
                    total = len(band_data) * 2
                    agree_list.append(matches/total * 100)
                    suppress_list.append(suppresses/total * 100)
                else:
                    agree_list.append(0)
                    suppress_list.append(0)
            else:
                agree_list.append(0)
                suppress_list.append(0)
    
    ax1.bar(x - width/2, l_stn_agree, width, label='L_STN (agreement)',
            color='blue', alpha=0.7, edgecolor='black')
    ax1.bar(x + width/2, r_stn_agree, width, label='R_STN (agreement)',
            color='red', alpha=0.7, edgecolor='black')
    ax1.bar(x - width/2, l_stn_suppress, width, label='L_STN (suppression)',
            color='blue', alpha=0.3, edgecolor='black', hatch='///')
    ax1.bar(x + width/2, r_stn_suppress, width, label='R_STN (suppression)',
            color='red', alpha=0.3, edgecolor='black', hatch='///')
    
    ax1.axhline(y=60, color='orange', linestyle='--', alpha=0.5, label='60% (Good)')
    ax1.axhline(y=80, color='green', linestyle='--', alpha=0.5, label='80% (Excellent)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(subband_labels)
    ax1.set_ylabel('Percentage (%)')
    ax1.set_title('Subband Agreement and Suppression')
    ax1.legend(loc='upper right', fontsize=7)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim([0, 100])
    
    # Best bands
    ax2 = fig.add_subplot(gs[0, 2])
    
    band_agreement = []
    band_suppression = []
    for band in subbands:
        total_matches = total_suppresses = total_possible = 0
        for region in match_results.values():
            for r in region:
                if r['band'] == band:
                    total_matches += (1 if r['lh_match'] else 0) + (1 if r['rh_match'] else 0)
                    total_suppresses += (1 if r['lh_suppress'] else 0) + (1 if r['rh_suppress'] else 0)
                    total_possible += 2
        if total_possible > 0:
            band_agreement.append(total_matches / total_possible * 100)
            band_suppression.append(total_suppresses / total_possible * 100)
        else:
            band_agreement.append(0)
            band_suppression.append(0)
    
    best_agree_idx = np.argmax(band_agreement) if band_agreement else 0
    best_suppress_idx = np.argmax(band_suppression) if band_suppression else 0
    
    x_comp = np.arange(3)
    width_comp = 0.6
    
    ax2.bar(x_comp[0], band_agreement[best_agree_idx] if band_agreement else 0, width_comp,
            color='green', alpha=0.7, edgecolor='black',
            label=f'Best Agreement\n{subband_labels[best_agree_idx]} Hz')
    ax2.bar(x_comp[1], band_suppression[best_suppress_idx] if band_suppression else 0, width_comp,
            color='blue', alpha=0.7, edgecolor='black',
            label=f'Best Suppression\n{subband_labels[best_suppress_idx]} Hz')
    ax2.bar(x_comp[2], np.mean(band_agreement) if band_agreement else 0, width_comp,
            color='gray', alpha=0.5, edgecolor='black',
            label=f'Mean Agreement\n{np.mean(band_agreement):.1f}%')
    
    ax2.axhline(y=60, color='orange', linestyle='--', alpha=0.5)
    ax2.axhline(y=80, color='green', linestyle='--', alpha=0.5)
    ax2.set_xticks(x_comp)
    ax2.set_xticklabels(['Best Agree', 'Best Suppress', 'Mean'])
    ax2.set_ylabel('Percentage (%)')
    ax2.set_title('Best Performing Subbands')
    ax2.legend(loc='lower right', fontsize=7)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim([0, 100])
    
    # Heatmap
    ax3 = fig.add_subplot(gs[1, :])
    
    all_subjects = sorted(set([r['subject'] for region in match_results.values() for r in region]))
    conditions = [f'{band}\n{metric}' for band in subbands for metric in ['Match', 'Suppress']]
    
    matrix = np.zeros((len(all_subjects), len(conditions)))
    for i, subj in enumerate(all_subjects):
        for j, cond_str in enumerate(conditions):
            band, metric = cond_str.split('\n')
            total = count = 0
            for region in match_results.values():
                for r in region:
                    if r['subject'] == subj and r['band'] == band:
                        if metric == 'Match':
                            total += (1 if r['lh_match'] else 0) + (1 if r['rh_match'] else 0)
                        else:
                            total += (1 if r['lh_suppress'] else 0) + (1 if r['rh_suppress'] else 0)
                        count += 2
            if count > 0:
                matrix[i, j] = total / count * 100
    
    im = ax3.imshow(matrix, cmap='RdYlGn', vmin=0, vmax=100, aspect='auto')
    ax3.set_yticks(range(len(all_subjects)))
    ax3.set_yticklabels(all_subjects)
    ax3.set_xticks(range(len(conditions)))
    ax3.set_xticklabels(conditions, rotation=90, fontsize=8)
    ax3.set_title(f'Subject Performance by Subband - {baseline_name}')
    plt.colorbar(im, ax=ax3, label='Success Rate (%)', fraction=0.015, pad=0.02)
    
    # Summary text
    ax4 = fig.add_subplot(gs[2, :])
    ax4.axis('off')
    
    subject_performance = {}
    for subj in all_subjects:
        subj_matches = subj_suppresses = subj_total = 0
        for region in match_results.values():
            for r in region:
                if r['subject'] == subj:
                    subj_matches += (1 if r['lh_match'] else 0) + (1 if r['rh_match'] else 0)
                    subj_suppresses += (1 if r['lh_suppress'] else 0) + (1 if r['rh_suppress'] else 0)
                    subj_total += 2
        if subj_total > 0:
            subject_performance[subj] = {
                'match': subj_matches / subj_total * 100,
                'suppress': subj_suppresses / subj_total * 100
            }
    
    summary_text = [
        f"GROUP SUMMARY - {baseline_name.upper()} BASELINE",
        "="*70,
        "",
        "SUBJECT CLASSIFICATION:",
        f"  ✅ EXCELLENT (≥70%): {', '.join([s for s,p in subject_performance.items() if p['match']>=70]) or 'None'}",
        f"  ⚠️ GOOD (50-70%): {', '.join([s for s,p in subject_performance.items() if 50<=p['match']<70]) or 'None'}",
        f"  ❌ POOR (<50%): {', '.join([s for s,p in subject_performance.items() if p['match']<50]) or 'None'}",
        "",
        f"BEST BANDS: Agreement={subband_labels[best_agree_idx]}Hz ({band_agreement[best_agree_idx]:.1f}%), "
        f"Suppression={subband_labels[best_suppress_idx]}Hz ({band_suppression[best_suppress_idx]:.1f}%)"
    ]
    
    y_pos = 0.95
    for line in summary_text:
        ax4.text(0.05, y_pos, line, transform=ax4.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace')
        y_pos -= 0.04
    
    plt.tight_layout()
    plt.show()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close(fig)


def plot_correlation_summary(
    correlation_results: Dict,
    config: STNConfig = DEFAULT_CONFIG,
    baseline: str = 'c',
    save_path: Optional[Path] = None
):
    """Create visualization of correlation results"""
    
    bands = list(correlation_results.keys())
    n_bands = len(bands)
    n_cols = min(3, n_bands)
    n_rows = (n_bands + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    conditions = ['left_hand_c', 'right_hand_c']
    condition_labels = ['LH', 'RH']
    
    for idx, band in enumerate(bands):
        if idx >= len(axes):
            break
            
        ax = axes[idx]
        
        rhos = []
        ci_lows = []
        ci_highs = []
        agreements = []
        condition_names = []
        
        for cond in conditions:
            if cond in correlation_results[band]:
                res = correlation_results[band][cond]
                if res['n_subjects'] >= 3:
                    rhos.append(res['spearman_rho'])
                    ci_lows.append(res['ci_95_lower'])
                    ci_highs.append(res['ci_95_upper'])
                    agreements.append(res['directional_agreement_pct'])
                    condition_names.append(cond)
        
        x = np.arange(len(rhos))
        
        # Plot correlation coefficients with CI
        ax.errorbar(x, rhos, 
                   yerr=[np.array(rhos) - np.array(ci_lows), 
                         np.array(ci_highs) - np.array(rhos)],
                   fmt='o', color='darkblue', capsize=5, capthick=2,
                   markersize=10, markeredgecolor='white', markeredgewidth=2)
        
        # Add horizontal line at 0
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, linewidth=1)
        
        # Add agreement percentages as text
        for i, (rho, agree) in enumerate(zip(rhos, agreements)):
            ax.annotate(f'{agree:.0f}%', (x[i], rho + 0.1 if rho >= 0 else rho - 0.15),
                       ha='center', fontsize=9, fontweight='bold')
        
        ax.set_xticks(x)
        ax.set_xticklabels(condition_labels[:len(x)])
        ax.set_ylabel("Spearman's ρ")
        band_range = config.beta_bands[band]
        ax.set_title(f"{band_range[0]:.0f}-{band_range[1]:.0f} Hz")
        ax.set_ylim([-1.1, 1.1])
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add significance indicators
        for i, cond in enumerate(condition_names[:len(x)]):
            p_val = correlation_results[band][cond]['p_value']
            if p_val < 0.05:
                y_pos = rhos[i] + 0.15 if rhos[i] >= 0 else rhos[i] - 0.2
                ax.text(x[i], y_pos, '*' if p_val < 0.05 else '',
                       ha='center', fontsize=14, fontweight='bold')
    
    # Hide unused subplots
    for idx in range(len(bands), len(axes)):
        axes[idx].set_visible(False)
    
    baseline_name = config.baseline_names.get(baseline, baseline)
    plt.suptitle(f'Correlation Analysis: iEEG vs LCMV\n'
                f'Baseline: {baseline_name} | Error bars = 95% Bootstrap CI',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    plt.close(fig)


# =============================================================================
# HTML REPORT GENERATION
# =============================================================================

def generate_html_report(
    match_results: Dict,
    config: STNConfig = DEFAULT_CONFIG,
    analysis_type: str = 'voxel',
    analysis_name: str = 'Power Change',
    baseline: str = 'c',
    output_dir: Path = Path('.')
) -> Path:
    """Generate HTML report with results"""
    
    baseline_name = config.baseline_names.get(baseline, baseline)
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>STN Validation Report - {analysis_type.upper()} ({analysis_name})</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            h1 {{ color: #2c3e50; border-bottom: 2px solid #3498db; }}
            h2 {{ color: #34495e; }}
            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
            th {{ background-color: #3498db; color: white; padding: 10px; }}
            td {{ padding: 8px; border: 1px solid #ddd; }}
            tr:nth-child(even) {{ background-color: #f2f2f2; }}
            .baseline-section {{ margin: 30px 0; padding: 20px; border: 1px solid #3498db; border-radius: 5px; }}
            .config-info {{ background-color: #f8f9fa; padding: 15px; border-left: 4px solid #3498db; margin: 20px 0; font-family: monospace; }}
        </style>
    </head>
    <body>
        <h1>STN Validation Report: {analysis_type.upper()} Analysis ({analysis_name})</h1>
        <p><strong>Date:</strong> {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
        <p><strong>Valid subjects:</strong> {', '.join(config.valid_subjects)}</p>
        <p><strong>Baseline:</strong> {baseline_name}</p>
        
        <div class="config-info">
            <strong>Beta bands:</strong><br>
            {', '.join([f"{b}: {r[0]:.0f}-{r[1]:.0f}Hz" for b, r in config.beta_bands.items()])}
        </div>
        
        <div class="baseline-section">
            <h2>{baseline_name} Results</h2>
            <table>
                <tr><th>Band</th><th>Range</th><th>L_STN Match</th><th>L_STN Suppress</th><th>R_STN Match</th><th>R_STN Suppress</th></tr>
    """
    
    for band, (fmin, fmax) in config.beta_bands.items():
        row = [band, f"{fmin:.0f}-{fmax:.0f} Hz"]
        for region in ['L_STN', 'R_STN']:
            if region in match_results:
                band_data = [r for r in match_results[region] if r['band'] == band]
                if band_data:
                    matches = sum([(1 if r['lh_match'] else 0) + (1 if r['rh_match'] else 0) for r in band_data])
                    suppresses = sum([(1 if r['lh_suppress'] else 0) + (1 if r['rh_suppress'] else 0) for r in band_data])
                    total = len(band_data) * 2
                    row.append(f"{matches/total*100:.1f}% ({matches}/{total})")
                    row.append(f"{suppresses/total*100:.1f}% ({suppresses}/{total})")
                else:
                    row.extend(['0% (0/0)', '0% (0/0)'])
            else:
                row.extend(['0% (0/0)', '0% (0/0)'])
        
        html_content += "<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>"
    
    html_content += f"""
            </table>
        </div>
        
        <p><em>Report generated automatically by STN Validation Unified Library</em></p>
    </body>
    </html>
    """
    
    report_path = output_dir / f"validation_report_{analysis_type}_{analysis_name.replace(' ', '_')}.html"
    with open(report_path, 'w') as f:
        f.write(html_content)
    
    logger.info(f"📊 HTML report saved to {report_path}")
    return report_path


# =============================================================================
# MAIN ANALYSIS PIPELINES
# =============================================================================

def run_power_change_analysis(
    ieeg_path: Path,
    lcmv_path: Path,
    output_dir: Path,
    config: STNConfig = DEFAULT_CONFIG,
    analysis_type: AnalysisType = 'voxel',
    baselines: List[str] = ['c', 'o'],
    make_plots: bool = True,
    save_plots: bool = True,
    **kwargs
):
    """
    Run power change analysis (P_task - P_baseline)
    
    Args:
        ieeg_path: Path to consolidated_ieeg.npz
        lcmv_path: Path to consolidated_lcmv.npz
        output_dir: Directory to save outputs
        config: Configuration object
        analysis_type: 'atlas' or 'voxel'
        baselines: List of baselines to analyze
        make_plots: Whether to generate plots
        save_plots: Whether to save plots to disk
        **kwargs: Additional arguments for data loading
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    logger.info("="*60)
    logger.info("POWER CHANGE ANALYSIS")
    logger.info("="*60)
    ieeg_data, lcmv_data = load_consolidated_data(ieeg_path, lcmv_path)
    
    # Collect subject data
    collected_data = collect_all_data(
        ieeg_data, lcmv_data, analysis_type, config,
        align_to_zero=True,  # Power change analysis uses alignment
        **kwargs
    )
    
    if not collected_data:
        logger.error("No valid subjects found!")
        return
    
    # Compute power changes
    logger.info("\n" + "="*60)
    logger.info("COMPUTING POWER CHANGES")
    logger.info("="*60)
    
    computed_results = {}
    for subject, data in collected_data.items():
        logger.info(f"Processing {subject}...")
        computed_results[subject] = compute_power_changes(data, config)
    
    # Get match results for each baseline
    match_results = {}
    for baseline in baselines:
        match_results[baseline] = get_power_match_results(computed_results, baseline)
    
    # Generate plots
    if make_plots:
        logger.info("\n" + "="*60)
        logger.info("GENERATING PLOTS")
        logger.info("="*60)
        
        for subject, regions in computed_results.items():
            for region in regions.keys():
                logger.info(f"Plotting {subject} - {region}")
                
                # Power spectra
                if save_plots:
                    spec_path = output_dir / f"{subject}_{region}_{analysis_type}_spectra.png"
                else:
                    spec_path = None
                plot_power_spectra(
                    collected_data[subject][region], subject, region, config,
                    save_path=spec_path, show=False
                )
                
                # Subject summaries for each baseline
                for baseline in baselines:
                    if save_plots:
                        sum_path = output_dir / f"{subject}_{region}_{analysis_type}_{baseline}_summary.png"
                    else:
                        sum_path = None
                    plot_subject_summary(
                        regions, subject, region, config,
                        baseline=baseline, analysis_name='Power Change',
                        save_path=sum_path
                    )
        
        # Group summaries
        for baseline in baselines:
            if save_plots:
                group_path = output_dir / f"group_summary_{baseline}_{analysis_type}.png"
            else:
                group_path = None
            plot_group_summary(
                match_results[baseline], config,
                baseline=baseline, analysis_name='Power Change',
                save_path=group_path
            )
    
    # Generate HTML reports
    logger.info("\n" + "="*60)
    logger.info("GENERATING REPORTS")
    logger.info("="*60)
    
    for baseline in baselines:
        generate_html_report(
            match_results[baseline], config, analysis_type,
            analysis_name='Power_Change', baseline=baseline,
            output_dir=output_dir
        )
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("✅ ANALYSIS COMPLETE")
    logger.info("="*60)
    logger.info(f"Subjects analyzed: {len(collected_data)}")
    logger.info(f"Analysis type: {analysis_type}")
    logger.info(f"Output directory: {output_dir}")
    logger.info("="*60)
    
    return computed_results, match_results


def run_ersp_analysis(
    ieeg_path: Path,
    lcmv_path: Path,
    output_dir: Path,
    config: STNConfig = DEFAULT_CONFIG,
    analysis_type: AnalysisType = 'voxel',
    baselines: List[str] = ['c'],
    trim_seconds: float = 5.0,
    do_statistics: bool = True,
    make_plots: bool = True,
    save_plots: bool = True,
    **kwargs
):
    """
    Run ERSP analysis (10 * log10(P_task / P_baseline))
    
    Args:
        ieeg_path: Path to consolidated_ieeg.npz
        lcmv_path: Path to consolidated_lcmv.npz
        output_dir: Directory to save outputs
        config: Configuration object
        analysis_type: 'atlas' or 'voxel'
        baselines: List of baselines to analyze
        trim_seconds: Seconds to trim from start/end of trials
        do_statistics: Whether to perform statistical analysis
        make_plots: Whether to generate plots
        save_plots: Whether to save plots to disk
        **kwargs: Additional arguments for data loading
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    logger.info("="*60)
    logger.info("ERSP ANALYSIS")
    logger.info("="*60)
    logger.info(f"ERSP (dB) = 10 * log10(P_task / P_baseline)")
    logger.info(f"Edge trimming: removing {trim_seconds}s from each end")
    ieeg_data, lcmv_data = load_consolidated_data(ieeg_path, lcmv_path)
    
    # Collect subject data with trimming
    collected_data = collect_all_data(
        ieeg_data, lcmv_data, analysis_type, config,
        trim_seconds=trim_seconds,
        apply_filter=True,  # ERSP uses filtering
        align_to_zero=False,  # ERSP doesn't use alignment
        **kwargs
    )
    
    if not collected_data:
        logger.error("No valid subjects found!")
        return
    
    # Compute ERSP
    logger.info("\n" + "="*60)
    logger.info("COMPUTING ERSP")
    logger.info("="*60)
    
    computed_results = {}
    for subject, data in collected_data.items():
        logger.info(f"Processing {subject}...")
        computed_results[subject] = compute_ersp(data, config, baselines=baselines)
    
    # Get match results for each baseline
    match_results = {}
    for baseline in baselines:
        match_results[baseline] = get_ersp_match_results(computed_results, baseline)
    
    # Statistical analysis
    if do_statistics:
        logger.info("\n" + "="*60)
        logger.info("STATISTICAL ANALYSIS")
        logger.info("="*60)
        
        correlation_results = {}
        for baseline in baselines:
            correlation_results[baseline] = analyze_correlations(
                computed_results, config, baseline=baseline
            )
            print_correlation_summary(correlation_results[baseline], baseline)
            
            if make_plots and save_plots:
                corr_path = output_dir / f"correlation_summary_{baseline}_{analysis_type}.png"
                plot_correlation_summary(
                    correlation_results[baseline], config,
                    baseline=baseline, save_path=corr_path
                )
    
    # Generate plots
    if make_plots:
        logger.info("\n" + "="*60)
        logger.info("GENERATING PLOTS")
        logger.info("="*60)
        
        for subject, regions in computed_results.items():
            for region in regions.keys():
                logger.info(f"Plotting {subject} - {region}")
                
                # Power spectra
                if save_plots:
                    spec_path = output_dir / f"{subject}_{region}_{analysis_type}_spectra.png"
                else:
                    spec_path = None
                plot_power_spectra(
                    collected_data[subject][region], subject, region, config,
                    runs=[('c', 'Eyes Closed'), ('l', 'Left Hand'), ('r', 'Right Hand')],
                    save_path=spec_path, show=False
                )
                
                # Subject summaries for each baseline
                for baseline in baselines:
                    if save_plots:
                        sum_path = output_dir / f"{subject}_{region}_{analysis_type}_ersp_{baseline}_summary.png"
                    else:
                        sum_path = None
                    plot_subject_summary(
                        regions, subject, region, config,
                        baseline=baseline, analysis_name='ERSP',
                        save_path=sum_path
                    )
        
        # Group summaries
        for baseline in baselines:
            if save_plots:
                group_path = output_dir / f"group_ersp_summary_{baseline}_{analysis_type}.png"
            else:
                group_path = None
            title_extra = f"\n(Trimmed: first/last {trim_seconds}s)"
            plot_group_summary(
                match_results[baseline], config,
                baseline=baseline, analysis_name='ERSP',
                title_extra=title_extra, save_path=group_path
            )
    
    # Generate HTML reports
    logger.info("\n" + "="*60)
    logger.info("GENERATING REPORTS")
    logger.info("="*60)
    
    for baseline in baselines:
        generate_html_report(
            match_results[baseline], config, analysis_type,
            analysis_name='ERSP', baseline=baseline,
            output_dir=output_dir
        )
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("✅ ANALYSIS COMPLETE")
    logger.info("="*60)
    logger.info(f"Subjects analyzed: {len(collected_data)}")
    logger.info(f"Analysis type: {analysis_type}")
    logger.info(f"Edge trimming: removed {trim_seconds}s from each end")
    logger.info(f"Output directory: {output_dir}")
    logger.info("="*60)
    
    if do_statistics:
        return computed_results, match_results, correlation_results
    else:
        return computed_results, match_results


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_custom_config(
    beta_bands: Optional[Dict[str, Tuple[float, float]]] = None,
    valid_subjects: Optional[List[str]] = None,
    **kwargs
) -> STNConfig:
    """Create a custom configuration with specified parameters"""
    config = STNConfig()
    
    if beta_bands is not None:
        config.beta_bands = beta_bands
        # Auto-generate colors
        import matplotlib.cm as cm
        for i, band in enumerate(beta_bands.keys()):
            config.band_colors[band] = cm.tab10(i % 10)
    
    if valid_subjects is not None:
        config.valid_subjects = valid_subjects
    
    # Update any other attributes
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    return config


""""
from stn_validation import *
from pathlib import Path

# ========== ERSP ANALYSIS ==========
ersp_results, ersp_matches, ersp_corr = run_ersp_analysis(
    ieeg_path=Path('/mnt/movement/users/jaizor/xtra/derivatives/integrated/consolidated_ieeg.npz'),
    lcmv_path=Path('/mnt/movement/users/jaizor/xtra/derivatives/integrated/consolidated_lcmv.npz'),
    output_dir=Path('report_ersp'),
    analysis_type='voxel',
    baselines=['c'],
    trim_seconds=5.0,
    do_statistics=True,
    make_plots=True,
    save_plots=False
)

# Print correlation summary for Eyes Closed baseline
print_correlation_summary(ersp_corr['c'], baseline='c')

# ========== POWER CHANGE ANALYSIS ==========
power_results, power_matches = run_power_change_analysis(
    ieeg_path=Path('/mnt/movement/users/jaizor/xtra/derivatives/integrated/consolidated_ieeg.npz'),
    lcmv_path=Path('/mnt/movement/users/jaizor/xtra/derivatives/integrated/consolidated_lcmv.npz'),
    output_dir=Path('report_power'),
    analysis_type='voxel',
    baselines=['c', 'o'],
    make_plots=True,
    save_plots=False
)

# Run correlation on power change results
power_corr = analyze_correlations(power_results, baseline='c')
print_correlation_summary(power_corr, baseline='c')

"""