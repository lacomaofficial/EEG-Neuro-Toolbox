# =============================================================================
# imat.py v1.2 - A toolkit for exploring and analyzing PD_Ecog iEEG data
# =============================================================================
# PD_ECOG EXPLORATION TOOLKIT
# First-principles code for iEEG file inspection and analysis
#
# Changelog:
#   v1.0 - Initial release
#   v1.1 - Fixed: NaN interpolation for PSD, length mismatch handling
#   v1.2 - Fixed: Percept extraction with condition_keyword=None; added type hints
# =============================================================================

from scipy.io import loadmat
from scipy.signal import welch, coherence, correlate, detrend as scipy_detrend
from scipy.stats import pearsonr
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Dict, Union, List

# =============================================================================
# CONFIGURATION (Change once, use everywhere)
# =============================================================================

BASE_PATH = Path('/mnt/movement/users/jaizor/xtra/data/eeg/PD_Ecog')

# =============================================================================
# 0. HELPER FUNCTIONS (Internal utilities)
# =============================================================================

def _interpolate_nans(data: np.ndarray) -> np.ndarray:
    """
    Linear interpolation for NaNs. Returns copy with NaNs filled.
    Critical for PSD computation (Welch's method breaks with NaNs).
    
    Safe for <5% randomly distributed NaNs (validated on PD_Ecog data).
    """
    if not np.any(np.isnan(data)):
        return data
    
    clean_data = data.copy()
    nan_mask = np.isnan(clean_data)
    valid_idx = np.where(~nan_mask)[0]
    invalid_idx = np.where(nan_mask)[0]
    
    if len(valid_idx) > 1:
        clean_data[invalid_idx] = np.interp(invalid_idx, valid_idx, clean_data[valid_idx])
    else:
        # Fallback: fill with mean of valid points
        clean_data[nan_mask] = np.nanmean(clean_data)
    
    return clean_data


def _inspect_recursive(val, depth: int = 0, max_depth: int = 2):
    """Helper: recursive structure inspection."""
    indent = "  " * (depth + 1)
    
    if depth > max_depth:
        print(f"{indent}... (max depth)")
        return
    
    if hasattr(val, '_fieldnames'):
        for field in val._fieldnames[:10]:  # Limit output
            field_val = getattr(val, field)
            if hasattr(field_val, '_fieldnames'):
                print(f"{indent}├─ {field}: <struct ({len(field_val._fieldnames)} fields)>")
            elif isinstance(field_val, np.ndarray):
                if field_val.dtype.kind in ['f', 'i', 'u']:
                    nans = np.isnan(field_val).sum() if field_val.dtype.kind == 'f' else 0
                    print(f"{indent}├─ {field}: {field_val.shape}, NaNs={nans}")
    elif isinstance(val, np.ndarray):
        print(f"{indent}└─ {val.shape}, dtype={val.dtype}")


# =============================================================================
# 1. FILE INSPECTOR (What's inside this .mat file?)
# =============================================================================

def inspect_mat_file(filepath: Union[str, Path], max_depth: int = 2):
    """
    Open a .mat file and print its structure.
    Use this FIRST for any new file type.
    
    Parameters
    ----------
    filepath : str or Path
        Path to .mat file
    max_depth : int
        How many levels of nesting to explore (default: 2)
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        print(f"❌ FILE NOT FOUND: {filepath}")
        return None
    
    print(f"\n{'='*70}")
    print(f"FILE: {filepath.name}")
    print(f"{'='*70}")
    print(f"Size: {filepath.stat().st_size / 1024:.1f} KB")
    
    mat = loadmat(filepath, squeeze_me=True, struct_as_record=False)
    keys = [k for k in mat.keys() if not k.startswith('__')]
    
    print(f"\n📦 Top-level keys ({len(keys)}):")
    for key in keys:
        val = mat[key]
        if hasattr(val, '_fieldnames'):
            print(f"  • {key} → STRUCT ({len(val._fieldnames)} fields)")
        elif isinstance(val, np.ndarray):
            print(f"  • {key} → ARRAY {val.shape}")
    
    # Deep dive into first key (usually the data struct)
    if keys:
        print(f"\n🔍 Deep dive: [{keys[0]}]")
        print("-"*70)
        _inspect_recursive(mat[keys[0]], depth=0, max_depth=max_depth)
    
    return mat


# =============================================================================
# 2. DATA EXTRACTOR (Get the LFP data out)
# =============================================================================

def extract_percept_data(filepath: Union[str, Path], condition_keyword: Optional[str] = None) -> Dict:
    """
    Extract LFP data from Percept .mat file.
    Returns dict with one entry per matching condition.
    
    Parameters
    ----------
    filepath : str or Path
        Path to Percept .mat file
    condition_keyword : str or None
        If None, return ALL conditions. If string, filter by keyword (case-insensitive).
        Examples: 'bima', 'OFF', 'RHF', 'gait'
    
    Returns
    -------
    results : dict
        Dictionary mapping condition name → {data, sfreq, channel, timestamp, duration_sec}
    """
    mat = loadmat(filepath, squeeze_me=True, struct_as_record=False)
    
    # Find the data struct (name varies by file: Percept_JSON_parsed, Percept_JSON_parsed_Left, etc.)
    data_struct = None
    for key in mat.keys():
        if not key.startswith('__') and hasattr(mat[key], '_fieldnames'):
            if hasattr(mat[key], 'BrainSenseTimeDomain'):
                data_struct = mat[key]
                break
    
    if data_struct is None:
        print("❌ No BrainSenseTimeDomain found")
        return {}
    
    bst = data_struct.BrainSenseTimeDomain
    results = {}
    
    for i, seg in enumerate(bst):
        if hasattr(seg, 'Notes'):
            # FIX v1.2: Allow None to mean "return all conditions"
            if condition_keyword is None or condition_keyword.lower() in seg.Notes.lower():
                data = seg.TimeDomainData
                results[seg.Notes] = {
                    'data': data,
                    'sfreq': seg.SampleRateInHz,
                    'channel': seg.Channel,
                    'timestamp': getattr(seg, 'FirstPacketDateTime', 'N/A'),
                    'duration_sec': len(data) / seg.SampleRateInHz,
                }
    
    return results


def extract_rcs_data(filepath: Union[str, Path], detrend: bool = False) -> Dict:
    """
    Extract LFP data from RCS .mat file.
    Returns dict with all available channels.
    
    Parameters
    ----------
    filepath : str or Path
        Path to RCS .mat file
    detrend : bool
        If True, remove linear trends from all channels (recommended for PSD)
    
    Returns
    -------
    results : dict
        Dictionary with {sfreq, channels: {channel_name: data}}
    """
    mat = loadmat(filepath, squeeze_me=True, struct_as_record=False)
    
    if 'SDATA' not in mat:
        print("❌ No SDATA struct found")
        return {}
    
    sdata = mat['SDATA']
    
    # FIX: Access info safely (handle struct vs dict)
    info = sdata['info'] if isinstance(sdata, dict) else getattr(sdata, 'info', None)
    if info is None:
        print("⚠️ Could not access info struct")
        return {}
    
    # Extract sampling rate
    sfreq = None
    for hemisphere in ['rcs_dbs_left', 'rcs_dbs_right']:
        if hasattr(info, hemisphere):
            try:
                sfreq = float(getattr(info, hemisphere).sampling_rate)
                break
            except (AttributeError, TypeError, ValueError) as e:
                print(f"⚠️ Could not extract sfreq from {hemisphere}: {e}")
                continue
    
    if sfreq is None:
        sfreq = 250  # Default
        print("⚠️ Could not extract sampling rate. Using default 250 Hz")
    
    # Extract channels
    channel_map = {
        'STN-L': 'rcs_dbs_left',
        'STN-R': 'rcs_dbs_right',
        'ECOG-8-9-L': 'rcs_ecog_8_9_left',
        'ECOG-10-11-L': 'rcs_ecog_10_11_left',
        'ECOG-8-9-R': 'rcs_ecog_8_9_right',
        'ECOG-10-11-R': 'rcs_ecog_10_11_right',
    }
    
    results = {'sfreq': sfreq, 'channels': {}}
    
    for ch_name, key in channel_map.items():
        try:
            data = getattr(sdata, key).astype(np.float64)
            if data.ndim == 2 and data.shape[1] == 1:
                data = data.squeeze()
            results['channels'][ch_name] = data
        except (AttributeError, KeyError, TypeError) as e:
            # Channel not present (unilateral case) or access error
            continue
    
    # 🔧 FIX 1: Handle length mismatches (RCS wireless dropout)
    if len(results['channels']) > 1:
        lengths = [ch.shape[0] for ch in results['channels'].values()]
        min_len = min(lengths)
        
        if len(set(lengths)) > 1:
            print(f"⚠️ Length mismatch detected. Truncating to {min_len} samples.")
            for ch in results['channels'].keys():
                results['channels'][ch] = results['channels'][ch][:min_len]
    
    # 🔧 FIX 2: Optional detrending (removes linear drifts)
    if detrend:
        print("ℹ Detrending channels (removing linear trends)...")
        for ch in results['channels'].keys():
            results['channels'][ch] = scipy_detrend(results['channels'][ch])
    
    return results


# =============================================================================
# 3. VISUALIZATION (See the data)
# =============================================================================

def plot_raw_trace(data: np.ndarray, sfreq: float, title: str = 'Raw Trace', 
                   duration: float = 10, ch_names: Optional[List[str]] = None):
    """
    Plot raw LFP trace(s).
    
    Parameters
    ----------
    data : np.ndarray
        Signal data (n_channels × n_samples)
    sfreq : float
        Sampling frequency in Hz
    title : str
        Plot title
    duration : float
        How many seconds to display (default: 10)
    ch_names : list, optional
        Channel names (default: ['CH0', 'CH1', ...])
    """
    if not isinstance(data, np.ndarray):
        raise TypeError(f"data must be np.ndarray, got {type(data)}")
    if data.size == 0:
        raise ValueError("data array is empty")
    
    if data.ndim == 1:
        data = data[np.newaxis, :]
    
    n_ch = data.shape[0]
    if ch_names is None:
        ch_names = [f'CH{i}' for i in range(n_ch)]
    
    times = np.arange(data.shape[1]) / sfreq
    max_samples = int(duration * sfreq)
    
    plt.figure(figsize=(14, 2*n_ch))
    for i in range(n_ch):
        plt.subplot(n_ch, 1, i+1)
        plt.plot(times[:max_samples], data[i, :max_samples], linewidth=0.5)
        plt.ylabel(f'{ch_names[i]}\n(μV)')
        plt.grid(True, alpha=0.3)
        if i == 0:
            plt.title(f'{title} (First {duration}s)')
    plt.xlabel('Time (s)')
    plt.tight_layout()
    plt.show()


def plot_psd(data: np.ndarray, sfreq: float, title: str = 'Power Spectral Density', 
             ch_names: Optional[List[str]] = None, fmax: float = 50, 
             interpolate_nans: bool = True):
    """
    Plot PSD with beta band highlighted.
    
    Parameters
    ----------
    data : np.ndarray
        Signal data (n_channels × n_samples)
    sfreq : float
        Sampling frequency in Hz
    title : str
        Plot title
    ch_names : list, optional
        Channel names (default: ['CH0', 'CH1', ...])
    fmax : float
        Maximum frequency to display (default: 50 Hz)
    interpolate_nans : bool
        If True, interpolate NaNs before PSD computation (default: True)
    """
    if not isinstance(data, np.ndarray):
        raise TypeError(f"data must be np.ndarray, got {type(data)}")
    if data.size == 0:
        raise ValueError("data array is empty")
    
    if data.ndim == 1:
        data = data[np.newaxis, :]
    
    n_ch = data.shape[0]
    if ch_names is None:
        ch_names = [f'CH{i}' for i in range(n_ch)]
    
    plt.figure(figsize=(10, 5))
    for i in range(n_ch):
        # 🔧 FIX: Interpolate NaNs before PSD (Welch's method breaks with NaNs)
        ch_data = data[i]
        if interpolate_nans and np.any(np.isnan(ch_data)):
            ch_data = _interpolate_nans(ch_data)
        
        freqs, psd = welch(ch_data, sfreq, nperseg=sfreq*2, noverlap=sfreq)
        plt.semilogy(freqs, psd, label=ch_names[i], alpha=0.7)
    
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power (μV²/Hz)')
    plt.title(title)
    plt.axvspan(13, 30, alpha=0.3, color='red', label='Beta (13-30 Hz)')
    plt.xlim(0, fmax)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_coherence(data1: np.ndarray, data2: np.ndarray, sfreq: float, 
                   title: str = 'Inter-channel Coherence'):
    """
    Plot coherence between two signals.
    
    Parameters
    ----------
    data1, data2 : np.ndarray
        Two signals (must be same length)
    sfreq : float
        Sampling frequency in Hz
    title : str
        Plot title
    """
    if not isinstance(data1, np.ndarray) or not isinstance(data2, np.ndarray):
        raise TypeError("data1 and data2 must be np.ndarray")
    if len(data1) != len(data2):
        raise ValueError("data1 and data2 must have same length")
    
    # Interpolate NaNs if present
    if np.any(np.isnan(data1)):
        data1 = _interpolate_nans(data1)
    if np.any(np.isnan(data2)):
        data2 = _interpolate_nans(data2)
    
    freqs, coh = coherence(data1, data2, sfreq, nperseg=sfreq*2)
    
    plt.figure(figsize=(10, 4))
    plt.plot(freqs, coh, linewidth=1.5, color='purple')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Coherence')
    plt.title(title)
    plt.axvspan(13, 30, alpha=0.3, color='red', label='Beta band')
    plt.xlim(0, 50)
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Print beta coherence
    beta_mask = (freqs >= 13) & (freqs <= 30)
    print(f"📊 Beta coherence: {np.mean(coh[beta_mask]):.3f}")


# =============================================================================
# 4. QUALITY METRICS (Is this data usable?)
# =============================================================================

def compute_quality_metrics(data: np.ndarray, sfreq: float) -> Dict:
    """
    Compute basic quality metrics for LFP data.
    
    Parameters
    ----------
    data : np.ndarray
        Signal data (n_channels × n_samples)
    sfreq : float
        Sampling frequency in Hz
    
    Returns
    -------
    metrics : dict
        Dictionary with quality metrics per channel
    """
    if not isinstance(data, np.ndarray):
        raise TypeError(f"data must be np.ndarray, got {type(data)}")
    
    if data.ndim == 1:
        data = data[np.newaxis, :]
    
    metrics = {}
    for i in range(data.shape[0]):
        ch_data = data[i]
        
        # Interpolate NaNs for PSD
        if np.any(np.isnan(ch_data)):
            ch_data_clean = _interpolate_nans(ch_data)
        else:
            ch_data_clean = ch_data
        
        freqs, psd = welch(ch_data_clean, sfreq, nperseg=sfreq*2)
        
        beta_mask = (freqs >= 13) & (freqs <= 30)
        theta_mask = (freqs >= 4) & (freqs <= 8)
        
        metrics[f'ch{i}'] = {
            'duration_sec': len(ch_data) / sfreq,
            'nan_pct': 100 * np.isnan(ch_data).sum() / len(ch_data) if len(ch_data) > 0 else 100,
            'std_uv': np.nanstd(ch_data),
            'beta_power': np.mean(psd[beta_mask]),
            'theta_power': np.mean(psd[theta_mask]),
            'beta_theta_ratio': np.mean(psd[beta_mask]) / np.mean(psd[theta_mask]) if np.mean(psd[theta_mask]) > 0 else np.nan,
        }
    
    return metrics


# =============================================================================
# 5. QUICK WORKFLOW EXAMPLES
# =============================================================================

def example_inspect_percept():
    """Example: Inspect a Percept file."""
    filepath = BASE_PATH / 'sbj11_PM11_Percept' / 'Percept_JSON_parsed_Left.mat'
    inspect_mat_file(filepath)


def example_extract_and_plot():
    """Example: Extract bima data and plot."""
    filepath = BASE_PATH / 'sbj11_PM11_Percept' / 'Percept_JSON_parsed_Left.mat'
    data_dict = extract_percept_data(filepath, condition_keyword='bima')
    
    for condition, info in data_dict.items():
        print(f"\n{condition}:")
        print(f"  Duration: {info['duration_sec']:.1f}s")
        print(f"  Channel: {info['channel']}")
        
        plot_raw_trace(info['data'], info['sfreq'], title=condition, ch_names=[info['channel']])
        plot_psd(info['data'], info['sfreq'], title=f'PSD - {condition}', ch_names=[info['channel']])


def example_bilateral_analysis():
    """Example: Merge and analyze bilateral Percept data."""
    left_file = BASE_PATH / 'sbj11_PM11_Percept' / 'Percept_JSON_parsed_Left.mat'
    right_file = BASE_PATH / 'sbj11_PM11_Percept' / 'Percept_JSON_parsed_Right.mat'
    
    left_data = extract_percept_data(left_file, 'bima')
    right_data = extract_percept_data(right_file, 'bima')
    
    # Get first (usually only) condition from each
    left_cond = list(left_data.keys())[0]
    right_cond = list(right_data.keys())[0]
    
    left_sig = left_data[left_cond]['data']
    right_sig = right_data[right_cond]['data']
    sfreq = left_data[left_cond]['sfreq']
    
    # Truncate to same length
    min_len = min(len(left_sig), len(right_sig))
    merged = np.vstack([left_sig[:min_len], right_sig[:min_len]])
    
    print(f"Merged shape: {merged.shape}")
    print(f"Timestamps: {left_data[left_cond]['timestamp']} vs {right_data[right_cond]['timestamp']}")
    
    plot_raw_trace(merged, sfreq, title='Bilateral STN', ch_names=['STN-L', 'STN-R'])
    plot_psd(merged, sfreq, title='Bilateral PSD', ch_names=['STN-L', 'STN-R'])
    plot_coherence(merged[0], merged[1], sfreq, title='STN-L ↔ STN-R Coherence')


def example_rcs_extraction():
    """Example: Extract and plot RCS data (sbj1)."""
    filepath = BASE_PATH / 'sbj1_PM1_RCS' / 'RCS_DBSOFF_bima.mat'
    
    print(f"\n🔍 Extracting: {filepath.name}")
    rcs_data = extract_rcs_data(filepath, detrend=False)
    
    if rcs_data['channels']:
        print(f"\n✓ Extracted {len(rcs_data['channels'])} channels")
        print(f"✓ Sampling rate: {rcs_data['sfreq']} Hz")
        
        # Stack all available channels
        ch_names = list(rcs_data['channels'].keys())
        data_stack = np.vstack([rcs_data['channels'][ch] for ch in ch_names])
        
        # Plot
        plot_raw_trace(data_stack, rcs_data['sfreq'], 
                      title=f'sbj1 - RCS DBS OFF', 
                      ch_names=ch_names)
        
        plot_psd(data_stack, rcs_data['sfreq'], 
                title=f'sbj1 - PSD', 
                ch_names=ch_names,
                interpolate_nans=True)  # 🔧 Critical for RCS data


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__version__ = "1.2"

__all__ = [
    '__version__',
    'BASE_PATH',
    'inspect_mat_file',
    'extract_percept_data',
    'extract_rcs_data',
    'plot_raw_trace',
    'plot_psd',
    'plot_coherence',
    'compute_quality_metrics',
    'example_inspect_percept',
    'example_extract_and_plot',
    'example_bilateral_analysis',
    'example_rcs_extraction',
]


# =============================================================================
# MAIN EXECUTION (When run as script)
# =============================================================================

if __name__ == '__main__':
    print("\n" + "="*70)
    print(f"PD_ECOG EXPLORATION TOOLKIT v{__version__}")
    print("="*70)
    print("\nAvailable functions:")
    print("  • inspect_mat_file(filepath)")
    print("  • extract_percept_data(filepath, condition_keyword=None)")
    print("  • extract_rcs_data(filepath, detrend=False)")
    print("  • plot_raw_trace(data, sfreq, ...)")
    print("  • plot_psd(data, sfreq, ..., interpolate_nans=True)")
    print("  • plot_coherence(data1, data2, sfreq)")
    print("  • compute_quality_metrics(data, sfreq)")
    print("\nExample workflows:")
    print("  • example_inspect_percept()")
    print("  • example_extract_and_plot()")
    print("  • example_bilateral_analysis()")
    print("  • example_rcs_extraction()")
    print("\n" + "="*70)
    print("Tip: Import this module in Jupyter and override BASE_PATH if needed:")
    print("  import imat")
    print("  imat.BASE_PATH = Path('/your/custom/path')")
    print("="*70 + "\n")