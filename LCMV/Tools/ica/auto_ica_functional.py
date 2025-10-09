# auto_ica_functional.py - Functional version
# Standard library
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional, Union

# Third-party scientific stack
import numpy as np
from scipy.stats import median_abs_deviation, kurtosis
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# MNE and related
import mne
from mne import io
from mne.io import constants
from mne_icalabel import label_components

# Configure MNE
mne.set_log_level('WARNING')

# ============================================================================
# CONSTANTS
# ============================================================================

CHANNEL_RENAME_MAP = {**{str(i): f'E{i}' for i in range(1, 281)}, 'REF CZ': 'Cz'}
PROTECTED_CHANNELS = {'E31', 'E19', 'E41', 'E274', 'E227', 'E229', 'E280', 'E52'}

# Artifact detection channels
VVEOG = ('E31', 'E19')
HEOG = ('E41', 'E274')
ECG = ('E227', 'E229')
EMG_CHS = ['E280', 'E52']
FRONTAL_CHS = ['E31', 'E19']

# ICLabel thresholds
ICALABEL_THRESHOLDS = {
    'eye blink': 0.80,
    'heart beat': 0.80,
    'muscle artifact': 0.75,
    'line noise': 0.80,
    'channel noise': 0.80
}


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def setup_logging(subject: str, output_path: Path, log_to_file: bool = True) -> Path:
    """Setup logging and return log file path."""
    log_file = output_path / f"{subject}_preproc_log.txt"
    if log_to_file:
        log(f"Initialized preprocessing for {subject}", log_file, log_to_file)
    return log_file


def log(msg: str, log_file: Path, log_to_file: bool = True, detail: str = "normal"):
    """Log message to file and optionally console."""
    if log_to_file:
        with open(log_file, 'a') as f:
            f.write(f"{msg}\n")
    if detail == "normal":
        print(msg)


def parse_gpsc(filepath: Path) -> List[Tuple[str, float, float, float]]:
    """Parse GPSC file efficiently."""
    channels = []
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                try:
                    name = parts[0]
                    x, y, z = map(float, parts[1:4])
                    channels.append((name, x, y, z))
                except ValueError:
                    continue
    return channels


# ============================================================================
# DATA LOADING AND PREPARATION
# ============================================================================

def load_raw_data(input_path: Path, input_format: str, log_file: Path, 
                  log_to_file: bool = True) -> mne.io.Raw:
    """Load raw data from MFF or FIF format."""
    if input_format == "mff":
        log("Loading raw data from .mff...", log_file, log_to_file)
        raw = mne.io.read_raw_egi(str(input_path), preload=True)
    elif input_format == "fif":
        log(f"Loading raw data from .fif: {input_path}", log_file, log_to_file)
        if not input_path.is_file() or input_path.suffix != '.fif':
            raise ValueError(f"Invalid .fif file: {input_path}")
        raw = mne.io.read_raw_fif(str(input_path), preload=True)
    else:
        raise ValueError("input_format must be 'mff' or 'fif'")
    
    return raw


def apply_channel_renaming(raw: mne.io.Raw, log_file: Path, 
                           log_to_file: bool = True) -> mne.io.Raw:
    """Apply channel renaming."""
    log("Applying channel renaming...", log_file, log_to_file)
    
    existing_map = {old: new for old, new in CHANNEL_RENAME_MAP.items() 
                   if old in raw.ch_names}
    if existing_map:
        raw.rename_channels(existing_map)
        log(f"Renamed {len(existing_map)} channels.", log_file, log_to_file)
    
    return raw


def apply_montage(raw: mne.io.Raw, gpsc_file: Path, log_file: Path,
                  log_to_file: bool = True) -> mne.io.Raw:
    """Apply GPS montage from GPSC file."""
    channels = parse_gpsc(gpsc_file)
    if not channels:
        raise ValueError("No valid channels in .gpsc file")
    
    # Normalize positions
    gpsc_array = np.array([ch[1:4] for ch in channels])
    mean_pos = gpsc_array.mean(axis=0)
    log(f"Original mean position (mm): {mean_pos}", log_file, log_to_file)
    
    ch_pos = {
        ch[0]: np.array([ch[1] - mean_pos[0], ch[2] - mean_pos[1], ch[3] - mean_pos[2]]) / 1000.0
        for ch in channels
    }
    
    # Create montage with fiducials
    montage = mne.channels.make_dig_montage(
        ch_pos=ch_pos,
        nasion=ch_pos.get('FidNz'),
        lpa=ch_pos.get('FidT9'),
        rpa=ch_pos.get('FidT10'),
        coord_frame='head'
    )
    raw.set_montage(montage, on_missing='warn')
    log("Montage applied.", log_file, log_to_file)
    
    return raw


# ============================================================================
# FILTERING FUNCTIONS
# ============================================================================

def apply_highpass_filter(raw: mne.io.Raw, l_freq: float, log_file: Path,
                          log_to_file: bool = True) -> mne.io.Raw:
    """Apply highpass filter."""
    log(f"Applying highpass filter at {l_freq} Hz...", log_file, log_to_file)
    return raw.copy().filter(
        l_freq=l_freq, h_freq=None, picks=['eeg'],
        method='fir', phase='zero', fir_window='hamming',
        fir_design='firwin', n_jobs=-1
    )


def apply_lowpass_filter(raw: mne.io.Raw, h_freq: float, log_file: Path,
                         log_to_file: bool = True) -> mne.io.Raw:
    """Apply lowpass filter."""
    log(f"Applying lowpass filter at {h_freq} Hz...", log_file, log_to_file)
    return raw.copy().filter(
        l_freq=None, h_freq=h_freq, picks=['eeg'],
        method='fir', phase='zero', fir_window='hamming',
        fir_design='firwin', n_jobs=-1
    )



def apply_notch_filter(raw: mne.io.Raw, line_freq: float, log_file: Path,
                       log_to_file: bool = True, max_freq: float = 100.0) -> mne.io.Raw:
    """Apply notch filter up to max_freq (default: 100 Hz)."""
    nyquist = raw.info["sfreq"] / 2
    upper = min(nyquist, max_freq)
    notch_freqs = np.arange(line_freq, upper + line_freq, line_freq)
    notch_freqs = notch_freqs[notch_freqs < upper]

    if len(notch_freqs) > 0:
        log(f"Applying notch filter at: {notch_freqs}", log_file, log_to_file)
        return raw.copy().notch_filter(
            freqs=notch_freqs, picks='eeg', method='spectrum_fit',
            filter_length='auto', mt_bandwidth=1.0, p_value=0.05
        )
    return raw

def filter_data(raw: mne.io.Raw, apply_highpass: bool, apply_lowpass: bool,
                apply_notch: bool, l_freq: float, h_freq: float, 
                line_freq: float, log_file: Path, log_to_file: bool = True) -> mne.io.Raw:
    """Apply all selected filters in sequence."""
    log("Applying filters...", log_file, log_to_file)
    
    filtered_raw = raw.copy()
    
    if apply_highpass and l_freq is not None:
        filtered_raw = apply_highpass_filter(filtered_raw, l_freq, log_file, log_to_file)
    
    if apply_lowpass and h_freq is not None:
        filtered_raw = apply_lowpass_filter(filtered_raw, h_freq, log_file, log_to_file)
    
    if apply_notch:
        filtered_raw = apply_notch_filter(filtered_raw, line_freq, log_file, log_to_file)
    
    if not (apply_highpass or apply_lowpass or apply_notch):
        log("No filters applied (all filter types disabled).", log_file, log_to_file)
    
    # Check Cz for flat signal
    if 'Cz' in filtered_raw.ch_names:
        if np.std(filtered_raw.get_data(picks=['Cz'])[0]) < 1e-6:
            filtered_raw.info['bads'].append('Cz')
            log("Marked Cz as bad (flat signal).", log_file, log_to_file)
    
    return filtered_raw


# ============================================================================
# BAD CHANNEL DETECTION
# ============================================================================

def detect_bad_channels(raw: mne.io.Raw, subject: str, output_path: Path,
                        plot: bool, log_file: Path, log_to_file: bool = True,
                        mad_threshold: float = 5.0, 
                        min_amplitude_uv: float = 0.1,
                        protected_channels: Optional[set] = None) -> mne.io.Raw:
    """
    Detect bad channels using MAD-based outlier detection.

    Args:
        raw: The raw EEG data.
        subject: Subject identifier for logging and file naming.
        output_path: Path to save output files and plots.
        plot: Whether to generate and save diagnostic plots.
        log_file: Path to the log file.
        log_to_file: Whether to write logs to the file.
        mad_threshold: Threshold for MAD-based outlier detection.
        min_amplitude_uv: Minimum amplitude threshold for flat channel detection (µV).
        protected_channels: Optional set of channels to protect from being marked as bad.
                            If None, defaults to the global PROTECTED_CHANNELS constant.

    Returns:
        The input `raw` object with updated `info['bads']`.
    """
    # Determine which channels to protect
    if protected_channels is None:
        protected_chs_to_use = PROTECTED_CHANNELS
    else:
        protected_chs_to_use = protected_channels

    log(f"Detecting bad channels (flat < {min_amplitude_uv} µV, noisy Z > {mad_threshold})...",
        log_file, log_to_file)

    raw_eeg = raw.copy().pick_types(eeg=True)
    available_chs = set(raw_eeg.ch_names)
    # Use the potentially overridden protected channels set
    protected_chs = protected_chs_to_use & available_chs
    # Channels to analyze are those not in the protected set
    eeg_chs = [ch for ch in raw_eeg.ch_names if ch not in protected_chs]

    if not eeg_chs:
        log("No EEG channels available for detection.", log_file, log_to_file)
        return raw

    # Get data in µV
    raw_for_detection = raw_eeg.copy().pick(eeg_chs)
    data_uv = np.nan_to_num(raw_for_detection.get_data() * 1e6, nan=0.0, posinf=0.0, neginf=0.0)

    # Compute features
    variance = np.var(data_uv, axis=1)
    amplitude = np.ptp(data_uv, axis=1)

    # Detect flat channels
    flat_mask = amplitude < min_amplitude_uv
    flat_channels = [raw_for_detection.ch_names[i] for i in np.where(flat_mask)[0]]

    # Detect noisy channels using MAD
    noisy_mask = np.zeros(len(amplitude), dtype=bool)
    for feat in [variance, amplitude]:
        mad = median_abs_deviation(feat, scale='normal', nan_policy='omit')
        if not np.isnan(mad) and mad > 1e-12:
            z_scores = (feat - np.nanmedian(feat)) / mad
            noisy_mask |= (z_scores > mad_threshold)

    noisy_channels = [raw_for_detection.ch_names[i] for i in np.where(noisy_mask)[0]]

    # Combine and update bad channels
    detected_bads = sorted(set(flat_channels + noisy_channels))
    current_bads = set(raw.info['bads'])
    new_bads = [ch for ch in detected_bads if ch not in current_bads]
    raw.info['bads'] = sorted(current_bads | set(detected_bads))

    # Log results
    if protected_chs:
        log(f"Protected: {sorted(protected_chs)}", log_file, log_to_file)
    log(f"Detected bad channels: {detected_bads}", log_file, log_to_file)

    # Save plots if enabled
    if plot and detected_bads:
        try:
            montage = raw.get_montage()
            if montage is None:
                log("⚠️ No montage found — skipping topomap.", log_file, log_to_file)
            else:
                ch_pos = {}
                for d in montage.dig:
                    if d['kind'] == mne.io.constants.FIFF.FIFFV_POINT_EEG:
                        ch_name = montage.ch_names[d['ident'] - 1]
                        ch_pos[ch_name] = d['r']

                plot_bad_channels_topomap(ch_pos, detected_bads, subject, 
                                         output_path, log_file, log_to_file)
                plot_bad_channels_timeseries(raw, detected_bads, subject,
                                            output_path, log_file, log_to_file)

        except Exception as e:
            log(f"⚠️ Failed to generate/save bad channel plots: {e}", log_file, log_to_file)

    return raw


def plot_bad_channels_topomap(ch_pos: Dict[str, np.ndarray], bad_channels: List[str],
                               subject: str, output_path: Path, log_file: Path,
                               log_to_file: bool = True):
    """Plot and save topomap of bad channels."""
    if not bad_channels:
        return

    valid_bads = [ch for ch in bad_channels if ch in ch_pos]
    if not valid_bads:
        log("⚠️ No bad channels with valid positions.", log_file, log_to_file)
        return

    # Unique colors
    cmap = plt.get_cmap('viridis')
    colors = cmap(np.linspace(0, 1, len(valid_bads)))
    color_dict = {ch: colors[i] for i, ch in enumerate(valid_bads)}

    # Get and scale positions
    pos = np.array([ch_pos[ch][:2] for ch in valid_bads])
    max_radius = np.max(np.sqrt(np.sum(pos**2, axis=1)))
    pos_scaled = (pos / max_radius * 0.1 if max_radius > 0 else pos)
    pos_scaled[:, 1] -= 0.02

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, (x, y) in enumerate(pos_scaled):
        ch = valid_bads[i]
        color = color_dict[ch]
        ax.plot(x, y, 's', markersize=15, color=color, alpha=1.0)
        ax.text(x, y + 0.01, ch, fontsize=10, ha='center', va='bottom', color=color)

    # Add head outline
    try:
        mne.viz.plot_topomap(np.zeros(len(pos_scaled)), pos_scaled, axes=ax,
                            show=False, sphere=0.1, outlines='head')
    except Exception as e:
        log(f"Topomap background failed: {e}", log_file, log_to_file, detail="debug")

    ax.set_title('Detected Bad Channels (Topomap)', fontsize=12, pad=20)
    ax.set_xlim(-0.12, 0.12)
    ax.set_ylim(-0.12, 0.12)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = output_path / "plots" / f"{subject}_bad_channels_topomap.png"
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    log(f"🖼️ Bad channels topomap saved: {fig_path}", log_file, log_to_file)


def plot_bad_channels_timeseries(raw: mne.io.Raw, bad_channels: List[str],
                                  subject: str, output_path: Path, log_file: Path,
                                  log_to_file: bool = True):
    """Plot full-duration time series of bad channels."""
    if not bad_channels:
        return

    valid_bads = [ch for ch in bad_channels if ch in raw.ch_names]
    if not valid_bads:
        return

    # Get full data
    data, times = raw[valid_bads, :]

    # Unique colors
    cmap = plt.get_cmap('viridis')
    colors = cmap(np.linspace(0, 1, len(valid_bads)))
    color_dict = {ch: colors[i] for i, ch in enumerate(valid_bads)}

    # Plot
    n_ch = len(valid_bads)
    fig_height = min(2.2 * n_ch, 40)
    fig, axes = plt.subplots(n_ch, 1, figsize=(14, fig_height), sharex=True)
    if n_ch == 1:
        axes = [axes]

    for i, (ch, ax) in enumerate(zip(valid_bads, axes)):
        color = color_dict[ch]
        ax.plot(times, data[i, :] * 1e6, color=color, linewidth=1)
        ax.set_ylabel(f'{ch}\n(µV)', fontsize=10)
        ax.grid(True, alpha=0.3)

        mean_val = np.mean(data[i, :]) * 1e6
        std_val = np.std(data[i, :]) * 1e6
        ax.text(0.02, 0.98, f'μ={mean_val:.1f}, σ={std_val:.1f}',
                transform=ax.transAxes, va='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    axes[-1].set_xlabel('Time (s)', fontsize=12)
    total_duration = times[-1]
    plt.suptitle(f'Bad Channels — Full Time Series ({total_duration:.1f}s)',
                fontsize=14, weight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    fig_path = output_path / "plots" / f"{subject}_bad_channels_timeseries.png"
    fig.savefig(fig_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    log(f"📈 Bad channels full time series saved: {fig_path}", log_file, log_to_file)


# ============================================================================
# ICA FUNCTIONS
# ============================================================================

def create_bipolar_channels(raw: mne.io.Raw, log_file: Path,
                            log_to_file: bool = True,
                            use_artifact_detection_channels: bool = True) -> mne.io.Raw:
    """Create bipolar reference channels for artifact detection."""
    if not use_artifact_detection_channels:
        log("Skipping creation of bipolar channels (artifact detection disabled).", log_file, log_to_file)
        return raw

    bipolar_specs = [
        (VVEOG, 'vVEOG', 'eog', "blink detection"),
        (HEOG, 'BLINK_H', 'eog', "horizontal eye movement"),
        (ECG, 'ECG_BIO', 'ecg', "cardiac artifact")
    ]
    
    for (anode, cathode), name, ch_type, desc in bipolar_specs:
        if anode in raw.ch_names and cathode in raw.ch_names:
            raw = mne.set_bipolar_reference(
                raw, anode=anode, cathode=cathode,
                ch_name=name, drop_refs=False
            ).set_channel_types({name: ch_type})
            log(f"Created {name} ({anode}-{cathode}) for {desc}", 
                log_file, log_to_file, detail="debug")
        else:
            log(f"Skipping {name} - channels {anode}, {cathode} not found.", log_file, log_to_file, detail="debug")
    
    return raw

def detect_artifact_components(ica: mne.preprocessing.ICA, raw: mne.io.Raw,
                               log_file: Path, log_to_file: bool = True,
                               use_artifact_detection_channels: bool = True) -> Dict[str, List[int]]:
    """Detect artifact components using multiple methods."""
    results = {
        'blink': [], 'horizontal': [], 'ecg': [],
        'muscle': [], 'frontal_lf': [], 'line_noise': [],
        'icalabel': [], 'extreme': []
    }

    if use_artifact_detection_channels:
        # Blink detection
        if 'vVEOG' in raw.ch_names:
            try:
                idx, _ = ica.find_bads_eog(raw, ch_name='vVEOG', measure='zscore', threshold=3.0)
                results['blink'] = [int(i) for i in idx]
                log(f"Blink: {results['blink']}", log_file, log_to_file, detail="debug")
            except Exception as e:
                log(f"Blink detection failed: {e}", log_file, log_to_file, detail="debug")
        else:
            log("vVEOG channel not found, skipping blink detection.", log_file, log_to_file, detail="debug")

        # Horizontal eye movement
        if 'BLINK_H' in raw.ch_names:
            try:
                idx, _ = ica.find_bads_eog(raw, ch_name='BLINK_H', measure='zscore', threshold=3.0)
                results['horizontal'] = [int(i) for i in idx]
                log(f"Horizontal: {results['horizontal']}", log_file, log_to_file, detail="debug")
            except Exception as e:
                log(f"Horizontal detection failed: {e}", log_file, log_to_file, detail="debug")
        else:
            log("BLINK_H channel not found, skipping horizontal eye movement detection.", log_file, log_to_file, detail="debug")

        # ECG detection
        if 'ECG_BIO' in raw.ch_names:
            try:
                idx, _ = ica.find_bads_ecg(raw, ch_name='ECG_BIO', method='correlation', 
                                          measure='zscore', threshold=3.0)
                results['ecg'] = [int(i) for i in idx]
                log(f"ECG: {results['ecg']}", log_file, log_to_file, detail="debug")
            except Exception as e:
                log(f"ECG detection failed: {e}", log_file, log_to_file, detail="debug")
        else:
            log("ECG_BIO channel not found, skipping ECG detection.", log_file, log_to_file, detail="debug")

        # Muscle artifacts
        for ch in EMG_CHS:
            if ch in raw.ch_names:
                try:
                    idx, _ = ica.find_bads_eog(raw, ch_name=ch, measure='zscore',
                                              l_freq=30, h_freq=100, threshold=3.0)
                    results['muscle'].extend([int(i) for i in idx])
                except Exception as e:
                    log(f"EMG detection failed for {ch}: {e}", log_file, log_to_file, detail="debug")
            else:
                log(f"EMG channel {ch} not found, skipping.", log_file, log_to_file, detail="debug")
        results['muscle'] = list(set(results['muscle']))

        # Frontal low-frequency artifacts
        for ch in FRONTAL_CHS:
            if ch in raw.ch_names:
                try:
                    idx, _ = ica.find_bads_eog(raw, ch_name=ch, measure='zscore',
                                              l_freq=1.0, h_freq=10.0, threshold=3.5)
                    results['frontal_lf'].extend([int(i) for i in idx])
                except Exception as e:
                    log(f"Frontal LF failed for {ch}: {e}", log_file, log_to_file, detail="debug")
            else:
                log(f"Frontal channel {ch} not found, skipping.", log_file, log_to_file, detail="debug")
        results['frontal_lf'] = list(set(results['frontal_lf']))

    # Line noise detection (also conditional on the flag)
    if use_artifact_detection_channels:
        try:
            sfreq = raw.info['sfreq']
            src_data = ica.get_sources(raw).get_data()
            for i in range(ica.n_components_):
                psd, freqs = mne.time_frequency.psd_array_welch(
                    src_data[i], sfreq=sfreq, fmin=1, fmax=100, verbose=False
                )
                line_band = (freqs >= 58) & (freqs <= 62)
                ref_band = (freqs >= 1) & (freqs <= 100)
                flank_band = ((freqs >= 50) & (freqs < 58)) | ((freqs > 62) & (freqs <= 70))

                ref_mean = psd[ref_band].mean()
                flank_mean = psd[flank_band].mean()
                
                if ref_mean > 0 and flank_mean > 0:
                    line_ratio = psd[line_band].mean() / ref_mean
                    peak_prominence = psd[line_band].max() / flank_mean
                    
                    if line_ratio > 0.8 and peak_prominence > 5.0:
                        results['line_noise'].append(i)
        except Exception as e:
            log(f"Line noise detection failed: {e}", log_file, log_to_file, detail="debug")

    return results

def run_icalabel(ica: mne.preprocessing.ICA, raw: mne.io.Raw, 
                 excluded: List[int], log_file: Path,
                 log_to_file: bool = True) -> Tuple[List[int], Dict]:
    """Run ICLabel classification."""
    try:
        labels_dict = label_components(raw, ica, method="iclabel")
        labels = labels_dict["labels"]
        probas = labels_dict["y_pred_proba"]
        
        new_excluded = []
        label_info = {}
        
        for i, (label, prob) in enumerate(zip(labels, probas)):
            lbl = label.lower().strip()
            if lbl in ICALABEL_THRESHOLDS and prob > ICALABEL_THRESHOLDS[lbl]:
                if i not in excluded:
                    new_excluded.append(i)
                    label_info[i] = (label, prob)
        
        if new_excluded:
            info_strs = [f"C{i}({label}: {prob.max():.2f})" 
                       for i, (label, prob) in label_info.items()]
            log(f"ICLabel added {len(new_excluded)}: {', '.join(info_strs)}",
                log_file, log_to_file)
        
        return new_excluded, label_info
    except Exception as e:
        log(f"ICLabel failed: {e}", log_file, log_to_file)
        return [], {}

def detect_extreme_components(ica: mne.preprocessing.ICA, raw: mne.io.Raw,
                              excluded: List[int], log_file: Path,
                              log_to_file: bool = True) -> List[int]:
    """Detect components with extreme signal characteristics."""
    src_data = ica.get_sources(raw).get_data()
    extreme = []
    
    for i in range(ica.n_components_):
        if i in excluded:
            continue
        
        x = src_data[i]
        var = np.var(x)
        kurt = kurtosis(x)
        ptp = np.ptp(x)
        
        if var < 1e-14 or kurt > 10000 or ptp > 10000:
            extreme.append(i)
            log(f"Excluded C{i} via signal metrics", log_file, log_to_file, detail="debug")
    
    return extreme

def log_ica_summary(ica: mne.preprocessing.ICA, results: Dict[str, List[int]],
                    icalabel_info: Dict, log_file: Path, log_to_file: bool = True):
    """Log ICA artifact rejection summary."""
    log("\n" + "━" * 60, log_file, log_to_file)
    log("🧩 ICA ARTIFACT REJECTION SUMMARY", log_file, log_to_file)
    log("━" * 60, log_file, log_to_file)
    log(f"{'Total components':<18} {ica.n_components_}", log_file, log_to_file)
    log(f"{'Excluded':<18} {len(ica.exclude)}", log_file, log_to_file)
    log("", log_file, log_to_file)
    
    labels = {
        'blink': 'Blink',
        'horizontal': 'Horizontal eye',
        'ecg': 'ECG',
        'muscle': 'Muscle',
        'frontal_lf': 'Frontal LF',
        'line_noise': 'Line noise',
        'extreme': 'Signal metrics'
    }
    
    for key, label in labels.items():
        log(f"{label:<18} {sorted(results[key])}", log_file, log_to_file)
    
    if icalabel_info:
        info_str = ", ".join([f"C{i}({lbl}: {prob.max():.2f})" 
                             for i, (lbl, prob) in icalabel_info.items()])
        log(f"{'ICLabel':<18} {info_str}", log_file, log_to_file)
    else:
        log(f"{'ICLabel':<18} []", log_file, log_to_file)
    
    log(f"\n🔧 Final exclude list: {sorted(ica.exclude)}", log_file, log_to_file)
    log("━" * 60, log_file, log_to_file)

def save_ica_plots(ica: mne.preprocessing.ICA, subject: str, output_path: Path,
                   log_file: Path, log_to_file: bool = True, cmap: str = 'plasma'):
    """Save ICA component plots."""
    try:
        fig_components = ica.plot_components(cmap=cmap, show=False)
        if not isinstance(fig_components, list):
            fig_components = [fig_components]

        for i, fig in enumerate(fig_components):
            fig_path = output_path / "plots" / f"{subject}_ica_components_page{i}.png"
            fig.savefig(fig_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
        
        log(f"🖼️ Saved {len(fig_components)} ICA component page(s)", log_file, log_to_file)
    except Exception as e:
        log(f"⚠️ Failed to save ICA plots: {e}", log_file, log_to_file)

def run_automatic_ica_cleaning(eeg_data: mne.io.Raw, subject: str, output_path: Path,
                               plot: bool, log_file: Path, log_to_file: bool = True,
                               n_components: float = 0.99,
                               random_state: int = 99,
                               use_artifact_detection_channels: bool = True) -> Tuple[mne.io.Raw, Dict]:
    """Run complete ICA artifact detection and removal pipeline."""
    log("Running automatic ICA cleaning...", log_file, log_to_file)
    
    raw = eeg_data.copy().pick_types(eeg=True, eog=True, ecg=True, emg=True)
    raw.set_channel_types({ch: 'emg' for ch in EMG_CHS if ch in raw.ch_names})
    
    # Pass the flag to create_bipolar_channels
    raw_filtered = create_bipolar_channels(raw.copy(), log_file, log_to_file, 
                                          use_artifact_detection_channels=use_artifact_detection_channels)

    # Fit ICA
    log("Fitting ICA with Extended Infomax...", log_file, log_to_file, detail="debug")
    ica = mne.preprocessing.ICA(
        n_components=n_components,
        random_state=random_state,
        method='picard',
        fit_params=dict(ortho=False, extended=True),
        max_iter='auto'
    )
    ica.fit(raw_filtered)

    log(f"ICA fitted with {ica.n_components_} components", log_file, log_to_file, detail="debug")

    # Pass the flag to detect_artifact_components
    detection_results = detect_artifact_components(ica, raw_filtered, log_file, log_to_file,
                                                  use_artifact_detection_channels=use_artifact_detection_channels)

    # Combine all detections
    ica.exclude = []
    if use_artifact_detection_channels:
        # Only add EOG/ECG/EMG/FL results if the flag is True
        for key in ['blink', 'horizontal', 'ecg', 'muscle', 'frontal_lf', 'line_noise']:
            ica.exclude.extend(detection_results[key])
    # If flag is False, only ICLabel and extreme components will be added below

    # Run ICLabel (always runs)
    icalabel_excluded, icalabel_info = run_icalabel(ica, raw_filtered, ica.exclude, log_file, log_to_file)
    ica.exclude.extend(icalabel_excluded)
    detection_results['icalabel'] = icalabel_excluded

    # Detect extreme components (always runs)
    extreme = detect_extreme_components(ica, raw_filtered, ica.exclude, log_file, log_to_file)
    ica.exclude.extend(extreme)
    detection_results['extreme'] = extreme

    # Apply ICA
    log(f"Applying ICA, excluding {len(ica.exclude)} components", log_file, log_to_file)
    cleaned_data = ica.apply(eeg_data.copy())

    # Log summary
    log_ica_summary(ica, detection_results, icalabel_info, log_file, log_to_file)

    # Save plots if requested
    if plot and ica.exclude:
        save_ica_plots(ica, subject, output_path, log_file, log_to_file)

    # Package results
    ica_object = {
        'ica_model': ica,
        'original_data': eeg_data,
        'filtered_data': raw_filtered,
        'auto_excluded': ica.exclude.copy(),
        'detection_results': {k: sorted(v) for k, v in detection_results.items()},
        'icalabel_info': icalabel_info,
        'parameters': {
            'n_components': n_components,
            'random_state': random_state,
            'use_artifact_detection_channels': use_artifact_detection_channels
        }
    }
    
    log("✅ ICA cleaning complete", log_file, log_to_file)
    return cleaned_data, ica_object

def plot_psd_comparison(raw_filtered: mne.io.Raw, cleaned_data: mne.io.Raw, subject: str,
                        output_path: Path, log_file: Path, log_to_file: bool = True):
    """Plot PSD comparison before and after ICA."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)

    raw_filtered.compute_psd(fmax=120, picks='eeg', exclude='bads').plot(axes=ax1, show=False)
    ax1.set_title('Before ICA', fontsize=12)
    ax1.set_xlabel('')

    cleaned_data.compute_psd(fmax=120, picks='eeg', exclude='bads').plot(axes=ax2, show=False)
    ax2.set_title('After ICA', fontsize=12)

    fig.suptitle('Power Spectral Density: Before vs. After ICA', fontsize=16)
    plt.subplots_adjust(top=0.94, hspace=0.3)

    fig_path = output_path / "plots" / f"{subject}_psd_comparison.png"
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    log(f"📊 PSD comparison saved to: {fig_path}", log_file, log_to_file)


def save_data(cleaned_data: mne.io.Raw, subject: str, output_path: Path,
              log_file: Path, log_to_file: bool = True):
    """Save cleaned data to FIF file."""
    sub_id = subject.replace('sub-', '')
    fname = f"sub-{sub_id}_eeg_ica_cleaned_raw.fif"
    full_path = output_path / fname
    cleaned_data.save(str(full_path), overwrite=True)
    log(f"Cleaned data saved to: {full_path}", log_file, log_to_file)


def interpolate_bads(raw: mne.io.Raw, log_file: Path, log_to_file: bool = True) -> mne.io.Raw:
    """Interpolate bad channels if any exist."""
    bads = raw.info['bads']
    if bads:
        log(f"Interpolating bad channels: {bads}", log_file, log_to_file)
        raw.interpolate_bads(reset_bads=True)
    else:
        log("No bad channels to interpolate", log_file, log_to_file)
    return raw



# ============================================================================
# MAIN PIPELINE FUNCTION
# ============================================================================

def run_preprocessing_pipeline(
    subject: str,
    input_path: str,
    gpsc_file: str,
    # project_id: str, # Removed as it's not used
    base_output_path: str,
    plot: bool = True,
    random_state: int = 99,
    log_to_file: bool = True,
    apply_highpass: bool = True,
    apply_lowpass: bool = True,
    apply_notch: bool = True,
    l_freq: float = 1.0,
    h_freq: float = 100.0,
    line_freq: float = 60.0,
    input_format: str = "mff",
    append_subject_to_output: bool = True,
    use_artifact_detection_channels: bool = True,  # Add this new parameter
    pre_ica_mad_threshold: float = 25.0,  # Higher threshold for pre-ICA detection
    post_ica_mad_threshold: float = 12.0  # Lower threshold for post-ICA detection
):
    """
    Execute the complete EEG preprocessing pipeline using standalone functions.
    
    This function replicates the functionality of the EEGICAProcessor.run() method.
    """
    # Setup output directories
    if append_subject_to_output:
        output_path = Path(base_output_path) / subject
    else:
        output_path = Path(base_output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "plots").mkdir(exist_ok=True)

    # Setup logging
    log_file = setup_logging(subject, output_path, log_to_file)

    # --- Pipeline Execution ---
    log("🔄 Starting preprocessing...", log_file, log_to_file)

    # 1. Load and prepare data
    log("🔧 Loading data...", log_file, log_to_file)
    raw = load_raw_data(Path(input_path), input_format, log_file, log_to_file)
    raw = apply_channel_renaming(raw, log_file, log_to_file)
    raw = apply_montage(raw, Path(gpsc_file), log_file, log_to_file)
    log("✅ Loading data complete", log_file, log_to_file)

    # 2. Filter data
    log("🔧 Filtering...", log_file, log_to_file)
    raw_filtered = filter_data(
        raw, apply_highpass, apply_lowpass, apply_notch,
        l_freq, h_freq, line_freq, log_file, log_to_file
    )
    log("✅ Filtering complete", log_file, log_to_file)

    # 3. Detect bad channels (before ICA, WITH protection) - HIGHER threshold
    log("🔧 Detecting bad channels (before ICA, with protection, threshold=25.0)...", log_file, log_to_file)
    raw_filtered = detect_bad_channels(
        raw_filtered, subject, output_path, plot, log_file, log_to_file,
        protected_channels=PROTECTED_CHANNELS,  # Use the global constant for pre-ICA detection
        mad_threshold=pre_ica_mad_threshold  # Higher threshold (more conservative)
    )
    log("✅ Detecting bad channels (before ICA) complete", log_file, log_to_file)

    # 4. Apply CAR (Common Average Reference)
    log("🔧 Applying CAR...", log_file, log_to_file)
    raw_filtered = raw_filtered.set_eeg_reference('average', verbose=False)
    log("✅ Applying CAR complete", log_file, log_to_file)

    # 5. Interpolate bad channels (before ICA)
    log("🔧 Interpolating bad channels (before ICA)...", log_file, log_to_file)
    raw_filtered = interpolate_bads(raw_filtered, log_file, log_to_file)
    log("✅ Interpolating bad channels (before ICA) complete", log_file, log_to_file)

    # 6. Run ICA cleaning
    log("🔧 Running ICA...", log_file, log_to_file)
    cleaned_data, ica_obj = run_automatic_ica_cleaning(
        raw_filtered, subject, output_path, plot, log_file, log_to_file,
        n_components=0.99, random_state=random_state,
        use_artifact_detection_channels=use_artifact_detection_channels  # Pass the new parameter
    )
    log("✅ Running ICA complete", log_file, log_to_file)

    # Use the cleaned data from ICA as input - LOWER threshold
    log("🔧 Detecting bad channels (after ICA, NO protection, threshold=12.0)...", log_file, log_to_file)

    # Pass an empty set to disable protection
    post_ica_subject_name = f"{subject}_post_ica"
    cleaned_data_post_ica = detect_bad_channels(
        cleaned_data, post_ica_subject_name, output_path, plot, log_file, log_to_file,
        protected_channels=set(),  # Disable protection for post-ICA detection
        mad_threshold=post_ica_mad_threshold  # Lower threshold (more sensitive)
    )
    log("✅ Detecting bad channels (after ICA) complete", log_file, log_to_file)

    # 7. Interpolate bad channels (after ICA)
    log("🔧 Interpolating bad channels (after ICA)...", log_file, log_to_file)
    # Use the data after the new bad channel detection (which might have added more bads)
    cleaned_data_interpolated = interpolate_bads(cleaned_data_post_ica, log_file, log_to_file)
    log("✅ Interpolating bad channels (after ICA) complete", log_file, log_to_file)

    # 8. Plot PSD comparison (using the final interpolated data)
    log("🔧 Plotting PSD (after ICA & interpolation)...", log_file, log_to_file)
    # Use the fully processed data (after ICA and interpolation) for the final PSD plot
    plot_psd_comparison(raw_filtered, cleaned_data_interpolated, subject, output_path, log_file, log_to_file)
    log("✅ Plotting PSD complete", log_file, log_to_file)

    # 9. Save data (the final, fully processed data)
    log("🔧 Saving final data...", log_file, log_to_file)
    # Save the data after all processing steps
    save_data(cleaned_data_interpolated, subject, output_path, log_file, log_to_file)
    log("✅ Saving data complete", log_file, log_to_file)

    log("✅ FULL PREPROCESSING COMPLETE\n", log_file, log_to_file)
    
    # Optional: Return results for further use if needed
    # return cleaned_data_interpolated, ica_obj



# ============================================================================
# MAIN EXECUTION BLOCK (Example usage)
# ============================================================================

if __name__ == "__main__":
    # Example usage - replace these paths and parameters with your actual data
    SUBJECT_ID = "sub-001"
    INPUT_PATH = "/path/to/your/data/sub-001.mff" # or .fif
    GPSC_FILE = "/path/to/your/channels.gpsc"
    BASE_OUTPUT_PATH = "/path/to/output/directory"
    INPUT_FORMAT = "mff" # or "fif"

    # Run the full pipeline
    run_preprocessing_pipeline(
        subject=SUBJECT_ID,
        input_path=INPUT_PATH,
        gpsc_file=GPSC_FILE,
        base_output_path=BASE_OUTPUT_PATH,
        input_format=INPUT_FORMAT,
        use_artifact_detection_channels=True,  
        pre_ica_mad_threshold=15.0,   
        post_ica_mad_threshold=10.0   
    )
    print(f"Pipeline completed for subject {SUBJECT_ID}. Check output in {BASE_OUTPUT_PATH}/{SUBJECT_ID}")
