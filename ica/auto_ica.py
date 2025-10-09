# auto_ica.py - Optimized version

# Standard library
from pathlib import Path
import logging

# Third-party scientific stack
import numpy as np
from scipy.stats import median_abs_deviation, kurtosis
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt

# MNE and related
import mne
from mne import io
from mne.io import constants  # For montage dig point kinds
from mne_icalabel import label_components

# Type hints
from typing import Dict, List, Tuple, Optional, Union

# Configure MNE
mne.set_log_level('WARNING')


class EEGICAProcessor:
    """Automated EEG preprocessing with ICA artifact removal."""
    
    # Class-level constants
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
        'muscle artifact': 0.80,
        'line noise': 0.80,
        'channel noise': 0.80
    }

    def __init__(
        self,
        subject: str,
        input_path: str,
        gpsc_file: str,
        project_id: str,
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
        append_subject_to_output: bool = True
    ):
        self.subject = subject
        self.input_path = Path(input_path)
        self.gpsc_file = Path(gpsc_file)
        self.project_id = project_id
        self.plot = plot
        self.random_state = random_state
        self.input_format = input_format

        # Filtering config - separate controls for each filter type
        self.apply_highpass = apply_highpass
        self.apply_lowpass = apply_lowpass
        self.apply_notch = apply_notch
        self.l_freq = l_freq
        self.h_freq = h_freq
        self.line_freq = line_freq


        # Setup output directories
        if append_subject_to_output:
            self.output_path = Path(base_output_path) / subject
        else:
            self.output_path = Path(base_output_path)

        self.output_path.mkdir(parents=True, exist_ok=True)
        (self.output_path / "plots").mkdir(exist_ok=True)



        # Setup logging
        self.log_file = self.output_path / f"{subject}_preproc_log.txt"
        self.log_to_file = log_to_file
        if log_to_file:
            self._log(f"Initialized EEGICAProcessor for {subject} ({input_format})")

        # Data containers
        self.raw = None
        self.raw_filtered = None
        self.cleaned_data = None
        self.ica_obj = None

    def _log(self, msg: str, detail: str = "normal"):
        """Log to file and optionally console."""
        if self.log_to_file:
            with open(self.log_file, 'a') as f:
                f.write(f"{msg}\n")
        if detail == "normal":
            print(msg)

    def _parse_gpsc(self, filepath: Path) -> List[Tuple[str, float, float, float]]:
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

    def _apply_channel_renaming_and_montage(self, raw: mne.io.Raw) -> mne.io.Raw:
        """Apply channel renaming and GPS montage."""
        self._log("Applying channel renaming and montage...")
        
        # Rename channels
        existing_map = {old: new for old, new in self.CHANNEL_RENAME_MAP.items() 
                       if old in raw.ch_names}
        if existing_map:
            raw.rename_channels(existing_map)
            self._log(f"Renamed {len(existing_map)} channels.")

        # Parse and apply montage
        channels = self._parse_gpsc(self.gpsc_file)
        if not channels:
            raise ValueError("No valid channels in .gpsc file")
        
        # Normalize positions
        gpsc_array = np.array([ch[1:4] for ch in channels])
        mean_pos = gpsc_array.mean(axis=0)
        self._log(f"Original mean position (mm): {mean_pos}")
        
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
        self._log("Montage applied.")
        return raw

    def load_and_prepare_data(self):
        """Load data from MFF or FIF format."""
        if self.input_format == "mff":
            self._log("Loading raw data from .mff...")
            self.raw = mne.io.read_raw_egi(str(self.input_path), preload=True)
        elif self.input_format == "fif":
            self._log(f"Loading raw data from .fif: {self.input_path}")
            if not self.input_path.is_file() or self.input_path.suffix != '.fif':
                raise ValueError(f"Invalid .fif file: {self.input_path}")
            self.raw = mne.io.read_raw_fif(str(self.input_path), preload=True)
        else:
            raise ValueError("input_format must be 'mff' or 'fif'")
        
        self.raw = self._apply_channel_renaming_and_montage(self.raw)

    def apply_highpass_filter(self, raw: mne.io.Raw) -> mne.io.Raw:
        """Apply highpass filter."""
        if self.apply_highpass and self.l_freq is not None:
            self._log(f"Applying highpass filter at {self.l_freq} Hz...")
            raw = raw.copy().filter(
                l_freq=self.l_freq, h_freq=None, picks=['eeg'],
                method='fir', phase='zero', fir_window='hamming',
                fir_design='firwin', n_jobs=-1
            )
        return raw

    def apply_lowpass_filter(self, raw: mne.io.Raw) -> mne.io.Raw:
        """Apply lowpass filter."""
        if self.apply_lowpass and self.h_freq is not None:
            self._log(f"Applying lowpass filter at {self.h_freq} Hz...")
            raw = raw.copy().filter(
                l_freq=None, h_freq=self.h_freq, picks=['eeg'],
                method='fir', phase='zero', fir_window='hamming',
                fir_design='firwin', n_jobs=-1
            )
        return raw

    def apply_notch_filter(self, raw: mne.io.Raw) -> mne.io.Raw:
        """Apply notch filter."""
        if self.apply_notch:
            nyquist = raw.info["sfreq"] / 2
            notch_freqs = np.arange(
                self.line_freq,
                int(nyquist // self.line_freq + 1) * self.line_freq,
                self.line_freq
            )
            notch_freqs = notch_freqs[notch_freqs < nyquist]

            if len(notch_freqs) > 0:
                self._log(f"Applying notch filter at: {notch_freqs}")
                raw = raw.copy().notch_filter(
                    freqs=notch_freqs, picks='eeg', method='spectrum_fit',
                    filter_length='auto', mt_bandwidth=1.0, p_value=0.05
                )
        return raw

    def filter_data(self):
        """Apply all selected filters in sequence."""
        self._log("Applying filters...")
        
        # Start with raw data
        filtered_raw = self.raw.copy()
        
        # Apply highpass filter if enabled
        filtered_raw = self.apply_highpass_filter(filtered_raw)
        
        # Apply lowpass filter if enabled
        filtered_raw = self.apply_lowpass_filter(filtered_raw)
        
        # Apply notch filter if enabled
        filtered_raw = self.apply_notch_filter(filtered_raw)
        
        # If no filters were applied, just copy the raw data
        if not (self.apply_highpass or self.apply_lowpass or self.apply_notch):
            self._log("No filters applied (all filter types disabled).")
            filtered_raw = self.raw.copy()
        
        self.raw_filtered = filtered_raw

        # Check Cz for flat signal
        if 'Cz' in self.raw_filtered.ch_names:
            if np.std(self.raw_filtered.get_data(picks=['Cz'])[0]) < 1e-6:
                self.raw_filtered.info['bads'].append('Cz')
                self._log("Marked Cz as bad (flat signal).")


    def detect_bad_channels(self, mad_threshold: float = 20, min_amplitude_uv: float = 0.1):
        """Detect bad channels using MAD-based outlier detection and save diagnostic plots."""
        self._log(f"Detecting bad channels (flat < {min_amplitude_uv} µV, noisy Z > {mad_threshold})...")

        raw_eeg = self.raw_filtered.copy().pick_types(eeg=True)
        available_chs = set(raw_eeg.ch_names)
        protected_chs = self.PROTECTED_CHANNELS & available_chs
        eeg_chs = [ch for ch in raw_eeg.ch_names if ch not in protected_chs]

        if not eeg_chs:
            self._log("No EEG channels available for detection.")
            return

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
        current_bads = set(self.raw_filtered.info['bads'])
        new_bads = [ch for ch in detected_bads if ch not in current_bads]
        self.raw_filtered.info['bads'] = sorted(current_bads | set(detected_bads))

        # Log results
        if protected_chs:
            self._log(f"Protected: {sorted(protected_chs)}")
        self._log(f"Detected bad channels: {detected_bads}")

        # === Save plots of BAD CHANNELS ONLY (if enabled) ===
        if self.plot and detected_bads:
            try:
                montage = self.raw_filtered.get_montage()
                if montage is None:
                    self._log("⚠️ No montage found — skipping topomap.")
                else:
                    # Reconstruct channel position dictionary from montage
                    ch_pos = {}
                    for d in montage.dig:
                        if d['kind'] == mne.io.constants.FIFF.FIFFV_POINT_EEG:
                            ch_name = montage.ch_names[d['ident'] - 1]
                            ch_pos[ch_name] = d['r']

                    self._plot_and_save_topomap(ch_pos, detected_bads)
                    self._plot_and_save_time_series(self.raw_filtered, detected_bads)

            except Exception as e:
                self._log(f"⚠️ Failed to generate/save bad channel plots: {e}")


    def _plot_and_save_topomap(self, ch_pos: Dict[str, np.ndarray], bad_channels: List[str]):
        """Plot and save topomap of ONLY bad channels, each with unique color."""
        if not bad_channels:
            return

        valid_bads = [ch for ch in bad_channels if ch in ch_pos]
        if not valid_bads:
            self._log("⚠️ No bad channels with valid positions.")
            return

        # Unique colors (viridis)
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
            self._log(f"Topomap background failed: {e}", detail="debug")

        ax.set_title('Detected Bad Channels (Topomap)', fontsize=12, pad=20)
        ax.set_xlim(-0.12, 0.12)
        ax.set_ylim(-0.12, 0.12)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fig_path = self.output_path / "plots" / f"{self.subject}_bad_channels_topomap.png"
        fig.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        self._log(f"🖼️ Bad channels topomap saved: {fig_path}")


    def _plot_and_save_time_series(self, raw: mne.io.Raw, bad_channels: List[str]):
        """Plot full-duration time series of ONLY bad channels, each with unique color."""
        if not bad_channels:
            return

        valid_bads = [ch for ch in bad_channels if ch in raw.ch_names]
        if not valid_bads:
            return

        # Get FULL data (no duration limit)
        data, times = raw[valid_bads, :]  # all time points

        # Unique colors
        cmap = plt.get_cmap('viridis')
        colors = cmap(np.linspace(0, 1, len(valid_bads)))
        color_dict = {ch: colors[i] for i, ch in enumerate(valid_bads)}

        # Plot
        n_ch = len(valid_bads)
        fig_height = min(2.2 * n_ch, 40)  # cap height for very long lists
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

        fig_path = self.output_path / "plots" / f"{self.subject}_bad_channels_timeseries.png"
        fig.savefig(fig_path, dpi=120, bbox_inches='tight')
        plt.close(fig)
        self._log(f"📈 Bad channels full time series saved: {fig_path}")


    def _create_bipolar_channels(self, raw: mne.io.Raw) -> mne.io.Raw:
        """Create bipolar reference channels for artifact detection."""
        bipolar_specs = [
            (self.VVEOG, 'vVEOG', 'eog', "blink detection"),
            (self.HEOG, 'BLINK_H', 'eog', "horizontal eye movement"),
            (self.ECG, 'ECG_BIO', 'ecg', "cardiac artifact")
        ]
        
        for (anode, cathode), name, ch_type, desc in bipolar_specs:
            if anode in raw.ch_names and cathode in raw.ch_names:
                raw = mne.set_bipolar_reference(
                    raw, anode=anode, cathode=cathode,
                    ch_name=name, drop_refs=False
                ).set_channel_types({name: ch_type})
                self._log(f"Created {name} ({anode}-{cathode}) for {desc}", detail="debug")
        
        return raw

    def _detect_artifact_components(self, ica: mne.preprocessing.ICA, 
                                   raw: mne.io.Raw) -> Dict[str, List[int]]:
        """Detect artifact components using multiple methods."""
        results = {
            'blink': [], 'horizontal': [], 'ecg': [],
            'muscle': [], 'frontal_lf': [], 'line_noise': [],
            'icalabel': [], 'extreme': []
        }

        # Blink detection
        if 'vVEOG' in raw.ch_names:
            try:
                idx, _ = ica.find_bads_eog(raw, ch_name='vVEOG', measure='zscore', threshold=3.0)
                results['blink'] = [int(i) for i in idx]
                self._log(f"Blink: {results['blink']}", detail="debug")
            except Exception as e:
                self._log(f"Blink detection failed: {e}", detail="debug")

        # Horizontal eye movement
        if 'BLINK_H' in raw.ch_names:
            try:
                idx, _ = ica.find_bads_eog(raw, ch_name='BLINK_H', measure='zscore', threshold=3.0)
                results['horizontal'] = [int(i) for i in idx]
                self._log(f"Horizontal: {results['horizontal']}", detail="debug")
            except Exception as e:
                self._log(f"Horizontal detection failed: {e}", detail="debug")

        # ECG detection
        if 'ECG_BIO' in raw.ch_names:
            try:
                idx, _ = ica.find_bads_ecg(raw, ch_name='ECG_BIO', method='correlation', 
                                          measure='zscore', threshold=3.0)
                results['ecg'] = [int(i) for i in idx]
                self._log(f"ECG: {results['ecg']}", detail="debug")
            except Exception as e:
                self._log(f"ECG detection failed: {e}", detail="debug")

        # Muscle artifacts
        for ch in self.EMG_CHS:
            if ch in raw.ch_names:
                try:
                    idx, _ = ica.find_bads_eog(raw, ch_name=ch, measure='zscore',
                                              l_freq=30, h_freq=100, threshold=3.0)
                    results['muscle'].extend([int(i) for i in idx])
                except Exception as e:
                    self._log(f"EMG detection failed for {ch}: {e}", detail="debug")
        results['muscle'] = list(set(results['muscle']))

        # Frontal low-frequency artifacts
        for ch in self.FRONTAL_CHS:
            if ch in raw.ch_names:
                try:
                    idx, _ = ica.find_bads_eog(raw, ch_name=ch, measure='zscore',
                                              l_freq=1.0, h_freq=10.0, threshold=3.5)
                    results['frontal_lf'].extend([int(i) for i in idx])
                except Exception as e:
                    self._log(f"Frontal LF failed for {ch}: {e}", detail="debug")
        results['frontal_lf'] = list(set(results['frontal_lf']))

        # Line noise detection
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
            self._log(f"Line noise detection failed: {e}", detail="debug")

        return results

    def _run_icalabel(self, ica: mne.preprocessing.ICA, 
                     raw: mne.io.Raw, excluded: List[int]) -> Tuple[List[int], Dict]:
        """Run ICLabel classification."""
        try:
            labels_dict = label_components(raw, ica, method="iclabel")
            labels = labels_dict["labels"]
            probas = labels_dict["y_pred_proba"]
            
            new_excluded = []
            label_info = {}
            
            for i, (label, prob) in enumerate(zip(labels, probas)):
                lbl = label.lower().strip()
                if lbl in self.ICALABEL_THRESHOLDS and prob > self.ICALABEL_THRESHOLDS[lbl]:
                    if i not in excluded:
                        new_excluded.append(i)
                        label_info[i] = (label, prob)
            
            if new_excluded:
                info_strs = [f"C{i}({label}: {prob.max():.2f})" 
                           for i, (label, prob) in label_info.items()]
                self._log(f"ICLabel added {len(new_excluded)}: {', '.join(info_strs)}")
            
            return new_excluded, label_info
        except Exception as e:
            self._log(f"ICLabel failed: {e}")
            return [], {}

    def _detect_extreme_components(self, ica: mne.preprocessing.ICA,
                                   raw: mne.io.Raw, excluded: List[int]) -> List[int]:
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
                self._log(f"Excluded C{i} via signal metrics", detail="debug")
        
        return extreme

    def run_automatic_ica_cleaning(self, eeg_data: mne.io.Raw,
                                  n_components: float = 0.99,
                                  random_state: int = 99) -> Tuple[mne.io.Raw, Dict]:
        """Run complete ICA artifact detection and removal pipeline."""
        raw = eeg_data.copy().pick_types(eeg=True, eog=True, ecg=True, emg=True)
        raw.set_channel_types({ch: 'emg' for ch in self.EMG_CHS if ch in raw.ch_names})
        raw_filtered = self._create_bipolar_channels(raw.copy())

        # Fit ICA
        self._log("Fitting ICA with Extended Infomax...", detail="debug")
        ica = mne.preprocessing.ICA(
            n_components=n_components,
            random_state=random_state,
            method='picard',
            fit_params=dict(ortho=False, extended=True),
            max_iter='auto'
        )
        ica.fit(raw_filtered)
        self._log(f"ICA fitted with {ica.n_components_} components", detail="debug")

        # Detect artifacts
        detection_results = self._detect_artifact_components(ica, raw_filtered)
        
        # Combine all detections
        ica.exclude = []
        for key in ['blink', 'horizontal', 'ecg', 'muscle', 'frontal_lf', 'line_noise']:
            ica.exclude.extend(detection_results[key])
        
        # Run ICLabel
        icalabel_excluded, icalabel_info = self._run_icalabel(ica, raw_filtered, ica.exclude)
        ica.exclude.extend(icalabel_excluded)
        detection_results['icalabel'] = icalabel_excluded
        
        # Detect extreme components
        extreme = self._detect_extreme_components(ica, raw_filtered, ica.exclude)
        ica.exclude.extend(extreme)
        detection_results['extreme'] = extreme

        # Apply ICA
        self._log(f"Applying ICA, excluding {len(ica.exclude)} components")
        cleaned_data = ica.apply(eeg_data.copy())

        # Log summary
        self._log_ica_summary(ica, detection_results, icalabel_info)

        # Save plots if requested
        if self.plot and ica.exclude:
            self._save_ica_plots(ica)

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
                'random_state': random_state
            }
        }
        
        return cleaned_data, ica_object

    def _log_ica_summary(self, ica: mne.preprocessing.ICA,
                        results: Dict[str, List[int]],
                        icalabel_info: Dict):
        """Log ICA artifact rejection summary."""
        self._log("\n" + "━" * 60)
        self._log("🧩 ICA ARTIFACT REJECTION SUMMARY")
        self._log("━" * 60)
        self._log(f"{'Total components':<18} {ica.n_components_}")
        self._log(f"{'Excluded':<18} {len(ica.exclude)}")
        self._log("")
        
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
            self._log(f"{label:<18} {sorted(results[key])}")
        
        if icalabel_info:
            info_str = ", ".join([f"C{i}({lbl}: {prob.max():.2f})" 
                                 for i, (lbl, prob) in icalabel_info.items()])
            self._log(f"{'ICLabel':<18} {info_str}")
        else:
            self._log(f"{'ICLabel':<18} []")
        
        self._log(f"\n🔧 Final exclude list: {sorted(ica.exclude)}")
        self._log("━" * 60)

    def _save_ica_plots(self, ica: mne.preprocessing.ICA, cmap: str = 'plasma'):
        """Save ICA component plots."""
        try:
            fig_components = ica.plot_components(cmap=cmap, show=False)
            if not isinstance(fig_components, list):
                fig_components = [fig_components]

            for i, fig in enumerate(fig_components):
                fig_path = self.output_path / "plots" / f"{self.subject}_ica_components_page{i}.png"
                fig.savefig(fig_path, dpi=150, bbox_inches='tight')
                plt.close(fig)
            
            self._log(f"🖼️ Saved {len(fig_components)} ICA component page(s)")
        except Exception as e:
            self._log(f"⚠️ Failed to save ICA plots: {e}")

    def run_ica_cleaning(self):
        """Run ICA cleaning with default parameters."""
        self._log("Running automatic ICA cleaning...")
        self.cleaned_data, self.ica_obj = self.run_automatic_ica_cleaning(
            self.raw_filtered,
            n_components=0.99,
            random_state=self.random_state
        )

    def plot_psd_comparison(self):
        """Plot PSD comparison before and after ICA."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
        
        self.raw_filtered.compute_psd(fmax=120, picks='eeg', exclude='bads').plot(axes=ax1, show=False)
        ax1.set_title('Before ICA', fontsize=12)
        ax1.set_xlabel('')
        
        self.cleaned_data.compute_psd(fmax=120, picks='eeg', exclude='bads').plot(axes=ax2, show=False)
        ax2.set_title('After ICA', fontsize=12)
        
        fig.suptitle('Power Spectral Density: Before vs. After ICA', fontsize=16)
        plt.subplots_adjust(top=0.94, hspace=0.3)
        
        fig_path = self.output_path / "plots" / f"{self.subject}_psd_comparison.png"
        fig.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        self._log(f"📊 PSD comparison saved to: {fig_path}")

    def save_data(self):
        """Save cleaned data to FIF file."""
        sub_id = self.subject.replace('sub-', '')
        fname = f"sub-{sub_id}_eeg_ica_cleaned_raw.fif"
        full_path = self.output_path / fname
        self.cleaned_data.save(str(full_path), overwrite=True)
        self._log(f"Cleaned data saved to: {full_path}")

    def run(self):
        """Execute complete preprocessing pipeline."""
        steps = [
            ("Loading data", self.load_and_prepare_data),
            ("Filtering", self.filter_data),
            ("Detecting bad channels", self.detect_bad_channels),
            ("Applying CAR", lambda: setattr(self, 'raw_filtered', 
                                            self.raw_filtered.set_eeg_reference('average', verbose=False))),
            ("Interpolating bad channels", self._interpolate_bads),
            ("Running ICA", self.run_ica_cleaning),
            ("Plotting PSD", self.plot_psd_comparison),
            ("Saving data", self.save_data)
        ]
        
        self._log("🔄 Starting preprocessing...")
        for step_name, step_func in steps:
            self._log(f"🔧 {step_name}...")
            step_func()
            self._log(f"✅ {step_name} complete")
        
        self._log("✅ FULL PREPROCESSING COMPLETE\n")

    def _interpolate_bads(self):
        """Interpolate bad channels if any exist."""
        bads = self.raw_filtered.info['bads']
        if bads:
            self._log(f"Interpolating bad channels: {bads}")
            self.raw_filtered.interpolate_bads(reset_bads=True)
        else:
            self._log("No bad channels to interpolate")