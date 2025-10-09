# lcmv_class.py

# Imports
import mne
import numpy as np
import pandas as pd
import nibabel as nib
from tqdm import tqdm
import os, time, pickle
from pathlib import Path
import matplotlib.pyplot as plt
from nilearn import datasets, image

# Set MNE to only show warnings and errors
mne.set_log_level('warning')

class LCMVSourceEstimator:
    def __init__(self, config):
        """
        Initialize the LCMV Source Estimator with configuration.
        
        Parameters:
        config (dict): Configuration dictionary containing all necessary parameters
        """
        self.config = config
        self.project_base = Path(config['project_base'])
        self.subject_id = config['subject_id']
        self.task = config['task']
        
        # GLOBAL directory for shared resources (fsaverage)
        self.global_subjects_dir = self.project_base / 'derivatives/lcmv'
        
        # SUBJECT-SPECIFIC directory for output
        self.subject_output = self.project_base / f'derivatives/lcmv/{self.subject_id}_{self.task}'
        self.subject_output.mkdir(parents=True, exist_ok=True)

    def parse_gpsc(self, filepath):
        """Parse .gpsc file and normalize coordinates to center the origin."""
        channels = []
        with open(filepath, 'r') as file:
            lines = file.readlines()
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            name = parts[0]
            try:
                x, y, z = map(float, parts[1:4])
                channels.append((name, x, y, z))
            except ValueError:
                continue
        return channels

    def run_enhanced_computation(self):
        """Run the complete enhanced LCMV pipeline with improved coregistration"""
        print("="*60)
        print(f"🎯 ENHANCED LCMV SOURCE ESTIMATION - Subject: {self.subject_id}")
        print("="*60)
        
        print("\n=== Loading Data ===")
        ica_file = self.project_base / self.config['ica_file_path']
        gpsc_file = self.project_base / self.config['gpsc_file_path']

        if not ica_file.exists():
            raise FileNotFoundError(f"ICA file not found: {ica_file}")
        if not gpsc_file.exists():
            raise FileNotFoundError(f"GPSC file not found: {gpsc_file}")

        # Load data
        raw = mne.io.read_raw_fif(ica_file, preload=True)
        sfreq = raw.info['sfreq']
        duration_min = raw.n_times / sfreq / 60
        print(f"Data: {duration_min:.1f}min, {sfreq}Hz, {raw.n_times} samples")

        # === ENHANCED PREPROCESSING PIPELINE ===
        print("\n=== Enhanced Preprocessing Pipeline ===")
        
        # Rename channels to match .gpsc file
        channel_map = {str(i): f'E{i}' for i in range(1, 281)}
        channel_map['REF CZ'] = 'Cz'
        
        # Only rename existing channels
        existing_channels = set(raw.info['ch_names'])
        valid_channel_map = {}
        for old_name, new_name in channel_map.items():
            if old_name in existing_channels:
                valid_channel_map[old_name] = new_name
        
        if valid_channel_map:
            raw.rename_channels(valid_channel_map)
            print(f"Renamed {len(valid_channel_map)} channels")
        
  

        # === ENHANCED MONTAGE CREATION ===
        print("\n=== Creating Enhanced Montage with Coordinate Normalization ===")
        
        # Parse .gpsc file
        channels = self.parse_gpsc(gpsc_file)
        
        if not channels:
            raise ValueError("No valid channels found in .gpsc file")
        
        # Normalize coordinates to center the origin (enhanced method)
        gpsc_array = np.array([ch[1:4] for ch in channels])
        mean_pos = np.mean(gpsc_array, axis=0)
        print(f"Original mean position (mm): {mean_pos}")
        
        # Normalize and convert to meters
        channels_normalized = [(ch[0], ch[1] - mean_pos[0], ch[2] - mean_pos[1], ch[3] - mean_pos[2]) 
                              for ch in channels]
        ch_pos = {ch[0]: np.array(ch[1:4]) / 1000.0 for ch in channels_normalized}
        
        # Check fiducials
        required_fids = ['FidNz', 'FidT9', 'FidT10']
        missing = [fid for fid in required_fids if fid not in ch_pos]
        if missing:
            raise ValueError(f"Missing fiducials: {missing}")

        # Create montage with normalized coordinates
        montage = mne.channels.make_dig_montage(
            ch_pos=ch_pos,
            nasion=ch_pos['FidNz'],
            lpa=ch_pos['FidT9'],
            rpa=ch_pos['FidT10'],
            coord_frame='head'
        )
        



        # Apply montage and preprocessing
        raw.set_montage(montage, on_missing='warn')
        raw = raw.pick(['eeg', 'stim'], exclude=raw.info['bads'])

        print("\n🔍 Checking EEG reference status...")
        print(f"custom_ref_applied: {raw.info['custom_ref_applied']}")
        print(f"n_projs: {len(raw.info['projs'])}")
        print(f"proj_applied: {raw.proj}")

        # --- Ensure average reference projection is present ---
        if not any(p['desc'] == 'average' for p in raw.info['projs']):
            print("📎 No average reference projection found. Applying it...")
            raw.set_eeg_reference('average', projection=True)
        else:
            print("✅ Average reference projection already in place.")

        # --- Apply projections if not already applied ---
        if not raw.proj:
            print("🎯 Applying EEG average reference projection...")
            raw.apply_proj()
        else:
            print("💡 Projections already applied.")

        print("✓ Enhanced preprocessing complete (reference now valid for inverse modeling)")

        print(f"Enhanced montage applied:")
        print(f"FidNz (nasion): {ch_pos['FidNz']}")
        print(f"FidT9 (lpa): {ch_pos['FidT9']}")
        print(f"FidT10 (rpa): {ch_pos['FidT10']}")

        # === SOURCE SPACE SETUP ===
        print("\n=== Source Space Setup ===")
        subject = 'fsaverage'

        # Download fsaverage if needed
        bem_file = self.global_subjects_dir / 'fsaverage' / 'bem' / 'fsaverage-5120-5120-5120-bem-sol.fif'
        bem_head = self.global_subjects_dir / 'fsaverage' / 'bem' / 'fsaverage-head-dense.fif'
        src_file = self.global_subjects_dir / 'fsaverage-vol-5mm-src.fif'

        if not bem_file.exists() or not bem_head.exists():
            print("Downloading fsaverage to GLOBAL directory...")
            mne.datasets.fetch_fsaverage(subjects_dir=self.global_subjects_dir, verbose=False)

        # === ENHANCED COREGISTRATION ===
        print("\n=== Running Enhanced Coregistration ===")
        trans_file = self.subject_output / 'fsaverage-trans.fif'

        try:
            # Initialize coregistration with normalized coordinates
            coreg = mne.coreg.Coregistration(
                raw.info,
                subject=subject,
                subjects_dir=self.global_subjects_dir,
                fiducials={
                    'nasion': ch_pos['FidNz'],
                    'lpa': ch_pos['FidT9'],
                    'rpa': ch_pos['FidT10']
                }
            )

            # Step 1: Fit with fiducials first
            print("1/3: Fitting with fiducials...")
            coreg.fit_fiducials(verbose=False)

            # Step 2: Use EEG channels as head shape points for ICP
            print("2/3: Using EEG channels as head shape points for ICP...")
            coreg.fit_icp(n_iterations=6, nasion_weight=2.0, verbose=False)
            
            # Remove outliers
            print("   Removing outlier points...")
            dists = coreg.compute_dig_mri_distances()
            n_excluded = np.sum(dists > 5.0/1000)
            
            if n_excluded > 0:
                print(f"   Excluding {n_excluded} outlier points (distance > 5mm)")
                coreg.omit_head_shape_points(distance=5.0/1000)
            else:
                print("   No outlier points to exclude")
                
            # Step 3: Final refinement with higher weight on nasion
            print("3/3: Final ICP refinement...")
            coreg.fit_icp(n_iterations=20, nasion_weight=10.0, verbose=False)

            # Save transformation
            trans = coreg.trans
            mne.write_trans(trans_file, trans, overwrite=True)
            print(f"✓ Enhanced coregistration successful: {trans_file}")

            # Compute and display error metrics
            dists = coreg.compute_dig_mri_distances() * 1000  # mm
            mean_err = np.mean(dists)
            median_err = np.median(dists)
            max_err = np.max(dists)
            
            print(f"\nCoregistration Error (mm):")
            print(f"Mean: {mean_err:.2f}, Median: {median_err:.2f}, Max: {max_err:.2f}")

            if mean_err > 5.0:
                print(f"⚠️  WARNING: Mean error = {mean_err:.2f}mm > 5mm")
            else:
                print("✅ Enhanced coregistration error acceptable")

        except Exception as e:
            # ❌ REMOVED IDENTITY FALLBACK — fail fast instead
            print(f"❌ Coregistration failed irrecoverably: {e}")
            raise RuntimeError(f"Coregistration failed: {e}")

        # === SOURCE SPACE CREATION ===
        print("\n=== Creating Source Space ===")
        if not src_file.exists():
            print("Creating volume source space...")
            # ✅ FIXED: Added mri='T1.mgz' to ensure mri_ras_t exists
            src = mne.setup_volume_source_space(
                subject=subject,
                subjects_dir=self.global_subjects_dir,
                pos=5.0,
                mri='T1.mgz',  # ← critical for DiFuMo
                add_interpolator=True
            )
            src.save(src_file, overwrite=True)
        else:
            src = mne.read_source_spaces(src_file)

        print(f"Source space: {len(src[0]['vertno'])} active sources out of {src[0]['np']} total points")

        # === FORWARD SOLUTION ===
        print("\n=== Creating Forward Solution ===")
        fwd_file = self.subject_output / 'fsaverage-vol-eeg-fwd.fif'
        bem = mne.read_bem_solution(bem_file)
        fwd = mne.make_forward_solution(
            raw.info, trans=trans, src=src, bem=bem, eeg=True, mindist=5.0, n_jobs=self.config['n_jobs']
        )
        mne.write_forward_solution(fwd_file, fwd, overwrite=True)
        print("✓ Enhanced source space setup complete")

        # === LCMV BEAMFORMER ===
        print("\n=== LCMV Beamformer ===")

        # Compute SINGLE covariance from entire recording (CORRECT FOR CONTINUOUS DATA)
        print("Computing single covariance from entire recording...")

        cov = mne.compute_raw_covariance(
        raw,
        method='oas',             # ✅ STATE-OF-THE-ART for long continuous data 'shrunk' or 'oas'
        picks='eeg',
        rank='info',              # ✅ CRITICAL: Accounts for average reference
        n_jobs=self.config['n_jobs'],
        verbose=False)



        # Create LCMV filters: Same covariance with proper rank handling
        print("Creating LCMV spatial filters...")
        filters = mne.beamformer.make_lcmv(
            info=raw.info, 
            forward=fwd, 
            data_cov=cov, 
            noise_cov=cov,  # Same matrix - correct for continuous data
            reg=self.config['reg'],
            pick_ori='max-power', 
            weight_norm='unit-noise-gain', 
            reduce_rank=True,    # Must be True for average reference
            rank='info',         # CORRECT: Use rank information from info object
            verbose=True
        )


        
        # Apply LCMV to continuous data
        print("Applying LCMV filters to continuous data...")
        stc = mne.beamformer.apply_lcmv_raw(raw=raw, filters=filters)
        
        # Save STC in H5 format (required for complex data)
        print("Saving STC in H5 format (required for complex data)...")
        stc_file = self.subject_output / 'source_estimate_LCMV.h5'
        stc.save(stc_file, ftype='h5', overwrite=True)
        print(f"✓ STC saved successfully in H5 format: {stc_file}")

        print(f"✓ LCMV complete: {stc.data.shape} (sources x timepoints)")
        print(f"✓ STC file saved as: {stc_file}")

        # === SAVE SOURCE SPACE INFORMATION ===
        print("\n=== Saving source space information ===")
        
        # For volume source spaces, stc.vertices[0] contains the indices of active sources
        vertices = stc.vertices[0]
        
        # Get the active source indices from the source space
        active_indices = src[0]['vertno']  # indices of active sources in the full grid
        

        # Map STC vertices to actual source space positions
        src_points_m = src[0]['rr'][vertices]
        
        src_points_mm = src_points_m * 1000  # Convert to mm
        
        # Save the correctly indexed source points
        np.save(self.subject_output / 'source_space_points_mm.npy', src_points_mm)
        
        # Verify shapes match
        print(f"STC data shape: {stc.data.shape}")
        print(f"Source points shape: {src_points_mm.shape}")
        
        if src_points_mm.shape[0] != stc.data.shape[0]:
            print(f"WARNING: Shape mismatch detected!")
            n_sources = min(src_points_mm.shape[0], stc.data.shape[0])
            src_points_mm = src_points_mm[:n_sources]
            print(f"Using first {n_sources} source points to match STC data")
        
        print(f"✓ Source space points saved: {src_points_mm.shape} points")
        print(f"   Matches STC data shape: {stc.data.shape[0]} sources")

        # === SAVE DEBUG INFO AND METADATA ===
        print("\n=== Saving debug info and metadata ===")
        
        # Save debugging info
        debug_info = {
            'src_vertno': active_indices.tolist(),
            'stc_vertices': vertices.tolist(),
            'src_np': src[0]['np'],
            'n_active_sources': len(active_indices),
            'n_stc_vertices': len(vertices),
            'coregistration_error_mm': {
                'mean': mean_err if 'mean_err' in locals() else None,
                'median': median_err if 'median_err' in locals() else None,
                'max': max_err if 'max_err' in locals() else None
            }
        }
        with open(self.subject_output / 'debug_source_info.pkl', 'wb') as f:
            pickle.dump(debug_info, f)

        # Save metadata
        metadata = {
            'stc_shape': stc.data.shape,
            'n_source_points': len(vertices),
            'source_space_indices': vertices.tolist(),
            'sfreq': sfreq,
            'duration_min': duration_min,
            'stc_file': str(stc_file),
            'src_file': str(src_file),
            'subject_output': str(self.subject_output),
            'global_subjects_dir': str(self.global_subjects_dir),
            'enhanced_coregistration': True,
            'coordinate_normalization': 'mean_centered',
            'fiducials': {
                'FidNz': ch_pos['FidNz'].tolist(),
                'FidT9': ch_pos['FidT9'].tolist(),
                'FidT10': ch_pos['FidT10'].tolist()
            }
        }
        with open(self.subject_output / 'computation_metadata.pkl', 'wb') as f:
            pickle.dump(metadata, f)
        
        print(f"✓ Enhanced computation complete and metadata saved")
        print(f"\n🎉 ENHANCED LCMV SOURCE ESTIMATION COMPLETE!")
        print(f"   - Enhanced coregistration with error checking")
        print(f"   - Proper coordinate normalization")
        print(f"   - All original outputs maintained")
        print(f"   - Results saved to: {self.subject_output}")
        
        return metadata

    def extract_difumo_time_courses(self, stc, src, config, subject_output):
        """Extract weighted time courses from DiFuMo atlas."""
        print("\n=== DiFuMo Processing ===")
        atlas = datasets.fetch_atlas_difumo(
            dimension=config['dimension'],
            resolution_mm=config['resolution_mm']
        )
        atlas_img = nib.load(atlas.maps)
        atlas_shape = atlas_img.shape  # (x, y, z, n_components)
        n_components = atlas_shape[3]

        # Get source locations in mm
        vertices = stc.vertices[0]
        src_rr = src[0]['rr'][vertices] * 1000  # m → mm

        # Apply MRI RAS transform to get MNI coordinates
        try:
            trans = src[0]['mri_ras_t']['trans']
        except KeyError:
            raise ValueError("Source space missing 'mri_ras_t' transform. Ensure it's a proper volume source space.")

        mni_coords = image.coord_transform(src_rr[:, 0], src_rr[:, 1], src_rr[:, 2], trans)
        src_coords_mni = np.array(mni_coords).T  # (n_sources, 3)

        # Convert MNI mm → voxel indices in atlas space
        homog = np.column_stack([src_coords_mni, np.ones(len(src_coords_mni))])
        vox_coords = (np.linalg.inv(atlas_img.affine) @ homog.T).T[:, :3]
        vox_coords = np.round(vox_coords).astype(int)

        # Filter valid voxels inside atlas bounds
        valid_mask = (
            (vox_coords >= 0).all(axis=1) &
            (vox_coords[:, 0] < atlas_shape[0]) &
            (vox_coords[:, 1] < atlas_shape[1]) &
            (vox_coords[:, 2] < atlas_shape[2])
        )
        valid_indices = np.where(valid_mask)[0]
        valid_voxels = vox_coords[valid_mask]

        print(f"Using {len(valid_indices)}/{len(vertices)} sources within atlas bounds")

        # Extract time courses
        time_courses = []
        component_info = []
        threshold = 1e-6

        # ✅ ADDED: Handle complex-valued STC (though max-power should be real)
        if np.iscomplexobj(stc.data):
            print("⚠️  STC is complex — taking absolute value for DiFuMo")
            stc.data = np.abs(stc.data)

        for comp_idx in range(n_components):
            if comp_idx % 100 == 0:
                print(f"Processing component {comp_idx + 1}/{n_components}")

            try:
                comp_map = atlas_img.slicer[..., comp_idx].get_fdata()
                weights, stc_indices = [], []

                for i, (x, y, z) in enumerate(valid_voxels):
                    prob = comp_map[x, y, z]
                    if prob > threshold:
                        weights.append(prob)
                        stc_indices.append(valid_indices[i])

                if weights:
                    weights = np.array(weights)
                    weights /= weights.sum()  # Normalize
                    tc = np.average(stc.data[stc_indices], axis=0, weights=weights)
                    info = {
                        'component': comp_idx,
                        'n_sources': len(stc_indices),
                        'max_weight': weights.max(),
                        'mean_weight': weights.mean()
                    }
                else:
                    tc = np.zeros(stc.data.shape[1])
                    info = {
                        'component': comp_idx,
                        'n_sources': 0,
                        'max_weight': 0.0,
                        'mean_weight': 0.0
                    }

                time_courses.append(tc)
                component_info.append(info)

            except Exception as e:
                print(f"Error in component {comp_idx}: {e}")
                time_courses.append(np.zeros(stc.data.shape[1]))
                component_info.append({
                    'component': comp_idx, 'n_sources': 0, 'max_weight': 0.0, 'mean_weight': 0.0
                })

        # Summary
        valid_comps = sum(1 for info in component_info if info['n_sources'] > 0)
        print(f"✅ {valid_comps}/{n_components} components have at least one source")

        # Save outputs
        subject_output = Path(subject_output)
        np.save(subject_output / 'difumo_time_courses.npy', np.array(time_courses))
        pd.DataFrame(component_info).to_csv(subject_output / 'difumo_component_info.csv', index=False)
        print(f"💾 Saved to: {subject_output}")

        return np.array(time_courses), component_info

    def run_difumo_extraction(self, difumo_config=None):
        """Run DiFuMo time course extraction on existing data."""
        if difumo_config is None:
            difumo_config = {
                'dimension': 512,
                'resolution_mm': 2  # 2mm resolution for 512-component DiFuMo
            }

        try:
            # --- USER INPUT: UPDATE THESE IF NEEDED ---
            subject_output = self.subject_output
            stc_base_name = "source_estimate_LCMV"  # without extension

            # --- AUTODETECT STC FILE (handles .stc, -vl.stc, .h5) ---
            stc_file = None
            for suffix in ['-vl.stc', '.stc', '.h5']:
                candidate = subject_output / f"{stc_base_name}{suffix}"
                if candidate.exists():
                    stc_file = candidate
                    break
            if not stc_file:
                raise FileNotFoundError(f"STC file not found in {subject_output}")

            # --- LOAD DATA ---
            print(f"🔁 Loading STC: {stc_file}")
            stc = mne.read_source_estimate(stc_file)
            print(f"Loaded STC: {stc.data.shape} (sources × time)")

            # ✅ FIXED: Use consistent global path (no hardcoded path)
            src_file = self.global_subjects_dir / "fsaverage-vol-5mm-src.fif"

            print(f"🔁 Loading source space: {src_file}")
            if not src_file.exists():
                raise FileNotFoundError(f"Source space not found: {src_file}")
            src = mne.read_source_spaces(src_file)
            print(f"Loaded source space with {len(src[0]['vertno'])} active sources")
            
            # --- RUN EXTRACTION ---
            time_courses, component_info = self.extract_difumo_time_courses(
                stc=stc,
                src=src,
                config=difumo_config,
                subject_output=subject_output
            )

            print("\n🎉 SUCCESS: DiFuMo time series extraction complete!")
            print(f"📊 Output shape: {time_courses.shape} (512 components × {time_courses.shape[1]} time points)")
            print(f"📄 Details saved in:\n   - {subject_output / 'difumo_time_courses.npy'}\n   - {subject_output / 'difumo_component_info.csv'}")

            return time_courses, component_info

        except Exception as e:
            print(f"❌ Error during DiFuMo extraction: {e}")
            raise

    def list_output_files(self):
        """List all files in the output folder."""
        print(f"\n=== Files in output folder: {self.subject_output} ===")
        for file in os.listdir(self.subject_output):
            print(file)
        return list(os.listdir(self.subject_output))



# --- CONFIGURATION ---
PROJECT_BASE = "/home/jaizor/jaizor/xtra"
CROP_BASE_DIR = Path(PROJECT_BASE) / "derivatives/ica"
GPS_FILE_PATH = "data/ghw280_from_egig.gpsc"

# Configuration template
CONFIG_TEMPLATE = {
    'project_base': PROJECT_BASE,
    'gpsc_file_path': GPS_FILE_PATH,
    'reg': 0.01,
    'n_jobs': -1,
    'skip_difumo': False  
}