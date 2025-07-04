import os
import torch
from torch.utils.data import Dataset
import nibabel as nib
import numpy as np
from pathlib import Path
import random
import logging

logger = logging.getLogger(__name__)


class BraTS3DUnifiedDataset(Dataset):
    """
    Fixed BraTS 3D dataset with corrected available_modalities reporting
    and improved debugging capabilities
    """
    
    def __init__(self, data_root, phase='train', crop_size=(64, 64, 64), 
                 min_input_modalities=3, crops_per_volume=4, use_full_volumes=False,
                 input_size=(80, 80, 80)):
        self.data_root = Path(data_root)
        self.phase = phase
        self.crop_size = tuple(crop_size)
        self.input_size = tuple(input_size)
        self.min_input_modalities = min_input_modalities
        self.crops_per_volume = crops_per_volume
        self.use_full_volumes = use_full_volumes
        
        # Standard BraTS modalities
        self.modalities = ['t1n', 't1c', 't2w', 't2f']
        
        # Find all case directories
        self.cases = self._find_cases()
        
        if len(self.cases) == 0:
            raise ValueError(f"No valid cases found in {data_root}")
        
        # Create multiple crops per case for data augmentation (patches only)
        self.samples = []
        if self.use_full_volumes:
            # One sample per case for full volumes
            for case in self.cases:
                self.samples.append((case, 0))
        else:
            # Multiple crops per case for patch-based training
            for case in self.cases:
                for crop_idx in range(self.crops_per_volume):
                    self.samples.append((case, crop_idx))
        
        mode_str = "full volumes" if self.use_full_volumes else "patches"
        logger.info(f"Found {len(self.cases)} cases, {len(self.samples)} total samples for {phase} phase ({mode_str})")
    
    def _find_cases(self):
        """Find all valid BraTS case directories"""
        cases = []
        
        if not self.data_root.exists():
            raise FileNotFoundError(f"Data root {self.data_root} does not exist")
        
        for item in self.data_root.iterdir():
            if item.is_dir():
                available_count = sum(1 for mod in self.modalities 
                                    if self._find_modality_file(item, mod) is not None)
                
                if available_count >= self.min_input_modalities:
                    cases.append(item)
                    logger.debug(f"Found case {item.name} with {available_count} modalities")
        
        return sorted(cases, key=lambda x: x.name)
    
    def _find_modality_file(self, case_dir, modality):
        """Find modality file using specific BraTS patterns"""
        patterns = [
            f'*-{modality}.nii*',        # Exact match: BraTS-GLI-xxxxx-t1n.nii.gz
            f'*_{modality}.nii*',        # Underscore: case_t1n.nii.gz
            f'*{modality.upper()}.nii*', # Uppercase exact: caset1N.nii.gz
            f'*{modality}.nii*'          # Last resort: exact modality name
        ]
        
        for pattern in patterns:
            files = list(case_dir.glob(pattern))
            if files:
                # Filter to ensure the modality name appears exactly at word boundaries
                exact_files = []
                for f in files:
                    fname = f.stem.lower()
                    if (f'-{modality}.' in f.name.lower() or 
                        f'_{modality}.' in f.name.lower() or
                        f'{modality}.' in f.name.lower() or
                        fname.endswith(f'-{modality}') or
                        fname.endswith(f'_{modality}')):
                        exact_files.append(f)
                
                if exact_files:
                    exact_files.sort(key=lambda x: (x.suffix != '.gz', x.name))
                    return exact_files[0]
        return None
    
    def _load_full_volume(self, filepath):
        """Load full NIfTI volume without resizing"""
        try:
            nii = nib.load(str(filepath))
            volume = nii.get_fdata().astype(np.float32)
            return volume, nii.affine
        except Exception as e:
            logger.error(f"Error loading {filepath}: {e}")
            return None, None
    
    def _get_processing_size(self):
        """Get the target size for processing"""
        if self.use_full_volumes:
            return self.input_size
        else:
            return self.crop_size
    
    def _resize_volume(self, volume, target_size):
        """Resize volume to target size using trilinear interpolation"""
        from scipy.ndimage import zoom
        
        # Calculate zoom factors for each dimension
        zoom_factors = [target_size[i] / volume.shape[i] for i in range(3)]
        
        # Resize using trilinear interpolation
        resized = zoom(volume, zoom_factors, order=1)  # order=1 for trilinear
        
        return resized
    
    def _get_random_crop_coords(self, volume_shape):
        """Get random crop coordinates that fit within volume"""
        coords = []
        for i, (vol_dim, crop_dim) in enumerate(zip(volume_shape, self.crop_size)):
            if vol_dim <= crop_dim:
                start = 0
            else:
                max_start = vol_dim - crop_dim
                start = random.randint(0, max_start)
            coords.append((start, start + crop_dim))
        
        return coords
    
    def _extract_crop(self, volume, crop_coords):
        """Extract crop from volume using coordinates"""
        x_start, x_end = crop_coords[0]
        y_start, y_end = crop_coords[1]
        z_start, z_end = crop_coords[2]
        
        # Handle cases where volume is smaller than crop size
        crop = np.zeros(self.crop_size, dtype=volume.dtype)
        
        # Calculate actual crop size
        actual_x = min(x_end, volume.shape[0]) - x_start
        actual_y = min(y_end, volume.shape[1]) - y_start
        actual_z = min(z_end, volume.shape[2]) - z_start
        
        if actual_x > 0 and actual_y > 0 and actual_z > 0:
            crop[:actual_x, :actual_y, :actual_z] = volume[
                x_start:x_start+actual_x,
                y_start:y_start+actual_y,
                z_start:z_start+actual_z
            ]
        
        return crop
    
    def _normalize_volume(self, volume):
        """Normalize to [-1, 1] range"""
        if not np.any(volume > 0):
            return np.full_like(volume, -1.0)  # Background = -1 for [-1,1]
        v_min = np.amin(volume)
        v_max = np.amax(volume)
        if v_max > v_min:
            volume = 2 * (volume - v_min) / (v_max - v_min) - 1
        return np.clip(volume, -1.0, 1.0)
    
    def validate_normalization_consistency(self, sample):
        """Validate that normalization is consistent with [-1,1]"""
        target_min, target_max = sample['target'].min(), sample['target'].max()
        input_min, input_max = sample['input'].min(), sample['input'].max()
        assert target_min >= -1 and target_max <= 1, f"Target range [{target_min}, {target_max}] not in [-1,1]"
        assert input_min >= -1 and input_max <= 1, f"Input range [{input_min}, {input_max}] not in [-1,1]"
    
    def debug_normalization(self, idx=0):
        """Debug normalization values for [-1,1] range"""
        sample = self[idx]
        target = sample['target']
        inputs = sample['input']
        
        print(f"Target range: [{target.min():.3f}, {target.max():.3f}]")
        print(f"Background values in target: {target[target < -0.5].unique()}")
        print(f"Input ranges: {[(inputs[i].min().item(), inputs[i].max().item()) for i in range(4)]}")
        
        # Check if background is actually -1
        background_mask = target < -0.5
        if background_mask.any():
            bg_vals = target[background_mask]
            print(f"Background values: min={bg_vals.min():.3f}, max={bg_vals.max():.3f}, mean={bg_vals.mean():.3f}")
            print(f"Background should be -1.0, is it? {torch.allclose(bg_vals, torch.tensor(-1.0), atol=1e-3)}")
        
        self.validate_normalization_consistency(sample)
        
        return sample
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        case_dir, crop_idx = self.samples[idx]
        
        # Load ALL modalities and track which ones are successfully loaded
        modality_volumes = {}
        successfully_loaded_modalities = []
        volume_shape = None
        
        for modality in self.modalities:
            modality_file = self._find_modality_file(case_dir, modality)
            
            if modality_file:
                volume, affine = self._load_full_volume(modality_file)
                if volume is not None:
                    modality_volumes[modality] = volume
                    successfully_loaded_modalities.append(modality)
                    if volume_shape is None:
                        volume_shape = volume.shape
        
        # Ensure we have enough modalities to proceed
        if len(successfully_loaded_modalities) < self.min_input_modalities:
            # Fallback to zero volumes
            volume_shape = (240, 240, 155)  # Standard BraTS size
        
        # Fill missing modalities with zeros to maintain consistent 4-channel input
        for modality in self.modalities:
            if modality not in modality_volumes:
                if volume_shape is None:
                    volume_shape = (240, 240, 155)
                modality_volumes[modality] = np.zeros(volume_shape, dtype=np.float32)
        
        # Process volumes based on mode
        target_size = self._get_processing_size()
        processed_volumes = {}
        
        if self.use_full_volumes:
            # Full volume mode: resize to input_size
            for modality in self.modalities:
                volume = modality_volumes.get(modality, np.zeros(volume_shape, dtype=np.float32))
                resized = self._resize_volume(volume, target_size)
                normalized = self._normalize_volume(resized)
                processed_volumes[modality] = torch.FloatTensor(normalized)
        else:
            # Patch mode: extract random crops
            random.seed(hash((case_dir.name, crop_idx)) % (2**32))
            crop_coords = self._get_random_crop_coords(volume_shape)
            
            for modality in self.modalities:
                volume = modality_volumes.get(modality, np.zeros(volume_shape, dtype=np.float32))
                cropped = self._extract_crop(volume, crop_coords)
                normalized = self._normalize_volume(cropped)
                processed_volumes[modality] = torch.FloatTensor(normalized)
        
        # Select target modality from successfully loaded modalities
        if self.phase == 'train' and len(successfully_loaded_modalities) > 0:
            target_modality = random.choice(successfully_loaded_modalities)
        elif len(successfully_loaded_modalities) > 0:
            target_modality = successfully_loaded_modalities[0]
        else:
            target_modality = self.modalities[0]  # Fallback
        
        target_idx = self.modalities.index(target_modality)
        target_volume = processed_volumes[target_modality]
        
        # FIXED: Create input with target modality replaced by ZEROS
        input_modalities = torch.stack([processed_volumes[mod] for mod in self.modalities])
        input_modalities[target_idx] = torch.zeros_like(input_modalities[target_idx])
        
        # FIXED: Correct available_modalities reporting
        # Available modalities are the 3 non-target modalities that were successfully loaded
        available_non_target_modalities = [mod for mod in successfully_loaded_modalities if mod != target_modality]
        
        # For display purposes, show the actual status more clearly
        if target_modality in successfully_loaded_modalities:
            # Target was successfully loaded but is set to zeros in input
            display_available_modalities = available_non_target_modalities.copy()
        else:
            # Target was not successfully loaded (missing file)
            display_available_modalities = available_non_target_modalities.copy()
        
        return {
            'input': input_modalities,      # [4, H, W, D] with target as ZEROS, range [-1,1]
            'target': target_volume,        # [H, W, D] target modality, range [-1,1]
            'target_idx': target_idx,
            'case_name': case_dir.name,
            'target_modality': target_modality,
            'available_modalities': display_available_modalities,
            'successfully_loaded_modalities': successfully_loaded_modalities,
            'crop_coords': crop_coords if not self.use_full_volumes else None,
            'processing_mode': 'full_volume' if self.use_full_volumes else 'patch'
        }