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
    Updated BraTS 3D dataset following professor's recommendations:
    1. Random cropping (no rescaling)
    2. Min-max normalization to [0,1]
    3. Consistent spatial regions across modalities
    """
    
    def __init__(self, data_root, phase='train', crop_size=(64, 64, 64), 
                 min_input_modalities=3, crops_per_volume=4):
        self.data_root = Path(data_root)
        self.phase = phase
        self.crop_size = tuple(crop_size)
        self.min_input_modalities = min_input_modalities
        self.crops_per_volume = crops_per_volume  # Multiple crops per MRI
        
        # Standard BraTS modalities
        self.modalities = ['t1n', 't1c', 't2w', 't2f']
        
        # Find all case directories
        self.cases = self._find_cases()
        
        if len(self.cases) == 0:
            raise ValueError(f"No valid cases found in {data_root}")
        
        # Create multiple crops per case for data augmentation
        self.samples = []
        for case in self.cases:
            for crop_idx in range(self.crops_per_volume):
                self.samples.append((case, crop_idx))
        
        logger.info(f"Found {len(self.cases)} cases, {len(self.samples)} total samples for {phase} phase")
    
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
        """Find modality file using common BraTS patterns"""
        patterns = [
            f'*{modality}*.nii*',
            f'*{modality.upper()}*.nii*',
            f'*-{modality}.nii*',
            f'*_{modality}.nii*'
        ]
        
        for pattern in patterns:
            files = list(case_dir.glob(pattern))
            if files:
                files.sort(key=lambda x: (x.suffix != '.gz', x.name))
                return files[0]
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
    
    def _get_random_crop_coords(self, volume_shape):
        """
        Get random crop coordinates that fit within volume
        Returns the same coordinates for all modalities to ensure spatial consistency
        """
        coords = []
        for i, (vol_dim, crop_dim) in enumerate(zip(volume_shape, self.crop_size)):
            if vol_dim <= crop_dim:
                # If volume dimension is smaller than crop, start at 0
                start = 0
            else:
                # Random start position
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
    
    def _normalize_volume_0_1(self, volume):
        """
        Professor's recommendation: Min-max normalization to [0, 1]
        """
        # Handle zero or constant volumes
        if not np.any(volume > 0):
            return np.zeros_like(volume)
        
        vmin = volume.min()
        vmax = volume.max()
        
        if vmax <= vmin:
            return np.zeros_like(volume)
        
        # Min-max normalization to [0, 1]
        volume_norm = (volume - vmin) / (vmax - vmin)
        
        # Validate normalization
        if np.any(np.isnan(volume_norm)) or np.any(np.isinf(volume_norm)):
            logger.warning("NaN/Inf values detected after normalization, using zero volume")
            return np.zeros_like(volume)
        
        return np.clip(volume_norm, 0.0, 1.0)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        case_dir, crop_idx = self.samples[idx]
        
        # Load all modalities
        modality_volumes = {}
        available_modalities = []
        volume_shape = None
        
        for modality in self.modalities:
            modality_file = self._find_modality_file(case_dir, modality)
            
            if modality_file:
                volume, affine = self._load_full_volume(modality_file)
                if volume is not None:
                    modality_volumes[modality] = volume
                    available_modalities.append(modality)
                    if volume_shape is None:
                        volume_shape = volume.shape
        
        if len(available_modalities) < self.min_input_modalities:
            # Fallback to zero volumes
            volume_shape = (240, 240, 155)  # Standard BraTS size
            for modality in self.modalities:
                if modality not in modality_volumes:
                    modality_volumes[modality] = np.zeros(volume_shape, dtype=np.float32)
        
        # Generate consistent random crop coordinates for all modalities
        # Use crop_idx as seed for reproducible crops per case
        random.seed(hash((case_dir.name, crop_idx)) % (2**32))
        crop_coords = self._get_random_crop_coords(volume_shape)
        
        # Debug print for first few samples to show original vs crop size
        if idx < 3:  # Only print for first 3 samples to avoid spam
            print(f"\n=== Dataset Debug Info (Sample {idx}) ===")
            print(f"Case: {case_dir.name}")
            print(f"Original volume shape: {volume_shape}")
            print(f"Crop size: {self.crop_size}")
            print(f"Crop coordinates: {crop_coords}")
            print(f"Available modalities: {available_modalities}")
            print(f"=== End Dataset Debug Info ===\n")
        
        # Extract crops and normalize each modality
        cropped_volumes = {}
        for modality in self.modalities:
            volume = modality_volumes.get(modality, np.zeros(volume_shape, dtype=np.float32))
            
            # Extract crop using same coordinates
            cropped = self._extract_crop(volume, crop_coords)
            
            # Professor's normalization: min-max to [0, 1]
            normalized = self._normalize_volume_0_1(cropped)
            
            cropped_volumes[modality] = torch.FloatTensor(normalized)
        
        # Select target modality for synthesis
        if self.phase == 'train' and len(available_modalities) > 0:
            target_modality = random.choice(available_modalities)
        elif len(available_modalities) > 0:
            target_modality = available_modalities[0]
        else:
            target_modality = self.modalities[0]
        
        target_idx = self.modalities.index(target_modality)
        target_volume = cropped_volumes[target_modality]
        
        # Create input with target modality masked (set to zero)
        input_modalities = torch.stack([cropped_volumes[mod] for mod in self.modalities])
        input_modalities[target_idx] = torch.zeros_like(input_modalities[target_idx])
        
        # Validation
        assert input_modalities.shape == (4, *self.crop_size)
        assert target_volume.shape == self.crop_size
        assert torch.all(input_modalities >= 0) and torch.all(input_modalities <= 1)
        assert torch.all(target_volume >= 0) and torch.all(target_volume <= 1)
        
        return {
            'input': input_modalities,      # [4, H, W, D] with target masked, range [0,1]
            'target': target_volume,        # [H, W, D] target modality, range [0,1]
            'target_idx': target_idx,
            'case_name': case_dir.name,
            'target_modality': target_modality,
            'available_modalities': available_modalities,
            'crop_coords': crop_coords
        }