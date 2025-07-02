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
    Simplified BraTS 3D dataset for 4->1 modality synthesis
    Loads 4 modalities, randomly masks one as target for synthesis
    """
    
    def __init__(self, data_root, phase='train', volume_size=(64, 64, 64), 
                 min_input_modalities=3):
        self.data_root = Path(data_root)
        self.phase = phase
        self.volume_size = tuple(volume_size)
        self.min_input_modalities = min_input_modalities
        
        # Standard BraTS modalities
        self.modalities = ['t1n', 't1c', 't2w', 't2f']
        
        # Find all case directories
        self.cases = self._find_cases()
        
        if len(self.cases) == 0:
            raise ValueError(f"No valid cases found in {data_root}")
        
        logger.info(f"Found {len(self.cases)} cases for {phase} phase")
    
    def _find_cases(self):
        """Find all valid BraTS case directories"""
        cases = []
        
        if not self.data_root.exists():
            raise FileNotFoundError(f"Data root {self.data_root} does not exist")
        
        # Look for directories containing NIfTI files
        for item in self.data_root.iterdir():
            if item.is_dir():
                # Count available modalities
                available_count = sum(1 for mod in self.modalities 
                                    if self._find_modality_file(item, mod) is not None)
                
                if available_count >= self.min_input_modalities:
                    cases.append(item)
                    logger.debug(f"Found case {item.name} with {available_count} modalities")
        
        return sorted(cases, key=lambda x: x.name)
    
    def _find_modality_file(self, case_dir, modality):
        """Simple pattern matching for BraTS modality files"""
        # Try common patterns for this modality
        patterns = [
            f'*{modality}*.nii*',
            f'*{modality.upper()}*.nii*',
            f'*-{modality}.nii*',
            f'*_{modality}.nii*'
        ]
        
        for pattern in patterns:
            files = list(case_dir.glob(pattern))
            if files:
                # Prefer .nii.gz over .nii
                files.sort(key=lambda x: (x.suffix != '.gz', x.name))
                return files[0]
        
        return None
    
    def _load_volume(self, filepath):
        """Load and preprocess a NIfTI volume"""
        try:
            nii = nib.load(str(filepath))
            volume = nii.get_fdata().astype(np.float32)
            
            # Resize to target volume size
            volume = self._resize_volume(volume, self.volume_size)
            
            # Normalize to [-1, 1] range
            volume = self._normalize_volume(volume)
            
            return torch.FloatTensor(volume)
            
        except Exception as e:
            logger.error(f"Error loading {filepath}: {e}")
            # Return zero volume as fallback
            return torch.zeros(self.volume_size, dtype=torch.float32)
    
    def _resize_volume(self, volume, target_size):
        """Resize volume using interpolation or crop/pad"""
        if volume.shape == target_size:
            return volume
        
        try:
            from scipy.ndimage import zoom
            zoom_factors = [t/c for t, c in zip(target_size, volume.shape)]
            return zoom(volume, zoom_factors, order=1, mode='nearest')
        except ImportError:
            # Fallback to crop/pad
            return self._crop_or_pad(volume, target_size)
    
    def _crop_or_pad(self, volume, target_size):
        """Simple crop or pad to target size"""
        result = np.zeros(target_size, dtype=volume.dtype)
        
        # Calculate copy regions
        slices_src = []
        slices_dst = []
        
        for src_dim, dst_dim in zip(volume.shape, target_size):
            if src_dim > dst_dim:
                # Crop: take center region
                start = (src_dim - dst_dim) // 2
                slices_src.append(slice(start, start + dst_dim))
                slices_dst.append(slice(None))
            else:
                # Pad: place in center
                start = (dst_dim - src_dim) // 2
                slices_src.append(slice(None))
                slices_dst.append(slice(start, start + src_dim))
        
        result[tuple(slices_dst)] = volume[tuple(slices_src)]
        return result
    
    def _normalize_volume(self, volume):
        """Simplified, robust normalization to [-1, 1] range"""
        # Handle zero or constant volumes
        if not np.any(volume > 0):
            return np.zeros_like(volume)
        
        # Simple min-max normalization for stability
        vmin = volume.min()
        vmax = volume.max()
        
        if vmax <= vmin:
            return np.zeros_like(volume)
        
        # Normalize to [0, 1] first
        volume = (volume - vmin) / (vmax - vmin + 1e-8)
        
        # Scale to [-1, 1]
        volume = 2 * volume - 1
        
        # Final validation
        if np.any(np.isnan(volume)) or np.any(np.isinf(volume)):
            logger.warning("NaN/Inf values detected after normalization, using zero volume")
            return np.zeros_like(volume)
        
        # Ensure range is correct
        volume = np.clip(volume, -1.0, 1.0)
        
        return volume
    
    def __len__(self):
        return len(self.cases)
    
    def __getitem__(self, idx):
        case_dir = self.cases[idx]
        
        # Load all modalities
        modality_data = {}
        available_modalities = []
        
        for modality in self.modalities:
            modality_file = self._find_modality_file(case_dir, modality)
            
            if modality_file:
                volume = self._load_volume(modality_file)
                modality_data[modality] = volume
                available_modalities.append(modality)
            else:
                # Use zero volume for missing modalities
                modality_data[modality] = torch.zeros(self.volume_size, dtype=torch.float32)
        
        # Create input tensor [4, H, W, D] - all modalities
        all_modalities = torch.stack([modality_data[mod] for mod in self.modalities])
        
        # Select target modality for synthesis
        if self.phase == 'train' and len(available_modalities) > 0:
            # Random target selection during training
            target_modality = random.choice(available_modalities)
        elif len(available_modalities) > 0:
            # Use first available modality for validation/test
            target_modality = available_modalities[0]
        else:
            # Fallback
            target_modality = self.modalities[0]
        
        target_idx = self.modalities.index(target_modality)
        target_volume = all_modalities[target_idx]
        
        # Create input with target modality masked (set to zero)
        input_modalities = all_modalities.clone()
        input_modalities[target_idx] = torch.zeros_like(input_modalities[target_idx])
        
        # Validation
        assert input_modalities.shape == (4, *self.volume_size)
        assert target_volume.shape == self.volume_size
        
        return {
            'input': input_modalities,      # [4, H, W, D] with target masked
            'target': target_volume,        # [H, W, D] target modality
            'target_idx': target_idx,       # Which modality is being synthesized
            'case_name': case_dir.name,
            'target_modality': target_modality,
            'available_modalities': available_modalities
        }


if __name__ == "__main__":
    # Simple test
    import sys
    if len(sys.argv) > 1:
        dataset = BraTS3DUnifiedDataset(sys.argv[1], volume_size=(64, 64, 64))
        print(f"Dataset created with {len(dataset)} cases")
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"Sample shapes - Input: {sample['input'].shape}, Target: {sample['target'].shape}")
            print(f"Target modality: {sample['target_modality']}")
    else:
        print("Usage: python brain_3d_unified.py /path/to/brats/data")