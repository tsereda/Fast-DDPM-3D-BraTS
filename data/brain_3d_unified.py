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
    def __init__(self, data_root, phase='train', volume_size=(96, 96, 96), 
                 min_input_modalities=3):
        self.data_root = Path(data_root)
        self.phase = phase
        self.volume_size = tuple(volume_size)
        self.min_input_modalities = min_input_modalities
        
        # BraTS modalities - more comprehensive patterns
        self.modalities = ['t1c', 't1n', 't2f', 't2w']
        self.modality_aliases = {
            't1c': ['t1c', 't1ce', 't1-ce', 't1_ce', 't1gd', 't1-gd', 't1_gd'],
            't1n': ['t1n', 't1', 't1native', 't1-native', 't1_native'],
            't2f': ['t2f', 't2flair', 't2-flair', 't2_flair', 'flair'],
            't2w': ['t2w', 't2', 't2-w', 't2_w']
        }
        
        # Find all case directories
        self.cases = self._find_cases()
        
        if len(self.cases) == 0:
            raise ValueError(f"No valid cases found in {data_root}")
        
        logger.info(f"Found {len(self.cases)} cases for {phase} phase")
    
    def _find_cases(self):
        """Find all valid BraTS cases with improved pattern matching"""
        cases = []
        
        if not self.data_root.exists():
            raise FileNotFoundError(f"Data root {self.data_root} does not exist")
        
        # Look for case directories with BraTS pattern
        for item in self.data_root.iterdir():
            if item.is_dir():
                # Check if directory contains NIfTI files
                nii_files = list(item.glob('*.nii*'))
                if len(nii_files) >= self.min_input_modalities:
                    # More robust check for BraTS modality patterns
                    found_modalities = self._count_modalities_in_case(item)
                    if found_modalities >= self.min_input_modalities:
                        cases.append(item)
                        logger.debug(f"Found case {item.name} with {found_modalities} modalities")
        
        # Sort for reproducibility
        cases = sorted(cases, key=lambda x: x.name)
        
        if len(cases) > 0:
            logger.info(f"Sample cases: {[c.name for c in cases[:3]]}")
            # Log files in first case for debugging
            first_case_files = list(cases[0].glob('*.nii*'))
            logger.info(f"Files in first case: {[f.name for f in first_case_files]}")
        
        return cases
    
    def _count_modalities_in_case(self, case_dir):
        """Count how many modalities are available in a case directory"""
        found_count = 0
        all_files = list(case_dir.glob('*.nii*'))
        
        for modality in self.modalities:
            if self._find_modality_file(case_dir, modality):
                found_count += 1
        
        return found_count
    
    def _load_volume(self, filepath):
        """Load and preprocess a NIfTI volume with better error handling"""
        try:
            nii = nib.load(str(filepath))
            volume = nii.get_fdata().astype(np.float32)
            
            # Resize to target volume size
            volume = self._resize_volume(volume, self.volume_size)
            
            # Normalize to [-1, 1] with improved stability
            volume = self._normalize_volume(volume)
            
            return torch.FloatTensor(volume)
            
        except Exception as e:
            logger.error(f"Error loading {filepath}: {e}")
            # Return zero volume as fallback
            return torch.zeros(self.volume_size, dtype=torch.float32)
    
    def _resize_volume(self, volume, target_size):
        """Resize volume to target size using interpolation"""
        if volume.shape == target_size:
            return volume
        
        try:
            from scipy.ndimage import zoom
            
            current_size = volume.shape
            zoom_factors = [t/c for t, c in zip(target_size, current_size)]
            
            resized = zoom(volume, zoom_factors, order=1, mode='nearest')
            return resized
        except ImportError:
            logger.warning("scipy not available, using crop/pad fallback")
            return self._crop_or_pad(volume, target_size)
    
    def _crop_or_pad(self, volume, target_size):
        """Simple crop or pad to target size"""
        result = np.zeros(target_size, dtype=volume.dtype)
        
        # Calculate slices for cropping/copying
        slices_src = []
        slices_dst = []
        
        for i, (src_dim, dst_dim) in enumerate(zip(volume.shape, target_size)):
            if src_dim > dst_dim:
                # Crop source
                start = (src_dim - dst_dim) // 2
                slices_src.append(slice(start, start + dst_dim))
                slices_dst.append(slice(None))
            else:
                # Pad destination
                start = (dst_dim - src_dim) // 2
                slices_src.append(slice(None))
                slices_dst.append(slice(start, start + src_dim))
        
        # Copy data
        result[tuple(slices_dst)] = volume[tuple(slices_src)]
        
        return result
    
    def _normalize_volume(self, volume):
        """Normalize volume to [-1, 1] range with improved stability"""
        # Handle zero volumes
        if not np.any(volume > 0):
            return np.zeros_like(volume)
        
        # Get valid (non-zero) voxels
        valid_voxels = volume[volume > 0]
        if len(valid_voxels) == 0:
            return np.zeros_like(volume)
        
        # Clip outliers (1st and 99th percentile)
        p1, p99 = np.percentile(valid_voxels, [1, 99])
        
        # Check for valid range
        if p99 <= p1:
            logger.warning("Invalid intensity range detected, using zero volume")
            return np.zeros_like(volume)
        
        # Clip and normalize
        volume = np.clip(volume, p1, p99)
        # Add small epsilon to prevent numerical issues
        volume = 2 * (volume - p1) / (p99 - p1 + 1e-8) - 1
        
        # Final check for NaN values
        if np.any(np.isnan(volume)):
            logger.warning("NaN values detected in normalized volume, using zero volume")
            return np.zeros_like(volume)
        
        return volume
    
    def _find_modality_file(self, case_dir, modality):
        """Find file for a specific modality with comprehensive pattern matching"""
        # Get all possible aliases for this modality
        aliases = self.modality_aliases.get(modality, [modality])
        
        # Try each alias with different patterns
        for alias in aliases:
            patterns = [
                f'*{alias}*.nii*',
                f'*-{alias}.nii*',
                f'*_{alias}.nii*',
                f'*{alias.upper()}*.nii*',
                f'**/*{alias}*.nii*',  # Search subdirectories
                f'*{alias}*.*',  # Any extension
            ]
            
            for pattern in patterns:
                files = list(case_dir.glob(pattern))
                if files:
                    # Return the first match, preferring .nii.gz over .nii
                    files.sort(key=lambda x: (x.suffix != '.gz', x.name))
                    logger.debug(f"Found {modality} file: {files[0].name}")
                    return files[0]
        
        # If no match found, log available files for debugging
        all_files = list(case_dir.glob('*.nii*'))
        logger.debug(f"Could not find {modality} in {case_dir.name}")
        logger.debug(f"Available files: {[f.name for f in all_files]}")
        return None
    
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
                try:
                    modality_data[modality] = self._load_volume(modality_file)
                    available_modalities.append(modality)
                    logger.debug(f"Successfully loaded {modality} from {case_dir.name}")
                except Exception as e:
                    logger.warning(f"Failed to load {modality} from {case_dir.name}: {e}")
                    # Use zeros for failed loads
                    modality_data[modality] = torch.zeros(self.volume_size, dtype=torch.float32)
            else:
                logger.warning(f"Missing {modality} in {case_dir.name}")
                # Use zeros for missing modalities
                modality_data[modality] = torch.zeros(self.volume_size, dtype=torch.float32)
        
        # Check if we have enough modalities
        if len(available_modalities) < self.min_input_modalities:
            logger.warning(f"Case {case_dir.name} has only {len(available_modalities)} modalities")
        
        # Stack all modalities
        all_modalities = torch.stack([modality_data[mod] for mod in self.modalities])
        
        # Select target modality
        if self.phase == 'train':
            # Random training: select target from available modalities
            if len(available_modalities) > 0:
                target_modality = random.choice(available_modalities)
                target_idx = self.modalities.index(target_modality)
            else:
                # Fallback to first modality
                target_idx = 0
                target_modality = self.modalities[0]
        else:
            # For validation/test: use last available modality as target
            if len(available_modalities) > 0:
                target_modality = available_modalities[-1]
                target_idx = self.modalities.index(target_modality)
            else:
                target_idx = 0
                target_modality = self.modalities[0]
        
        # Get target volume
        target_volume = all_modalities[target_idx]
        
        # Create input with target masked (set to zero)
        input_modalities = all_modalities.clone()
        input_modalities[target_idx] = torch.zeros_like(input_modalities[target_idx])
        
        # Validate shapes and check for NaN values
        assert input_modalities.shape == (4, *self.volume_size), \
            f"Input shape mismatch: {input_modalities.shape} vs expected {(4, *self.volume_size)}"
        assert target_volume.shape == self.volume_size, \
            f"Target shape mismatch: {target_volume.shape} vs expected {self.volume_size}"
        
        # Check for NaN values
        if torch.any(torch.isnan(input_modalities)) or torch.any(torch.isnan(target_volume)):
            logger.warning(f"NaN values detected in case {case_dir.name}, using zero tensors")
            input_modalities = torch.zeros((4, *self.volume_size), dtype=torch.float32)
            target_volume = torch.zeros(self.volume_size, dtype=torch.float32)
        
        return {
            'input': input_modalities,  # [4, H, W, D]
            'target': target_volume,    # [H, W, D]
            'target_idx': target_idx,
            'case_name': case_dir.name,
            'target_modality': target_modality,
            'available_modalities': available_modalities
        }


def test_dataset():
    """Test the dataset with improved diagnostics"""
    print("Testing BraTS3DUnifiedDataset...")
    
    import sys
    if len(sys.argv) > 1:
        data_path = sys.argv[1]
        print(f"Testing with data at: {data_path}")
        
        # Enable debug logging
        logging.basicConfig(level=logging.DEBUG)
        
        try:
            dataset = BraTS3DUnifiedDataset(
                data_root=data_path,
                phase='train',
                volume_size=(64, 64, 64)
            )
            print(f"✅ Dataset created with {len(dataset)} cases")
            
            # Test loading multiple samples
            if len(dataset) > 0:
                for i in range(min(3, len(dataset))):
                    sample = dataset[i]
                    print(f"✅ Sample {i} loaded successfully")
                    print(f"  Input shape: {sample['input'].shape}")
                    print(f"  Target shape: {sample['target'].shape}")
                    print(f"  Target idx: {sample['target_idx']}")
                    print(f"  Case name: {sample['case_name']}")
                    print(f"  Target modality: {sample['target_modality']}")
                    print(f"  Available modalities: {sample['available_modalities']}")
                    print(f"  Input range: [{sample['input'].min():.3f}, {sample['input'].max():.3f}]")
                    print(f"  Target range: [{sample['target'].min():.3f}, {sample['target'].max():.3f}]")
                    print()
                    
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("Provide data path as argument to test with real data")
        print("Example: python brain_3d_unified.py /path/to/brats/data")


if __name__ == "__main__":
    test_dataset()