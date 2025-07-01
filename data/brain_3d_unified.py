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
        
        # BraTS modalities
        self.modalities = ['t1c', 't1n', 't2f', 't2w']
        
        # Find all case directories
        self.cases = self._find_cases()
        
        if len(self.cases) == 0:
            raise ValueError(f"No valid cases found in {data_root}")
        
        logger.info(f"Found {len(self.cases)} cases for {phase} phase")
    
    def _find_cases(self):
        """Find all valid BraTS cases"""
        cases = []
        
        if not self.data_root.exists():
            raise FileNotFoundError(f"Data root {self.data_root} does not exist")
        
        # Look for case directories with BraTS pattern
        for item in self.data_root.iterdir():
            if item.is_dir():
                # Check if directory contains NIfTI files
                nii_files = list(item.glob('*.nii*'))
                if len(nii_files) >= self.min_input_modalities:
                    # Verify it has BraTS modality patterns
                    has_modalities = any(
                        any(mod in f.name.lower() for mod in self.modalities)
                        for f in nii_files
                    )
                    if has_modalities:
                        cases.append(item)
        
        # Sort for reproducibility
        cases = sorted(cases, key=lambda x: x.name)
        
        if len(cases) > 0:
            logger.info(f"Sample cases: {[c.name for c in cases[:3]]}")
            # Log files in first case for debugging
            first_case_files = list(cases[0].glob('*.nii*'))
            logger.info(f"Files in first case: {[f.name for f in first_case_files]}")
        
        return cases
    
    def _load_volume(self, filepath):
        """Load and preprocess a NIfTI volume"""
        try:
            nii = nib.load(str(filepath))
            volume = nii.get_fdata().astype(np.float32)
            
            # Resize to target volume size
            volume = self._resize_volume(volume, self.volume_size)
            
            # Normalize to [-1, 1]
            volume = self._normalize_volume(volume)
            
            return torch.FloatTensor(volume)
            
        except Exception as e:
            logger.error(f"Error loading {filepath}: {e}")
            raise
    
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
        """Normalize volume to [-1, 1] range"""
        # Handle zero volumes
        if not np.any(volume > 0):
            return np.zeros_like(volume)
        
        # Clip outliers (1st and 99th percentile)
        p1, p99 = np.percentile(volume[volume > 0], [1, 99])
        
        if p99 > p1:
            volume = np.clip(volume, p1, p99)
            # Normalize to [-1, 1]
            volume = 2 * (volume - p1) / (p99 - p1) - 1
        else:
            # If no valid range, return zeros
            volume = np.zeros_like(volume)
        
        return volume
    
    def _find_modality_file(self, case_dir, modality):
        """Find file for a specific modality"""
        # Try standard BraTS patterns
        patterns = [
            f'*{modality}*.nii*',
            f'*-{modality}.nii*',
            f'*_{modality}.nii*'
        ]
        
        for pattern in patterns:
            files = list(case_dir.glob(pattern))
            if files:
                return files[0]
        
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
        
        # Validate shapes
        assert input_modalities.shape == (4, *self.volume_size), \
            f"Input shape mismatch: {input_modalities.shape} vs expected {(4, *self.volume_size)}"
        assert target_volume.shape == self.volume_size, \
            f"Target shape mismatch: {target_volume.shape} vs expected {self.volume_size}"
        
        return {
            'input': input_modalities,  # [4, H, W, D]
            'target': target_volume,    # [H, W, D]
            'target_idx': target_idx,
            'case_name': case_dir.name,
            'target_modality': target_modality,
            'available_modalities': available_modalities
        }


def test_dataset():
    """Test the dataset"""
    print("Testing BraTS3DUnifiedDataset...")
    
    import sys
    if len(sys.argv) > 1:
        data_path = sys.argv[1]
        print(f"Testing with data at: {data_path}")
        try:
            dataset = BraTS3DUnifiedDataset(
                data_root=data_path,
                phase='train',
                volume_size=(64, 64, 64)
            )
            print(f"✅ Dataset created with {len(dataset)} cases")
            
            # Test loading one sample
            if len(dataset) > 0:
                sample = dataset[0]
                print(f"✅ Sample loaded successfully")
                print(f"  Input shape: {sample['input'].shape}")
                print(f"  Target shape: {sample['target'].shape}")
                print(f"  Target idx: {sample['target_idx']}")
                print(f"  Case name: {sample['case_name']}")
                print(f"  Target modality: {sample['target_modality']}")
                print(f"  Available modalities: {sample['available_modalities']}")
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("Provide data path as argument to test with real data")


if __name__ == "__main__":
    test_dataset()