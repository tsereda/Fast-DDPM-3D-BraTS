"""
3D Unified BraTS Dataset for Fast-DDPM
Supports 4→4 unified training approach
"""

import os
import torch
from torch.utils.data import Dataset
import nibabel as nib
import numpy as np
from pathlib import Path
import random
import logging

# Setup logging
logger = logging.getLogger(__name__)


class BraTS3DUnifiedDataset(Dataset):
    """
    Unified BraTS 3D Dataset for 4→4 modality synthesis
    """
    
    def __init__(self, data_root, phase='train', volume_size=(96, 96, 96), 
                 min_input_modalities=1, max_input_modalities=3):
        self.data_root = Path(data_root)
        self.phase = phase
        self.volume_size = tuple(volume_size)  # Ensure it's a tuple
        self.min_input_modalities = min_input_modalities
        self.max_input_modalities = max_input_modalities
        
        # BraTS modalities
        self.modalities = ['t1c', 't1n', 't2f', 't2w']
        
        # Find all case directories
        self.cases = self._find_cases()
        
        if len(self.cases) == 0:
            raise ValueError(f"No valid cases found in {data_root}")
        
        logger.info(f"Found {len(self.cases)} cases for {phase} phase")
    
    def _find_cases(self):
        """Find all valid BraTS cases with better pattern matching"""
        cases = []
        
        if not self.data_root.exists():
            logger.warning(f"Data root {self.data_root} does not exist")
            return cases
        
        # Try multiple patterns to find cases
        # Pattern 1: Direct case directories with NIfTI files
        for item in self.data_root.iterdir():
            if item.is_dir():
                # Check for BraTS naming patterns
                nii_files = list(item.glob('*.nii*'))
                if len(nii_files) >= 3:  # At least 3 modalities
                    # Check if files match BraTS patterns
                    has_modalities = any('-t1n' in f.name.lower() or 
                                       '-t1c' in f.name.lower() or
                                       '-t2w' in f.name.lower() or 
                                       '-t2f' in f.name.lower() 
                                       for f in nii_files)
                    if has_modalities:
                        cases.append(item)
                        continue
                
                # Also check for generic patterns
                modality_count = 0
                for mod in self.modalities:
                    if any(mod in f.name.lower() for f in nii_files):
                        modality_count += 1
                if modality_count >= 3:
                    cases.append(item)
        
        # Pattern 2: Nested structure (e.g., TrainingData/BraTS-xxx/)
        if len(cases) == 0:
            for subdir in self.data_root.rglob('*'):
                if subdir.is_dir() and ('BraTS' in subdir.name or 'UPENN' in subdir.name):
                    nii_files = list(subdir.glob('*.nii*'))
                    if len(nii_files) >= 3:
                        cases.append(subdir)
        
        # Sort cases for reproducibility
        cases = sorted(cases, key=lambda x: x.name)
        
        # Log some information
        if len(cases) > 0:
            logger.info(f"Sample cases: {[c.name for c in cases[:3]]}")
            # Check modalities in first case
            first_case_files = list(cases[0].glob('*.nii*'))
            logger.info(f"Files in first case: {[f.name for f in first_case_files[:5]]}")
        
        return cases
    
    def _load_volume(self, filepath):
        """Load and preprocess a NIfTI volume"""
        try:
            nii = nib.load(str(filepath))
            volume = nii.get_fdata()
            
            # Handle different data types
            volume = volume.astype(np.float32)
            
            # Resize to target volume size
            volume = self._resize_volume(volume, self.volume_size)
            
            # Normalize to [-1, 1]
            volume = self._normalize_volume(volume)
            
            return torch.FloatTensor(volume)
            
        except Exception as e:
            logger.error(f"Error loading {filepath}: {e}")
            # Return zeros if loading fails
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
        except:
            # Fallback: simple cropping/padding
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
        # Clip outliers (1st and 99th percentile)
        p1, p99 = np.percentile(volume[volume > 0], [1, 99]) if np.any(volume > 0) else (0, 1)
        
        if p99 > p1:
            volume = np.clip(volume, p1, p99)
            # Normalize to [-1, 1]
            volume = 2 * (volume - p1) / (p99 - p1) - 1
        else:
            # If no valid range, return zeros
            volume = np.zeros_like(volume)
        
        return volume
    
    def _find_modality_file(self, case_dir, modality):
        """Find file for a specific modality with flexible pattern matching"""
        # Try exact BraTS pattern first
        patterns = [
            f'*-{modality}.nii*',
            f'*_{modality}.nii*',
            f'*{modality}*.nii*'
        ]
        
        for pattern in patterns:
            files = list(case_dir.glob(pattern))
            if files:
                return files[0]
        
        # If not found, try case-insensitive search
        all_files = list(case_dir.glob('*.nii*'))
        for f in all_files:
            if modality.lower() in f.name.lower():
                return f
        
        return None
    
    def __len__(self):
        return len(self.cases)
    
    def __getitem__(self, idx):
        case_dir = self.cases[idx]
        
        # Load all modalities
        modality_data = {}
        available_modalities = []
        
        for modality in self.modalities:
            # Find file with this modality
            modality_file = self._find_modality_file(case_dir, modality)
            
            if modality_file:
                modality_data[modality] = self._load_volume(modality_file)
                available_modalities.append(modality)
            else:
                # Missing modality - use zeros
                modality_data[modality] = torch.zeros(self.volume_size, dtype=torch.float32)
        
        # Check if we have enough modalities
        if len(available_modalities) < 2:
            logger.warning(f"Case {case_dir.name} has only {len(available_modalities)} modalities")
        
        # For unified 4→4 training:
        # Stack all modalities
        all_modalities = torch.stack([modality_data[mod] for mod in self.modalities])
        
        if self.phase == 'train':
            # Random training: select target from available modalities
            if len(available_modalities) > 0:
                target_modality = random.choice(available_modalities)
                target_idx = self.modalities.index(target_modality)
            else:
                # Fallback if no modalities available
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
        
        # Create input with target masked
        input_modalities = all_modalities.clone()
        input_modalities[target_idx] = torch.zeros_like(input_modalities[target_idx])
        
        # Validate shapes before returning
        assert input_modalities.shape == (4, *self.volume_size), \
            f"Input shape mismatch: {input_modalities.shape} vs expected {(4, *self.volume_size)}"
        assert target_volume.shape == self.volume_size, \
            f"Target shape mismatch: {target_volume.shape} vs expected {self.volume_size}"
        
        return {
            'input': input_modalities,  # [4, H, W, D]
            'target': target_volume,    # [H, W, D]
            'case_name': case_dir.name,
            'target_modality': target_modality,
            'available_modalities': available_modalities
        }


# Test function
def test_dataset():
    """Test the dataset with dummy data"""
    print("Testing BraTS3DUnifiedDataset...")
    
    # This will fail with real data but shows the structure
    try:
        dataset = BraTS3DUnifiedDataset(
            data_root="/dummy/path",
            phase='train',
            volume_size=(64, 64, 64)
        )
        print(f"Dataset created with {len(dataset)} cases")
    except Exception as e:
        print(f"Expected error (no real data): {e}")
    
    # Test with a real path if provided
    import sys
    if len(sys.argv) > 1:
        data_path = sys.argv[1]
        print(f"\nTesting with real data at: {data_path}")
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
                print(f"  Case name: {sample['case_name']}")
                print(f"  Target modality: {sample['target_modality']}")
                print(f"  Available modalities: {sample['available_modalities']}")
        except Exception as e:
            print(f"❌ Error with real data: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    test_dataset()