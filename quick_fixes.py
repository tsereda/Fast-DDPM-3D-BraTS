#!/usr/bin/env python3
"""
Quick fix script to address remaining issues in Fast-DDPM-3D-BraTS
"""

import os
import sys
from pathlib import Path

def fix_sg_noise_estimation_loss():
    """Add the missing sg_noise_estimation_loss function"""
    losses_file = Path("functions/losses.py")
    
    if not losses_file.exists():
        print("❌ losses.py not found")
        return False
    
    # Read current content
    with open(losses_file, 'r') as f:
        content = f.read()
    
    # Check if function already exists
    if 'def sg_noise_estimation_loss' in content:
        print("✅ sg_noise_estimation_loss already exists")
        return True
    
    # Add the function
    sg_function = '''

def sg_noise_estimation_loss(model, x_available, x_target, t, e, betas):
    """
    Simple noise estimation loss (alias for fast_ddpm_loss with fixed variance)
    For backward compatibility with existing training scripts
    """
    return fast_ddpm_loss(model, x_available, x_target, t, e, betas, var_type='fixed')


def combined_loss(model, x_available, x_target, t, e, betas, alpha=0.8):
    """
    Combined loss function with multiple components
    """
    # Main diffusion loss
    main_loss = fast_ddpm_loss(model, x_available, x_target, t, e, betas, var_type='fixed')
    
    # L1 loss for additional regularization
    with torch.no_grad():
        # Get model prediction
        sqrt_alphas_cumprod = torch.sqrt(1.0 - betas).cumprod(dim=0)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(betas.cumsum(dim=0))
        
        a_t = sqrt_alphas_cumprod[t].view(-1, 1, 1, 1, 1)
        sqrt_one_minus_a_t = sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1, 1)
        
        x_noisy = x_target * a_t + e * sqrt_one_minus_a_t
        
        model_input = x_available.clone()
        model_input[:, 0:1] = x_noisy
        
        pred = model(model_input, t.float())
        if isinstance(pred, tuple):
            pred = pred[0]
        
        # Predict x0
        x0_pred = (x_noisy - sqrt_one_minus_a_t * pred) / a_t
        l1_loss = torch.nn.functional.l1_loss(x0_pred, x_target)
    
    return alpha * main_loss + (1 - alpha) * l1_loss
'''
    
    # Append to file
    with open(losses_file, 'a') as f:
        f.write(sg_function)
    
    print("✅ Added sg_noise_estimation_loss and combined_loss functions")
    return True


def fix_brain_3d_unified_import():
    """Create a basic brain_3d_unified.py if it doesn't exist"""
    data_file = Path("data/brain_3d_unified.py")
    
    if data_file.exists():
        print("✅ brain_3d_unified.py already exists")
        return True
    
    # Create the data directory if it doesn't exist
    data_file.parent.mkdir(exist_ok=True)
    
    # Create a basic dataset class
    dataset_code = '''"""
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


class BraTS3DUnifiedDataset(Dataset):
    """
    Unified BraTS 3D Dataset for 4→4 modality synthesis
    """
    
    def __init__(self, data_root, phase='train', volume_size=(96, 96, 96), 
                 min_input_modalities=1, max_input_modalities=3):
        self.data_root = Path(data_root)
        self.phase = phase
        self.volume_size = volume_size
        self.min_input_modalities = min_input_modalities
        self.max_input_modalities = max_input_modalities
        
        # BraTS modalities
        self.modalities = ['t1c', 't1n', 't2f', 't2w']
        
        # Find all case directories
        self.cases = self._find_cases()
        print(f"Found {len(self.cases)} cases for {phase} phase")
    
    def _find_cases(self):
        """Find all valid BraTS cases"""
        cases = []
        
        if not self.data_root.exists():
            print(f"Warning: Data root {self.data_root} does not exist")
            return cases
        
        for case_dir in self.data_root.iterdir():
            if case_dir.is_dir():
                # Check if this directory has NIfTI files
                nii_files = list(case_dir.glob('*.nii*'))
                if len(nii_files) >= 4:  # At least 4 modalities
                    cases.append(case_dir)
        
        return cases
    
    def _load_volume(self, filepath):
        """Load and preprocess a NIfTI volume"""
        try:
            nii = nib.load(str(filepath))
            volume = nii.get_fdata()
            
            # Resize to target volume size
            volume = self._resize_volume(volume, self.volume_size)
            
            # Normalize to [-1, 1]
            volume = self._normalize_volume(volume)
            
            return torch.FloatTensor(volume)
            
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            # Return zeros if loading fails
            return torch.zeros(self.volume_size, dtype=torch.float32)
    
    def _resize_volume(self, volume, target_size):
        """Resize volume to target size using interpolation"""
        from scipy.ndimage import zoom
        
        current_size = volume.shape
        zoom_factors = [t/c for t, c in zip(target_size, current_size)]
        
        try:
            resized = zoom(volume, zoom_factors, order=1)  # Linear interpolation
            return resized
        except:
            # Fallback: simple cropping/padding
            return self._crop_or_pad(volume, target_size)
    
    def _crop_or_pad(self, volume, target_size):
        """Simple crop or pad to target size"""
        result = np.zeros(target_size, dtype=volume.dtype)
        
        # Calculate slices for cropping/copying
        slices = []
        for i, (curr, target) in enumerate(zip(volume.shape, target_size)):
            if curr >= target:
                # Crop
                start = (curr - target) // 2
                slices.append(slice(start, start + target))
            else:
                # Will pad
                slices.append(slice(None))
        
        # Copy data
        source_slices = tuple(slices)
        
        # Calculate destination slices for padding
        dest_slices = []
        for i, (curr, target) in enumerate(zip(volume.shape, target_size)):
            if curr < target:
                start = (target - curr) // 2
                dest_slices.append(slice(start, start + curr))
            else:
                dest_slices.append(slice(None))
        
        if len(dest_slices) == len(source_slices):
            try:
                result[tuple(dest_slices)] = volume[source_slices]
            except:
                # Fallback
                result[:min(volume.shape[0], target_size[0]),
                       :min(volume.shape[1], target_size[1]),
                       :min(volume.shape[2], target_size[2])] = volume[:target_size[0], :target_size[1], :target_size[2]]
        
        return result
    
    def _normalize_volume(self, volume):
        """Normalize volume to [-1, 1] range"""
        # Clip outliers
        p1, p99 = np.percentile(volume, [1, 99])
        volume = np.clip(volume, p1, p99)
        
        # Normalize to [-1, 1]
        if p99 > p1:
            volume = 2 * (volume - p1) / (p99 - p1) - 1
        else:
            volume = np.zeros_like(volume)
        
        return volume
    
    def __len__(self):
        return len(self.cases)
    
    def __getitem__(self, idx):
        case_dir = self.cases[idx]
        
        # Load all modalities
        modality_data = {}
        for modality in self.modalities:
            # Find file with this modality
            files = list(case_dir.glob(f'*{modality}*.nii*'))
            if files:
                modality_data[modality] = self._load_volume(files[0])
            else:
                # Missing modality - use zeros
                modality_data[modality] = torch.zeros(self.volume_size, dtype=torch.float32)
        
        # For unified 4→4 training:
        # - Input: 4 channels (some may be masked/missing)
        # - Target: 1 channel (randomly selected from available)
        
        # Stack all modalities
        all_modalities = torch.stack([modality_data[mod] for mod in self.modalities])
        
        # Randomly select target modality
        target_idx = random.randint(0, 3)
        target_modality = all_modalities[target_idx:target_idx+1]  # Keep as [1, H, W, D]
        
        # Create input with target modality masked (set to zeros or noise)
        input_modalities = all_modalities.clone()
        input_modalities[target_idx] = torch.zeros_like(input_modalities[target_idx])
        
        return {
            'input': input_modalities,  # [4, H, W, D]
            'target': target_modality.squeeze(0),  # [H, W, D]
            'case_name': case_dir.name,
            'target_modality': self.modalities[target_idx]
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


if __name__ == "__main__":
    test_dataset()
'''
    
    with open(data_file, 'w') as f:
        f.write(dataset_code)
    
    print("✅ Created brain_3d_unified.py")
    return True


def fix_missing_directories():
    """Create missing directories"""
    dirs_to_create = [
        "experiments",
        "logs", 
        "checkpoints",
        "outputs"
    ]
    
    for dir_name in dirs_to_create:
        dir_path = Path(dir_name)
        if not dir_path.exists():
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"✅ Created directory: {dir_name}")
        else:
            print(f"✅ Directory already exists: {dir_name}")
    
    return True


def update_gitignore():
    """Update .gitignore with additional patterns"""
    gitignore_path = Path(".gitignore")
    
    additional_patterns = [
        "",
        "# Experiments and outputs",
        "experiments/",
        "logs/",
        "checkpoints/", 
        "outputs/",
        "*.log",
        "",
        "# Temporary files",
        "/tmp/",
        "dummy_*",
        "",
        "# IDE files",
        ".vscode/",
        ".idea/",
        "",
        "# OS files", 
        ".DS_Store",
        "Thumbs.db"
    ]
    
    try:
        # Read existing content
        if gitignore_path.exists():
            with open(gitignore_path, 'r') as f:
                existing_content = f.read()
        else:
            existing_content = ""
        
        # Add new patterns
        new_content = existing_content
        for pattern in additional_patterns:
            if pattern not in existing_content:
                new_content += pattern + "\\n"
        
        # Write back
        with open(gitignore_path, 'w') as f:
            f.write(new_content)
        
        print("✅ Updated .gitignore")
        return True
        
    except Exception as e:
        print(f"⚠️  Could not update .gitignore: {e}")
        return False


def main():
    """Run all fixes"""
    print("🔧 Running Quick Fixes for Fast-DDPM-3D-BraTS")
    print("="*60)
    
    fixes = [
        ("Missing loss functions", fix_sg_noise_estimation_loss),
        ("Missing dataset class", fix_brain_3d_unified_import), 
        ("Missing directories", fix_missing_directories),
        ("Update .gitignore", update_gitignore)
    ]
    
    success_count = 0
    
    for fix_name, fix_func in fixes:
        print(f"\\n🔧 Applying: {fix_name}")
        try:
            if fix_func():
                success_count += 1
        except Exception as e:
            print(f"❌ {fix_name} failed: {e}")
    
    print(f"\\n📊 Applied {success_count}/{len(fixes)} fixes successfully")
    
    if success_count == len(fixes):
        print("✅ All fixes applied successfully!")
    else:
        print("⚠️  Some fixes failed, but the main issues should be resolved")
    
    print("\\n🎯 Next Steps:")
    print("1. Run the comprehensive test: python test_comprehensive.py")
    print("2. Try training with: python scripts/train_3d.py --data_root /path/to/brats/data")
    print("3. Monitor GPU memory usage during training")


if __name__ == "__main__":
    main()
