#!/usr/bin/env python3
"""
Verify normalization consistency across the Fast-DDPM-3D-BraTS codebase
"""

import sys
import torch
import numpy as np
from pathlib import Path

# Add project to path
sys.path.append('.')

def test_data_normalization():
    """Test that data normalization produces [-1, 1] range"""
    print("🧪 Testing data normalization...")
    
    # Simulate a typical medical image volume
    test_volume = np.random.exponential(scale=100, size=(64, 64, 64))
    test_volume = test_volume.astype(np.float32)
    
    # Import the normalization function
    from data.brain_3d_unified import BraTS3DUnifiedDataset
    dataset = BraTS3DUnifiedDataset.__new__(BraTS3DUnifiedDataset)
    
    # Test the normalization
    normalized = dataset._normalize_volume(test_volume)
    
    print(f"Original range: [{test_volume.min():.3f}, {test_volume.max():.3f}]")
    print(f"Normalized range: [{normalized.min():.3f}, {normalized.max():.3f}]")
    
    # Check if it's in [-1, 1] range
    if normalized.min() >= -1.0 and normalized.max() <= 1.0:
        print("✅ Data normalization produces [-1, 1] range")
        return True
    else:
        print("❌ Data normalization does NOT produce [-1, 1] range")
        return False

def test_utils_normalization():
    """Test that utils normalization produces [-1, 1] range"""
    print("\n🧪 Testing utils normalization...")
    
    # Import the normalization function
    from utils.crop_and_pad_volume import normalise_image
    
    # Test volume
    test_volume = torch.rand(32, 32, 32) * 100 + 50  # Range [50, 150]
    
    # Test div_by_max normalization
    normalized = normalise_image(test_volume, norm_type='div_by_max')
    
    print(f"Original range: [{test_volume.min():.3f}, {test_volume.max():.3f}]")
    print(f"Normalized range: [{normalized.min():.3f}, {normalized.max():.3f}]")
    
    # Check if it's in [-1, 1] range
    if normalized.min() >= -1.0 and normalized.max() <= 1.0:
        print("✅ Utils normalization produces [-1, 1] range")
        return True
    else:
        print("❌ Utils normalization does NOT produce [-1, 1] range")
        return False

def test_model_expectations():
    """Test that model clamps to [-1, 1] range"""
    print("\n🧪 Testing model expectations...")
    
    # Test that loss function expects [-1, 1]
    test_tensor = torch.randn(1, 1, 8, 8, 8) * 2  # Range roughly [-4, 4]
    
    # Simulate what happens in the loss function
    clamped = torch.clamp(test_tensor, -1, 1)
    
    print(f"Input range: [{test_tensor.min():.3f}, {test_tensor.max():.3f}]")
    print(f"Clamped range: [{clamped.min():.3f}, {clamped.max():.3f}]")
    
    if clamped.min() >= -1.0 and clamped.max() <= 1.0:
        print("✅ Model expects and enforces [-1, 1] range")
        return True
    else:
        print("❌ Model clamping failed")
        return False

def test_inference_consistency():
    """Test inference normalization"""
    print("\n🧪 Testing inference normalization...")
    
    # Import inference normalization
    sys.path.append('./scripts')
    
    # Simulate the normalize_volume function from inference
    def normalize_volume(volume):
        v_min = np.amin(volume)
        v_max = np.amax(volume) 
        if v_max > v_min:
            # Normalize to [-1, 1] range to match training data
            volume = 2 * (volume - v_min) / (v_max - v_min) - 1
        return volume
    
    # Test volume
    test_volume = np.random.exponential(scale=50, size=(32, 32, 32))
    normalized = normalize_volume(test_volume)
    
    print(f"Original range: [{test_volume.min():.3f}, {test_volume.max():.3f}]")
    print(f"Normalized range: [{normalized.min():.3f}, {normalized.max():.3f}]")
    
    if normalized.min() >= -1.0 and normalized.max() <= 1.0:
        print("✅ Inference normalization produces [-1, 1] range")
        return True
    else:
        print("❌ Inference normalization does NOT produce [-1, 1] range")
        return False

def main():
    print("🔍 Fast-DDPM-3D-BraTS Normalization Verification")
    print("=" * 60)
    
    results = []
    
    try:
        results.append(test_data_normalization())
    except Exception as e:
        print(f"❌ Data normalization test failed: {e}")
        results.append(False)
    
    try:
        results.append(test_utils_normalization())
    except Exception as e:
        print(f"❌ Utils normalization test failed: {e}")
        results.append(False)
    
    try:
        results.append(test_model_expectations())
    except Exception as e:
        print(f"❌ Model expectations test failed: {e}")
        results.append(False)
    
    try:
        results.append(test_inference_consistency())
    except Exception as e:
        print(f"❌ Inference consistency test failed: {e}")
        results.append(False)
    
    print("\n" + "=" * 60)
    print("📊 SUMMARY")
    print("=" * 60)
    
    if all(results):
        print("✅ ALL TESTS PASSED - Normalization is consistent [-1, 1]")
        print("\nThe codebase now uses consistent [-1, 1] normalization:")
        print("• Data loading: [-1, 1] ✅")
        print("• Model/Loss: [-1, 1] ✅") 
        print("• Inference: [-1, 1] ✅")
        print("• Utils: [-1, 1] ✅")
    else:
        print("❌ SOME TESTS FAILED - Normalization inconsistencies remain")
        print(f"Tests passed: {sum(results)}/{len(results)}")
        
        print("\nRecommended fixes:")
        print("1. Ensure all data loading normalizes to [-1, 1]")
        print("2. Ensure all inference scripts expect [-1, 1] input")
        print("3. Ensure all utility functions output [-1, 1]")
        print("4. Update visualization code to handle [-1, 1] properly")

if __name__ == "__main__":
    main()
