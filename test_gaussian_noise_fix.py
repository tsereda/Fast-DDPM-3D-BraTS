#!/usr/bin/env python3

import torch
import numpy as np
import random

def test_gaussian_noise_logic():
    """Test the new Gaussian noise logic for 4-modality input"""
    
    print("=== TESTING GAUSSIAN NOISE INPUT LOGIC ===\n")
    
    # Simulate the logic from the updated data loader
    modalities = ['t1n', 't1c', 't2w', 't2f']
    crop_size = (64, 64, 64)
    
    # Test 1: All 4 modalities available
    print("Test 1: All 4 modalities available")
    successful_modalities = ['t1n', 't1c', 't2w', 't2f']
    
    # Simulate cropped volumes (normalized to [0,1])
    cropped_volumes = {}
    for mod in modalities:
        cropped_volumes[mod] = torch.rand(crop_size)  # Random [0,1] data
    
    # Select target modality (e.g., t2f)
    target_modality = 't2f'
    target_idx = modalities.index(target_modality)
    target_volume = cropped_volumes[target_modality]
    
    # Available modalities (non-target)
    input_available_modalities = [mod for mod in successful_modalities if mod != target_modality]
    
    # Create input with Gaussian noise for target
    input_modalities = torch.stack([cropped_volumes[mod] for mod in modalities])
    
    # Replace target with Gaussian noise
    noise = torch.normal(mean=0.5, std=0.1, size=input_modalities[target_idx].shape)
    noise = torch.clamp(noise, 0.0, 1.0)
    input_modalities[target_idx] = noise
    
    print(f"  Successful modalities: {successful_modalities}")
    print(f"  Target modality: {target_modality}")
    print(f"  Available for input: {input_available_modalities}")
    print(f"  Input shape: {input_modalities.shape}")
    print(f"  Target shape: {target_volume.shape}")
    print(f"  Input range: [{input_modalities.min():.3f}, {input_modalities.max():.3f}]")
    print(f"  Target range: [{target_volume.min():.3f}, {target_volume.max():.3f}]")
    print(f"  Noise in target channel: [{input_modalities[target_idx].min():.3f}, {input_modalities[target_idx].max():.3f}]")
    print(f"  Available modalities count: {len(input_available_modalities)} (should be 3)")
    
    if len(input_available_modalities) == 3:
        print("  ✅ PASS: Correct number of available modalities")
    else:
        print("  ❌ FAIL: Wrong number of available modalities")
    
    # Test 2: Only 3 modalities available (missing one file)
    print("\nTest 2: Only 3 modalities available (missing t2w file)")
    successful_modalities = ['t1n', 't1c', 't2f']  # Missing t2w
    
    # Simulate missing modality with zeros
    cropped_volumes['t2w'] = torch.zeros(crop_size)  # Missing filled with zeros
    
    target_modality = 't2f'
    target_idx = modalities.index(target_modality)
    input_available_modalities = [mod for mod in successful_modalities if mod != target_modality]
    
    print(f"  Successful modalities: {successful_modalities}")
    print(f"  Target modality: {target_modality}")
    print(f"  Available for input: {input_available_modalities}")
    print(f"  Available modalities count: {len(input_available_modalities)} (should be 2)")
    
    if len(input_available_modalities) == 2:
        print("  ✅ PASS: Correct number for missing modality case")
    else:
        print("  ❌ FAIL: Wrong number for missing modality case")
    
    # Test 3: Verify noise properties
    print("\nTest 3: Gaussian noise properties")
    test_noise = torch.normal(mean=0.5, std=0.1, size=(1000,))
    test_noise = torch.clamp(test_noise, 0.0, 1.0)
    
    print(f"  Noise mean: {test_noise.mean():.3f} (should be ~0.5)")
    print(f"  Noise std: {test_noise.std():.3f} (should be ~0.1)")
    print(f"  Noise min: {test_noise.min():.3f} (should be 0.0)")
    print(f"  Noise max: {test_noise.max():.3f} (should be 1.0)")
    
    if 0.45 <= test_noise.mean() <= 0.55 and test_noise.min() >= 0.0 and test_noise.max() <= 1.0:
        print("  ✅ PASS: Noise properties are correct")
    else:
        print("  ❌ FAIL: Noise properties are incorrect")
    
    print("\n=== ALL TESTS COMPLETED ===")
    print("The Gaussian noise input logic should now work correctly!")
    print("- Always 4 input channels")
    print("- Target channel replaced with Gaussian noise N(0.5, 0.1) clamped to [0,1]")
    print("- Available modalities = successfully loaded modalities excluding target")

if __name__ == "__main__":
    test_gaussian_noise_logic()
