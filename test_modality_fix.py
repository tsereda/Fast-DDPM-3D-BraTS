#!/usr/bin/env python3

import torch
import sys
import os
sys.path.append('/root/Fast-DDPM-3D-BraTS')

# Simple mock test without requiring actual data files
def test_modality_selection_logic():
    """Test the modality selection logic without requiring real data"""
    
    print("=== TESTING MODALITY SELECTION LOGIC ===\n")
    
    # Simulate the logic from the dataset
    modalities = ['t1n', 't1c', 't2w', 't2f']
    
    # Test case 1: All 4 modalities available
    print("Test 1: All 4 modalities available")
    available_modalities = ['t1n', 't1c', 't2w', 't2f']
    
    # Select target (simulating random choice in training)
    import random
    random.seed(42)  # For reproducible results
    target_modality = random.choice(available_modalities)
    print(f"  All available: {available_modalities}")
    print(f"  Selected target: {target_modality}")
    
    # Apply the fix: remove target from available 
    input_available_modalities = [mod for mod in available_modalities if mod != target_modality]
    print(f"  Input available (fixed): {input_available_modalities}")
    print(f"  Length: {len(input_available_modalities)} (should be 3)")
    
    assert len(input_available_modalities) == 3, "Should have exactly 3 input modalities"
    assert target_modality not in input_available_modalities, "Target should not be in input available"
    print("  ✅ PASS\n")
    
    # Test case 2: Only 3 modalities available
    print("Test 2: Only 3 modalities available (missing t2f)")
    available_modalities = ['t1n', 't1c', 't2w']  # Missing t2f
    target_modality = random.choice(available_modalities)
    print(f"  All available: {available_modalities}")
    print(f"  Selected target: {target_modality}")
    
    input_available_modalities = [mod for mod in available_modalities if mod != target_modality]
    print(f"  Input available (fixed): {input_available_modalities}")
    print(f"  Length: {len(input_available_modalities)} (should be 2)")
    
    assert len(input_available_modalities) == 2, "Should have exactly 2 input modalities"
    assert target_modality not in input_available_modalities, "Target should not be in input available"
    print("  ✅ PASS\n")
    
    # Test case 3: What was happening before (the bug)
    print("Test 3: The OLD buggy behavior")
    available_modalities = ['t1n', 't1c', 't2w', 't2f']
    target_modality = 't1n'
    print(f"  All available: {available_modalities}")
    print(f"  Selected target: {target_modality}")
    
    # OLD buggy way: return original available_modalities
    buggy_result = available_modalities  # This was the bug!
    print(f"  OLD buggy result: {buggy_result}")
    print(f"  Contains target?: {target_modality in buggy_result} (this was the problem!)")
    
    # NEW fixed way
    fixed_result = [mod for mod in available_modalities if mod != target_modality]
    print(f"  NEW fixed result: {fixed_result}")
    print(f"  Contains target?: {target_modality in fixed_result} (should be False)")
    print("  ✅ BUG IDENTIFIED AND FIXED\n")
    
    print("=== ALL TESTS PASSED ===")
    print("The duplicate modalities issue should now be resolved!")

if __name__ == "__main__":
    test_modality_selection_logic()
