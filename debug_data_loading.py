#!/usr/bin/env python3

import torch
import sys
sys.path.append('/root/Fast-DDPM-3D-BraTS')

from data.brain_3d_unified import BraTS3DUnifiedDataset
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def test_data_loading():
    """Test data loading to understand the duplicate modalities issue"""
    
    # Test with a minimal example
    try:
        # Initialize dataset
        dataset = BraTS3DUnifiedDataset(
            data_root='/root/Fast-DDPM-3D-BraTS/data/train',
            phase='train',
            crop_size=(32, 32, 32),
            min_input_modalities=1,
            crops_per_volume=1
        )
        
        print(f"Dataset found {len(dataset.cases)} cases")
        print(f"Dataset has {len(dataset)} total samples")
        
        if len(dataset) > 0:
            # Test first sample
            sample = dataset[0]
            
            print("\n=== SAMPLE 0 ===")
            print(f"Case name: {sample['case_name']}")
            print(f"Target modality: {sample['target_modality']}")
            print(f"Target index: {sample['target_idx']}")
            print(f"Available modalities: {sample['available_modalities']}")
            print(f"Input shape: {sample['input'].shape}")
            print(f"Target shape: {sample['target'].shape}")
            print(f"Input range: [{sample['input'].min():.3f}, {sample['input'].max():.3f}]")
            print(f"Target range: [{sample['target'].min():.3f}, {sample['target'].max():.3f}]")
            
            # Check for duplicates in available_modalities
            available = sample['available_modalities']
            unique_available = list(set(available))
            if len(available) != len(unique_available):
                print(f"\n!!! FOUND DUPLICATES IN AVAILABLE MODALITIES !!!")
                print(f"Available modalities: {available}")
                print(f"Unique modalities: {unique_available}")
            else:
                print(f"\nNo duplicates found in available modalities")
            
            # Test a few more samples
            for i in range(1, min(5, len(dataset))):
                sample = dataset[i]
                available = sample['available_modalities']
                unique_available = list(set(available))
                print(f"Sample {i}: available={available}, unique={unique_available}")
                if len(available) != len(unique_available):
                    print(f"  !!! DUPLICATES FOUND in sample {i}")
        
        # Also test dataset cases directly
        print(f"\n=== CHECKING CASES DIRECTLY ===")
        for i, case_dir in enumerate(dataset.cases[:3]):
            print(f"\nCase {i}: {case_dir.name}")
            
            available_count = 0
            found_modalities = []
            for modality in dataset.modalities:
                modality_file = dataset._find_modality_file(case_dir, modality)
                if modality_file:
                    available_count += 1
                    found_modalities.append(modality)
                    print(f"  {modality}: {modality_file.name}")
                else:
                    print(f"  {modality}: NOT FOUND")
            
            print(f"  Found modalities: {found_modalities}")
            print(f"  Available count: {available_count}")
            
            # Check for duplicates
            unique_found = list(set(found_modalities))
            if len(found_modalities) != len(unique_found):
                print(f"  !!! DUPLICATES: {found_modalities} vs {unique_found}")
    
    except Exception as e:
        print(f"Error testing data loading: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_data_loading()
