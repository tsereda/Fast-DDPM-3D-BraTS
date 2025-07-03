#!/usr/bin/env python3
"""
Debug script to test data loading with mock data structure
This bypasses permission issues and focuses on the data loading logic
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path

# Add current directory to path
sys.path.append('.')

def create_mock_data_structure(mock_root):
    """Create a mock BraTS data structure for testing"""
    mock_root = Path(mock_root)
    
    # Create train directory
    train_dir = mock_root / "train"
    train_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a few mock cases
    case_names = ["BraTS2021_00001", "BraTS2021_00002", "BraTS2021_00003"]
    
    for case_name in case_names:
        case_dir = train_dir / case_name
        case_dir.mkdir(exist_ok=True)
        
        # Create mock NIfTI files (just empty files for structure testing)
        modalities = ["t1n", "t1c", "t2w", "t2f"]
        for mod in modalities:
            mock_file = case_dir / f"{case_name}_{mod}.nii.gz"
            mock_file.touch()  # Create empty file
    
    print(f"✅ Created mock data structure at {mock_root}")
    return mock_root

def test_data_loading_logic():
    """Test the data loading logic with mock data"""
    print("=== Testing Data Loading Logic ===")
    
    # Create temporary mock data structure
    import tempfile
    with tempfile.TemporaryDirectory() as temp_dir:
        mock_root = create_mock_data_structure(temp_dir)
        
        try:
            # Import and test the dataset class
            from data.brain_3d_unified import BraTS3DUnifiedDataset
            
            # Create dataset instance with mock data
            dataset = BraTS3DUnifiedDataset(
                data_root=str(mock_root),
                phase='train'
            )
            
            print(f"✅ Dataset created successfully")
            print(f"Number of cases found: {len(dataset.cases)}")
            print(f"Cases: {dataset.cases}")
            
            # Test the modality selection logic
            if len(dataset.cases) > 0:
                case_name = dataset.cases[0]
                print(f"\n=== Testing Modality Selection for {case_name} ===")
                
                # Test _get_available_modalities method
                available_modalities = dataset._get_available_modalities(case_name)
                print(f"Available modalities: {available_modalities}")
                
                # Test _select_target_and_available method multiple times
                print("\n=== Testing Multiple Modality Selections ===")
                for i in range(10):
                    target_modality, available_modalities_list, target_idx = dataset._select_target_and_available(case_name)
                    print(f"Trial {i+1}: Target={target_modality}, Available={available_modalities_list}, Target_idx={target_idx}")
                    
                    # Check for duplicates
                    unique_available = list(set(available_modalities_list))
                    if len(unique_available) != len(available_modalities_list):
                        print(f"🚨 FOUND DUPLICATE BUG! Available modalities contain duplicates:")
                        print(f"   Original: {available_modalities_list}")
                        print(f"   Unique: {unique_available}")
                        return False
                
                print("✅ No duplicate modalities found in 10 trials")
                return True
            else:
                print("❌ No cases found in mock data")
                return False
                
        except Exception as e:
            print(f"❌ Error testing data loading: {e}")
            import traceback
            traceback.print_exc()
            return False

def inspect_dataset_code():
    """Inspect the dataset code to identify potential issues"""
    print("\n=== Inspecting Dataset Code ===")
    
    try:
        # Read the dataset file
        with open('data/brain_3d_unified.py', 'r') as f:
            content = f.read()
        
        # Look for the _select_target_and_available method
        lines = content.split('\n')
        in_method = False
        method_lines = []
        
        for i, line in enumerate(lines):
            if 'def _select_target_and_available' in line:
                in_method = True
                method_lines.append(f"{i+1}: {line}")
            elif in_method:
                if line.strip().startswith('def ') and not line.strip().startswith('def _select_target_and_available'):
                    break
                method_lines.append(f"{i+1}: {line}")
        
        print("Found _select_target_and_available method:")
        for line in method_lines:
            print(line)
            
    except Exception as e:
        print(f"❌ Error reading dataset code: {e}")

def main():
    """Main debug function"""
    print("🔍 Debugging Data Loading (Mock Mode)")
    print("=" * 50)
    
    # Test data loading logic
    success = test_data_loading_logic()
    
    # Inspect dataset code
    inspect_dataset_code()
    
    if success:
        print("\n✅ Data loading test passed - no obvious duplicate bug detected")
        print("🔍 The duplicate modality issue might be in the actual data loading process")
        print("   or in how the data is processed after loading.")
    else:
        print("\n❌ Data loading test failed - check the output above for issues")

if __name__ == "__main__":
    main()
