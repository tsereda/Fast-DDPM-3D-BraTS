#!/usr/bin/env python3
"""
Debug script to check BraTS dataset modalities
"""

import os
import sys
import glob
from collections import defaultdict

# Add path for imports
sys.path.append('.')
sys.path.append('..')

def analyze_brats_structure(data_root):
    """Analyze the structure of BraTS dataset"""
    print(f"🔍 Analyzing BraTS dataset structure at: {data_root}")
    print("=" * 80)
    
    if not os.path.exists(data_root):
        print(f"❌ Data root does not exist: {data_root}")
        return
    
    # Get all case directories
    case_dirs = [d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))]
    case_dirs = sorted([d for d in case_dirs if d.startswith('BraTS')])
    
    print(f"📁 Found {len(case_dirs)} case directories")
    
    if len(case_dirs) == 0:
        print("❌ No BraTS case directories found!")
        return
    
    # Analyze first few cases
    modality_stats = defaultdict(int)
    file_patterns = defaultdict(list)
    
    for i, case_dir in enumerate(case_dirs[:10]):  # Check first 10 cases
        case_path = os.path.join(data_root, case_dir)
        files = os.listdir(case_path)
        
        print(f"\n📂 Case {i+1}: {case_dir}")
        print(f"   Files: {files}")
        
        # Check for different modality patterns
        nii_files = [f for f in files if f.endswith('.nii.gz')]
        
        for file in nii_files:
            print(f"   📄 {file}")
            
            # Identify modality from filename
            if 't1n' in file.lower() or 't1.' in file.lower():
                modality_stats['t1n'] += 1
                file_patterns['t1n'].append(file)
            elif 't1c' in file.lower() or 't1ce' in file.lower():
                modality_stats['t1ce'] += 1
                file_patterns['t1ce'].append(file)
            elif 't2w' in file.lower() or 't2.' in file.lower():
                modality_stats['t2w'] += 1
                file_patterns['t2w'].append(file)
            elif 'flair' in file.lower():
                modality_stats['flair'] += 1
                file_patterns['flair'].append(file)
            elif 'seg' in file.lower():
                modality_stats['seg'] += 1
                file_patterns['seg'].append(file)
    
    print(f"\n📊 Modality Statistics (from first 10 cases):")
    for modality, count in modality_stats.items():
        print(f"   {modality}: {count} files")
    
    print(f"\n🔤 File Patterns:")
    for modality, patterns in file_patterns.items():
        unique_patterns = list(set(patterns))
        print(f"   {modality}: {unique_patterns[:3]}...")  # Show first 3 unique patterns
    
    return modality_stats, file_patterns


def debug_dataset_class(data_root):
    """Debug the actual dataset class"""
    try:
        from data.brain_3d_unified import BraTS3DUnifiedDataset
        
        print(f"\n🧪 Testing BraTS3DUnifiedDataset...")
        dataset = BraTS3DUnifiedDataset(data_root=data_root, phase='train')
        
        print(f"📊 Dataset info:")
        print(f"   Total samples: {len(dataset)}")
        
        if len(dataset) > 0:
            # Get first sample
            sample = dataset[0]
            print(f"   Sample keys: {list(sample.keys())}")
            print(f"   Input shape: {sample['input'].shape}")
            print(f"   Target shape: {sample['target'].shape}")
            print(f"   Target modality: {sample['target_modality']}")
            print(f"   Available modalities: {sample['available_modalities']}")
            print(f"   Target idx: {sample['target_idx']}")
            
            # Check a few more samples
            print(f"\n🔍 Checking first 5 samples for available modalities:")
            for i in range(min(5, len(dataset))):
                sample = dataset[i]
                print(f"   Sample {i}: available={sample['available_modalities']}, target={sample['target_modality']}")
        
    except Exception as e:
        print(f"❌ Error testing dataset class: {e}")
        import traceback
        traceback.print_exc()


def check_specific_case(data_root, case_name=None):
    """Check a specific case in detail"""
    case_dirs = [d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d)) and d.startswith('BraTS')]
    
    if not case_dirs:
        print("❌ No BraTS cases found")
        return
    
    if case_name is None:
        case_name = case_dirs[0]  # Use first case
    
    case_path = os.path.join(data_root, case_name)
    
    print(f"\n🔍 Detailed analysis of case: {case_name}")
    print(f"📁 Path: {case_path}")
    
    files = os.listdir(case_path)
    nii_files = [f for f in files if f.endswith('.nii.gz')]
    
    print(f"📄 All .nii.gz files:")
    for file in nii_files:
        file_path = os.path.join(case_path, file)
        size = os.path.getsize(file_path) / (1024 * 1024)  # MB
        print(f"   {file} ({size:.1f} MB)")
    
    # Try to load with nibabel if available
    try:
        import nibabel as nib
        print(f"\n🧠 Nibabel analysis:")
        for file in nii_files:
            file_path = os.path.join(case_path, file)
            img = nib.load(file_path)
            print(f"   {file}: shape={img.shape}, dtype={img.get_fdata().dtype}")
    except ImportError:
        print("   (nibabel not available for detailed analysis)")
    except Exception as e:
        print(f"   Error loading with nibabel: {e}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Debug BraTS dataset modalities')
    parser.add_argument('--data_root', type=str, required=True, help='Path to BraTS data')
    parser.add_argument('--case', type=str, help='Specific case to analyze in detail')
    
    args = parser.parse_args()
    
    # Run analysis
    analyze_brats_structure(args.data_root)
    debug_dataset_class(args.data_root)
    check_specific_case(args.data_root, args.case)