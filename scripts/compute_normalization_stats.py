#!/usr/bin/env python3
"""
Compute proper normalization statistics for BraTS dataset
This should be run before training to get accurate global min/max values
"""

import os
import sys
import argparse
import numpy as np
import nibabel as nib
from pathlib import Path
from tqdm import tqdm
import json

# Add path
sys.path.append('.')
sys.path.append('..')

from data.brain_3d_unified import BraTS3DUnifiedDataset


def find_modality_file(case_dir, modality):
    """Find modality file using common BraTS patterns"""
    patterns = [
        f'*{modality}*.nii*',
        f'*{modality.upper()}*.nii*',
        f'*-{modality}.nii*',
        f'*_{modality}.nii*'
    ]
    
    for pattern in patterns:
        files = list(case_dir.glob(pattern))
        if files:
            files.sort(key=lambda x: (x.suffix != '.gz', x.name))
            return files[0]
    return None


def compute_global_stats(data_root, num_samples=None):
    """
    Compute global normalization statistics for all modalities
    
    Args:
        data_root: Path to BraTS data directory
        num_samples: If specified, only use this many cases for computation
    """
    data_root = Path(data_root)
    modalities = ['t1n', 't1c', 't2w', 't2f']
    
    # Initialize statistics
    stats = {}
    for modality in modalities:
        stats[modality] = {
            'values': [],
            'min': float('inf'),
            'max': float('-inf'),
            'mean': 0.0,
            'std': 0.0,
            'percentiles': {}
        }
    
    # Find all case directories
    cases = []
    for item in data_root.iterdir():
        if item.is_dir():
            cases.append(item)
    
    cases = sorted(cases, key=lambda x: x.name)
    
    if num_samples:
        cases = cases[:num_samples]
    
    print(f"Computing statistics from {len(cases)} cases...")
    
    # Process each case
    for case_dir in tqdm(cases, desc="Processing cases"):
        for modality in modalities:
            modality_file = find_modality_file(case_dir, modality)
            
            if modality_file:
                try:
                    # Load volume
                    nii = nib.load(str(modality_file))
                    volume = nii.get_fdata().astype(np.float32)
                    
                    # Skip background (zero) voxels for more accurate statistics
                    nonzero_voxels = volume[volume > 0]
                    
                    if len(nonzero_voxels) > 0:
                        # Update min/max
                        vol_min = float(nonzero_voxels.min())
                        vol_max = float(nonzero_voxels.max())
                        
                        stats[modality]['min'] = min(stats[modality]['min'], vol_min)
                        stats[modality]['max'] = max(stats[modality]['max'], vol_max)
                        
                        # Store values for percentile computation
                        if len(stats[modality]['values']) < 1000000:  # Limit memory usage
                            stats[modality]['values'].extend(
                                nonzero_voxels[::max(1, len(nonzero_voxels)//10000)].tolist()
                            )
                
                except Exception as e:
                    print(f"Error processing {modality_file}: {e}")
                    continue
    
    # Compute final statistics
    final_stats = {}
    for modality in modalities:
        if stats[modality]['values']:
            values = np.array(stats[modality]['values'])
            
            final_stats[modality] = {
                'min': 0.0,  # Always start from 0 for medical images
                'max': float(stats[modality]['max']),
                'data_min': float(stats[modality]['min']),
                'data_max': float(stats[modality]['max']),
                'mean': float(values.mean()),
                'std': float(values.std()),
                'percentiles': {
                    '1': float(np.percentile(values, 1)),
                    '5': float(np.percentile(values, 5)),
                    '95': float(np.percentile(values, 95)),
                    '99': float(np.percentile(values, 99)),
                    '99.5': float(np.percentile(values, 99.5)),
                    '99.9': float(np.percentile(values, 99.9))
                }
            }
            
            print(f"\n{modality.upper()} statistics:")
            print(f"  Data range: [{final_stats[modality]['data_min']:.1f}, {final_stats[modality]['data_max']:.1f}]")
            print(f"  Mean: {final_stats[modality]['mean']:.1f}")
            print(f"  Std: {final_stats[modality]['std']:.1f}")
            print(f"  99.9%ile: {final_stats[modality]['percentiles']['99.9']:.1f}")
        else:
            print(f"No valid data found for {modality}")
            final_stats[modality] = {
                'min': 0.0,
                'max': 1000.0,  # Default fallback
                'data_min': 0.0,
                'data_max': 1000.0,
                'mean': 500.0,
                'std': 200.0,
                'percentiles': {'99.9': 1000.0}
            }
    
    return final_stats


def main():
    parser = argparse.ArgumentParser(description='Compute BraTS normalization statistics')
    parser.add_argument('--data_root', type=str, required=True, help='Path to BraTS data directory')
    parser.add_argument('--output', type=str, default='normalization_stats.json', help='Output JSON file')
    parser.add_argument('--num_samples', type=int, default=None, help='Number of samples to use (default: all)')
    
    args = parser.parse_args()
    
    # Compute statistics
    stats = compute_global_stats(args.data_root, args.num_samples)
    
    # Save to JSON
    with open(args.output, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\nStatistics saved to {args.output}")
    
    # Print Python code for easy copy-paste
    print("\n" + "="*50)
    print("Copy this to your dataset's global_stats:")
    print("="*50)
    print("self.global_stats = {")
    for modality in ['t1n', 't1c', 't2w', 't2f']:
        if modality in stats:
            # Use 99.9 percentile as max for better normalization
            max_val = stats[modality]['percentiles']['99.9']
            print(f"    '{modality}': {{'min': 0.0, 'max': {max_val:.1f}}},")
    print("}")


if __name__ == '__main__':
    main()
