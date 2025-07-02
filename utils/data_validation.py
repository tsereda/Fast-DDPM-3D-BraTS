"""
Data path validation utilities
"""
import os
import logging
from pathlib import Path


def validate_brats_data_structure(data_root):
    """
    Validate BraTS data structure
    
    Args:
        data_root: Path to BraTS data directory
        
    Returns:
        dict: Validation results with structure info
    """
    results = {
        'valid': False,
        'errors': [],
        'warnings': [],
        'case_count': 0,
        'modalities_found': [],
        'structure': {}
    }
    
    if not os.path.exists(data_root):
        results['errors'].append(f"Data root does not exist: {data_root}")
        return results
    
    data_path = Path(data_root)
    
    # Look for case directories
    case_dirs = []
    for item in data_path.iterdir():
        if item.is_dir() and (item.name.startswith('BraTS') or 
                             item.name.startswith('UPENN') or
                             len([f for f in item.glob('*.nii*')]) > 0):
            case_dirs.append(item)
    
    if len(case_dirs) == 0:
        results['errors'].append("No case directories found")
        return results
    
    results['case_count'] = len(case_dirs)
    
    # Check modalities in first few cases
    modality_patterns = ['t1c', 't1n', 't2f', 't2w', 'seg']
    modalities_found = set()
    
    for case_dir in case_dirs[:5]:  # Check first 5 cases
        nii_files = list(case_dir.glob('*.nii*'))
        
        for nii_file in nii_files:
            filename = nii_file.name.lower()
            for modality in modality_patterns:
                if modality in filename:
                    modalities_found.add(modality)
    
    results['modalities_found'] = list(modalities_found)
    
    # Validation checks
    if len(modalities_found) < 4:
        results['warnings'].append(f"Only found {len(modalities_found)} modalities: {modalities_found}")
    
    if 'seg' not in modalities_found:
        results['warnings'].append("No segmentation files found")
    
    # Structure info
    results['structure'] = {
        'total_cases': len(case_dirs),
        'sample_cases': [d.name for d in case_dirs[:3]],
        'modalities': list(modalities_found)
    }
    
    # Mark as valid if we have basic structure
    if len(case_dirs) > 0 and len(modalities_found) >= 3:
        results['valid'] = True
    else:
        results['errors'].append("Insufficient data structure")
    
    return results