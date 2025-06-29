"""
Utilities for finding and loading 3D volume files
"""
import os
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def get_available_3d_vol_names(case_path):
    """
    Get available volume filenames for a BraTS case
    
    Args:
        case_path: Path to case directory
        
    Returns:
        vol_files: dict mapping modality to filename
        available_modalities: list of available modality names
    """
    case_path = Path(case_path)
    
    # Initialize with all possible modalities
    vol_files = {
        't1n': None,
        't1c': None, 
        't2w': None,
        't2f': None,
        'seg': None  # Segmentation if available
    }
    
    # Find files for each modality
    for modality in vol_files.keys():
        # Try different naming patterns
        patterns = [
            f'*-{modality}.nii*',    # BraTS standard: case-t1n.nii.gz
            f'*_{modality}.nii*',     # Alternative: case_t1n.nii.gz
            f'*{modality}*.nii*',     # Loose match
        ]
        
        for pattern in patterns:
            files = list(case_path.glob(pattern))
            if files:
                vol_files[modality] = files[0].name
                break
        
        # If still not found, try case-insensitive search
        if vol_files[modality] is None and modality != 'seg':
            all_files = list(case_path.glob('*.nii*'))
            for f in all_files:
                if modality.lower() in f.name.lower() and 'seg' not in f.name.lower():
                    vol_files[modality] = f.name
                    break
    
    # Get list of available modalities (excluding seg)
    available_modalities = [mod for mod in ['t1n', 't1c', 't2w', 't2f'] 
                           if vol_files[mod] is not None]
    
    return vol_files, available_modalities


def find_brats_cases(data_root, min_modalities=3):
    """
    Find all valid BraTS cases in a directory
    
    Args:
        data_root: Root directory to search
        min_modalities: Minimum number of modalities required
        
    Returns:
        List of case directories
    """
    data_root = Path(data_root)
    valid_cases = []
    
    # Search for directories containing NIfTI files
    for item in data_root.rglob('*'):
        if item.is_dir():
            vol_files, available_modalities = get_available_3d_vol_names(item)
            
            if len(available_modalities) >= min_modalities:
                valid_cases.append(item)
                logger.debug(f"Found valid case: {item.name} with {len(available_modalities)} modalities")
    
    logger.info(f"Found {len(valid_cases)} valid cases in {data_root}")
    return sorted(valid_cases, key=lambda x: x.name)


def load_nifti_info(filepath):
    """
    Load NIfTI header information without loading the full volume
    
    Args:
        filepath: Path to NIfTI file
        
    Returns:
        dict with shape, affine, and other metadata
    """
    import nibabel as nib
    
    try:
        nii = nib.load(str(filepath))
        info = {
            'shape': nii.shape,
            'affine': nii.affine,
            'header': dict(nii.header),
            'dtype': nii.get_data_dtype()
        }
        return info
    except Exception as e:
        logger.error(f"Failed to load NIfTI info from {filepath}: {e}")
        return None


def get_modality_statistics(case_path):
    """
    Get basic statistics for all modalities in a case
    
    Args:
        case_path: Path to case directory
        
    Returns:
        dict with statistics for each modality
    """
    import nibabel as nib
    import numpy as np
    
    vol_files, available_modalities = get_available_3d_vol_names(case_path)
    stats = {}
    
    for modality in available_modalities:
        if vol_files[modality]:
            filepath = Path(case_path) / vol_files[modality]
            try:
                nii = nib.load(str(filepath))
                data = nii.get_fdata()
                
                # Calculate statistics
                stats[modality] = {
                    'shape': data.shape,
                    'min': float(np.min(data)),
                    'max': float(np.max(data)),
                    'mean': float(np.mean(data)),
                    'std': float(np.std(data)),
                    'percentiles': {
                        '1': float(np.percentile(data, 1)),
                        '99': float(np.percentile(data, 99))
                    }
                }
            except Exception as e:
                logger.error(f"Failed to get statistics for {modality} in {case_path}: {e}")
                stats[modality] = None
    
    return stats


# Test the functions
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        test_path = sys.argv[1]
        print(f"Testing with path: {test_path}")
        
        # Test finding cases
        cases = find_brats_cases(test_path)
        print(f"Found {len(cases)} cases")
        
        if cases:
            # Test with first case
            first_case = cases[0]
            print(f"\nTesting with case: {first_case.name}")
            
            vol_files, modalities = get_available_3d_vol_names(first_case)
            print(f"Available modalities: {modalities}")
            print(f"Files: {vol_files}")
            
            # Get statistics
            stats = get_modality_statistics(first_case)
            for mod, stat in stats.items():
                if stat:
                    print(f"\n{mod} statistics:")
                    print(f"  Shape: {stat['shape']}")
                    print(f"  Range: [{stat['min']:.2f}, {stat['max']:.2f}]")
                    print(f"  Mean: {stat['mean']:.2f}, Std: {stat['std']:.2f}")
    else:
        print("Usage: python image_folder.py <path_to_brats_data>")