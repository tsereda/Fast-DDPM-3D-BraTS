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


def print_validation_results(results):
    """Print validation results in a readable format"""
    print("\n" + "="*50)
    print("BraTS Data Validation Results")
    print("="*50)
    
    if results['valid']:
        print("✅ Data structure is valid")
    else:
        print("❌ Data structure has issues")
    
    print(f"\nStructure Info:")
    print(f"  Total cases: {results['case_count']}")
    print(f"  Modalities found: {results['modalities_found']}")
    
    if results['structure'].get('sample_cases'):
        print(f"  Sample cases: {results['structure']['sample_cases']}")
    
    if results['warnings']:
        print(f"\n⚠️  Warnings:")
        for warning in results['warnings']:
            print(f"    - {warning}")
    
    if results['errors']:
        print(f"\n❌ Errors:")
        for error in results['errors']:
            print(f"    - {error}")
    
    print("="*50)


def setup_data_logging(data_root):
    """Setup logging for data loading issues"""
    log_dir = os.path.join(os.path.dirname(data_root), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # Create data-specific logger
    logger = logging.getLogger('data_loader')
    logger.setLevel(logging.INFO)
    
    # File handler
    fh = logging.FileHandler(os.path.join(log_dir, 'data_loading.log'))
    fh.setLevel(logging.INFO)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)
    
    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger


def check_disk_space(data_root, min_gb=10):
    """Check if there's enough disk space for training"""
    try:
        import shutil
        total, used, free = shutil.disk_usage(data_root)
        free_gb = free / (1024**3)
        
        if free_gb < min_gb:
            print(f"⚠️  Low disk space: {free_gb:.1f} GB available (need at least {min_gb} GB)")
            return False
        else:
            print(f"✅ Disk space OK: {free_gb:.1f} GB available")
            return True
    except Exception as e:
        print(f"Could not check disk space: {e}")
        return True  # Assume OK if we can't check


def create_dummy_brats_data(output_dir, num_cases=5):
    """Create dummy BraTS data for testing"""
    import torch
    import nibabel as nib
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Creating dummy BraTS data in {output_dir}")
    
    modalities = ['t1c', 't1n', 't2f', 't2w']
    
    for i in range(num_cases):
        case_name = f"BraTS_dummy_{i:03d}"
        case_dir = output_path / case_name
        case_dir.mkdir(exist_ok=True)
        
        # Create dummy volumes (smaller for testing)
        volume_shape = (128, 128, 128)
        
        for modality in modalities:
            # Create random volume data
            data = torch.randn(*volume_shape).numpy()
            
            # Create NIfTI image
            nii_img = nib.Nifti1Image(data, affine=None)
            
            # Save file
            filename = f"{case_name}_{modality}.nii.gz"
            filepath = case_dir / filename
            nib.save(nii_img, str(filepath))
        
        # Create dummy segmentation
        seg_data = torch.randint(0, 4, volume_shape).numpy()
        seg_img = nib.Nifti1Image(seg_data, affine=None)
        seg_path = case_dir / f"{case_name}_seg.nii.gz"
        nib.save(seg_img, str(seg_path))
    
    print(f"✅ Created {num_cases} dummy cases in {output_dir}")
    return output_dir
