#!/usr/bin/env python3
"""
Environment verification script for Fast-DDPM-3D-BraTS
"""
import sys
import subprocess
import importlib

def check_package(package_name, import_name=None):
    """Check if a package is available"""
    if import_name is None:
        import_name = package_name
    
    try:
        importlib.import_module(import_name)
        print(f"✅ {package_name}: Available")
        return True
    except ImportError:
        print(f"❌ {package_name}: Missing")
        return False

def check_cuda():
    """Check CUDA availability"""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA: Available (PyTorch {torch.__version__})")
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            return True
        else:
            print(f"⚠️ CUDA: Not available (PyTorch {torch.__version__})")
            return False
    except ImportError:
        print("❌ PyTorch: Not installed")
        return False

def main():
    print("🔍 Fast-DDPM-3D-BraTS Environment Verification")
    print("=" * 50)
    
    # Core packages
    core_packages = [
        ("PyTorch", "torch"),
        ("NumPy", "numpy"),
        ("NiBabel", "nibabel"),
        ("SimpleITK", "simpleitk"),
        ("TQDM", "tqdm"),
        ("PyYAML", "yaml"),
        ("Matplotlib", "matplotlib"),
        ("SciPy", "scipy"),
        ("Pandas", "pandas"),
        ("OpenCV", "cv2"),
        ("Scikit-Image", "skimage"),
    ]
    
    # Optional packages
    optional_packages = [
        ("nnUNetv2", "nnunetv2"),
        ("Batch Generators", "batchgenerators"),
        ("GDown", "gdown"),
        ("Wandb", "wandb"),
        ("LPIPS", "lpips"),
        ("Einops", "einops"),
        ("Accelerate", "accelerate"),
        ("Diffusers", "diffusers"),
    ]
    
    print("Core Packages:")
    core_available = sum(check_package(name, import_name) for name, import_name in core_packages)
    
    print(f"\nOptional Packages:")
    optional_available = sum(check_package(name, import_name) for name, import_name in optional_packages)
    
    print(f"\nCUDA Check:")
    cuda_available = check_cuda()
    
    print(f"\nSummary:")
    print(f"Core packages: {core_available}/{len(core_packages)}")
    print(f"Optional packages: {optional_available}/{len(optional_packages)}")
    print(f"CUDA available: {'Yes' if cuda_available else 'No'}")
    
    if core_available == len(core_packages) and cuda_available:
        print(f"\n🎉 Environment is ready for Fast-DDPM-3D-BraTS!")
        return 0
    else:
        print(f"\n⚠️ Environment setup incomplete. Please run: bash setup_environment.sh")
        return 1

if __name__ == "__main__":
    sys.exit(main())
