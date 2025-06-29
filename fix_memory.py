#!/usr/bin/env python3
"""
Script to fix CUDA out of memory issues for Fast-DDPM-3D-BraTS
"""

import os
import torch
import yaml
import shutil
from pathlib import Path


def update_config_volume_size():
    """Update the config file with appropriate volume size"""
    config_path = Path("configs/fast_ddpm_3d.yml")
    
    # Backup original config
    backup_path = config_path.with_suffix('.yml.backup')
    if not backup_path.exists():
        shutil.copy(config_path, backup_path)
        print(f"✅ Backed up original config to {backup_path}")
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Get GPU memory
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"🖥️  Detected GPU memory: {gpu_memory:.1f} GB")
    else:
        print("❌ CUDA not available!")
        return False
    
    # Determine appropriate volume size
    if gpu_memory <= 8.5:
        volume_size = [64, 64, 64]
    elif gpu_memory <= 12:  # Your GPU falls here
        volume_size = [80, 80, 80]
    elif gpu_memory <= 18:
        volume_size = [96, 96, 96]
    elif gpu_memory <= 26:
        volume_size = [112, 112, 112]
    else:
        volume_size = [128, 128, 128]
    
    # Update config
    config['data']['volume_size'] = volume_size
    
    # Save updated config
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print(f"✅ Updated volume_size to {volume_size}")
    return True


def set_cuda_environment():
    """Set CUDA environment variables for better memory management"""
    env_vars = {
        'PYTORCH_CUDA_ALLOC_CONF': 'max_split_size_mb:512,expandable_segments:True',
        'CUDA_LAUNCH_BLOCKING': '1',  # Helps with debugging
    }
    
    print("\n🔧 Setting CUDA environment variables:")
    for key, value in env_vars.items():
        os.environ[key] = value
        print(f"  {key}={value}")
    
    # Create a shell script for persistence
    with open('cuda_env.sh', 'w') as f:
        f.write("#!/bin/bash\n")
        f.write("# CUDA environment variables for Fast-DDPM-3D-BraTS\n\n")
        for key, value in env_vars.items():
            f.write(f"export {key}='{value}'\n")
        f.write("\necho '✅ CUDA environment variables set'\n")
    
    os.chmod('cuda_env.sh', 0o755)
    print("✅ Created cuda_env.sh - source this before training")


def test_memory_with_new_config():
    """Test if the new configuration works without OOM"""
    print("\n🧪 Testing new configuration...")
    
    try:
        # Load updated config
        with open('configs/fast_ddpm_3d.yml', 'r') as f:
            config = yaml.safe_load(f)
        
        volume_size = tuple(config['data']['volume_size'])
        print(f"Testing with volume size: {volume_size}")
        
        # Simple memory test
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Clear cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        
        # Create test tensors
        batch_size = 1
        try:
            # Simulate model input/output
            x = torch.randn(batch_size, 4, *volume_size, device=device)
            print(f"✅ Created input tensor: {x.shape}")
            
            # Simulate intermediate features (rough estimate)
            features = torch.randn(batch_size, 48, *volume_size, device=device)
            print(f"✅ Created feature tensor: {features.shape}")
            
            # Check memory usage
            if torch.cuda.is_available():
                used_memory = torch.cuda.memory_allocated() / (1024**3)
                total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                print(f"\n💾 Memory usage: {used_memory:.2f} / {total_memory:.2f} GB ({used_memory/total_memory*100:.1f}%)")
                
                if used_memory / total_memory > 0.8:
                    print("⚠️  Memory usage is high! Consider reducing volume size further.")
                else:
                    print("✅ Memory usage looks good!")
            
            # Cleanup
            del x, features
            torch.cuda.empty_cache()
            
            return True
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"❌ Still getting OOM with volume size {volume_size}")
                print("   Try reducing volume_size further in the config")
                return False
            else:
                raise
                
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        return False


def create_optimized_training_script():
    """Create a memory-optimized training command"""
    script_content = """#!/bin/bash
# Memory-optimized training script for Fast-DDPM-3D-BraTS

# Source CUDA environment
source cuda_env.sh

# Clear GPU memory before starting
python -c "import torch; torch.cuda.empty_cache() if torch.cuda.is_available() else None"

# Run training with memory-efficient settings
python scripts/train_3d.py \\
    --data_root $1 \\
    --config configs/fast_ddpm_3d.yml \\
    --gpu 0 \\
    --debug  # Remove this for full training

# Monitor GPU memory during training (in another terminal):
# watch -n 1 nvidia-smi
"""
    
    with open('train_optimized.sh', 'w') as f:
        f.write(script_content)
    
    os.chmod('train_optimized.sh', 0o755)
    print("\n✅ Created train_optimized.sh")
    print("Usage: ./train_optimized.sh /path/to/brats/data")


def main():
    print("🔧 Fast-DDPM-3D-BraTS Memory Fix Script")
    print("=" * 60)
    
    # Step 1: Update config
    if not update_config_volume_size():
        print("❌ Failed to update config")
        return
    
    # Step 2: Set CUDA environment
    set_cuda_environment()
    
    # Step 3: Test new configuration
    success = test_memory_with_new_config()
    
    # Step 4: Create optimized training script
    create_optimized_training_script()
    
    print("\n" + "=" * 60)
    if success:
        print("✅ Memory issues should be resolved!")
        print("\nNext steps:")
        print("1. Source CUDA environment: source cuda_env.sh")
        print("2. Run training: ./train_optimized.sh /path/to/brats/data")
        print("3. Monitor GPU: watch -n 1 nvidia-smi")
    else:
        print("⚠️  Additional manual adjustments may be needed")
        print("\nTry:")
        print("1. Reduce volume_size to [64, 64, 64] in configs/fast_ddpm_3d.yml")
        print("2. Reduce ch (channels) from 48 to 32 in the config")
        print("3. Disable mixed precision training")


if __name__ == "__main__":
    main()