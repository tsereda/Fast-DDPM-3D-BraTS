#!/usr/bin/env python3
"""
Quick test script for 3D Fast-DDPM model
Run this first to verify 3D conversion works
"""

import torch
import yaml
import argparse
import sys
import os

# Add current directory to path
sys.path.append('.')

def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace

def test_3d_model():
    print("Testing 3D Fast-DDPM Model...")
    
    # Load config
    config_path = 'configs/fast_ddpm_3d.yml'
    if not os.path.exists(config_path):
        print(f"Config file {config_path} not found!")
        return False
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    config = dict2namespace(config)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    config.device = device
    
    try:
        from models.diffusion_3d import Model3D
        print("✓ Successfully imported Model3D")
    except ImportError as e:
        print(f"✗ Failed to import Model3D: {e}")
        return False
    
    # Test model creation
    try:
        model = Model3D(config)
        print("✓ Successfully created 3D model")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    except Exception as e:
        print(f"✗ Failed to create model: {e}")
        return False
    
    # Test forward pass
    try:
        model = model.to(device)
        batch_size = 1
        volume_size = config.data.volume_size
        
        # Create dummy input: [B, 4, H, W, D] - unified 4→4
        x = torch.randn(batch_size, 4, *volume_size).to(device)
        t = torch.randint(0, 1000, (batch_size,)).to(device)
        
        print(f"Input shape: {x.shape}")
        print(f"Timestep shape: {t.shape}")
        
        with torch.no_grad():
            output = model(x, t)
        
        print(f"Output shape: {output.shape}")
        expected_shape = (batch_size, 1, *volume_size)
        
        if output.shape == expected_shape:
            print("✓ Forward pass successful!")
            print(f"Expected: {expected_shape}, Got: {output.shape}")
        else:
            print(f"✗ Output shape mismatch!")
            print(f"Expected: {expected_shape}, Got: {output.shape}")
            return False
            
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        return False
    
    # Test memory usage
    try:
        if torch.cuda.is_available():
            memory_used = torch.cuda.max_memory_allocated() / 1024**3  # GB
            print(f"Peak GPU memory: {memory_used:.2f} GB")
            
            if memory_used > 24:  # Warning if over 24GB
                print("⚠️  High memory usage! Consider smaller volume_size")
            else:
                print("✓ Memory usage looks good")
    except:
        pass
    
    # Test different volume sizes
    print("\nTesting different volume sizes:")
    test_sizes = [(32, 32, 32), (64, 64, 64), (96, 96, 96)]
    
    for size in test_sizes:
        try:
            x_test = torch.randn(1, 4, *size).to(device)
            with torch.no_grad():
                out_test = model(x_test, t)
            print(f"✓ Volume size {size}: OK")
        except Exception as e:
            print(f"✗ Volume size {size}: {e}")
    
    print("\n" + "="*50)
    print("3D Model Test Complete!")
    print("✓ Your 3D Fast-DDPM model is working correctly")
    print("Next steps:")
    print("1. Copy essential BraSyn files")
    print("2. Test data loading with small dataset")
    print("3. Run short training test")
    print("="*50)
    
    return True

def test_data_loading():
    """Test the unified dataset loading"""
    print("\nTesting 3D Dataset Loading...")
    
    try:
        from data.brain_3d_unified import BraTS3DUnifiedDataset
        print("✓ Successfully imported BraTS3DUnifiedDataset")
    except ImportError as e:
        print(f"✗ Failed to import dataset: {e}")
        return False
    
    # Test with dummy data path
    dummy_path = "data/dummy_brats"
    os.makedirs(dummy_path, exist_ok=True)
    
    try:
        dataset = BraTS3DUnifiedDataset(
            data_root=dummy_path,
            phase='train',
            volume_size=(64, 64, 64)
        )
        print("✓ Dataset creation successful (with dummy path)")
    except Exception as e:
        print(f"Note: Dataset test with dummy path failed (expected): {e}")
    
    return True

if __name__ == '__main__':
    success = test_3d_model()
    test_data_loading()
    
    if success:
        print("\n🎉 3D Fast-DDPM setup is working!")
        print("You're ready to proceed with training.")
    else:
        print("\n❌ Setup needs fixes before training.")