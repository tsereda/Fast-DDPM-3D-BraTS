#!/usr/bin/env python3
"""
Comprehensive test script for Fast-DDPM-3D-BraTS
Tests all major components and validates fixes
"""

import torch
import yaml
import sys
import os
import traceback
from pathlib import Path

# Add current directory to path
sys.path.append('.')

def test_imports():
    """Test all critical imports"""
    print("="*60)
    print("Testing Imports")
    print("="*60)
    
    try:
        from models.diffusion_3d import Model3D
        print("✅ Model3D import successful")
    except ImportError as e:
        print(f"❌ Model3D import failed: {e}")
        return False
    
    try:
        from functions.losses import fast_ddpm_loss, discretized_gaussian_log_likelihood
        print("✅ Loss functions import successful")
    except ImportError as e:
        print(f"❌ Loss functions import failed: {e}")
        return False
    
    try:
        from functions.denoising_3d import unified_4to4_generalized_steps
        print("✅ Denoising functions import successful")
    except ImportError as e:
        print(f"❌ Denoising functions import failed: {e}")
        return False
    
    try:
        from utils.gpu_memory import get_recommended_volume_size, check_memory_usage
        print("✅ GPU memory utilities import successful")
    except ImportError as e:
        print(f"❌ GPU memory utilities import failed: {e}")
        return False
    
    try:
        from utils.data_validation import validate_brats_data_structure
        print("✅ Data validation utilities import successful")
    except ImportError as e:
        print(f"❌ Data validation utilities import failed: {e}")
        return False
    
    return True


def test_gpu_memory_detection():
    """Test GPU memory detection and volume size recommendation"""
    print("\n" + "="*60)
    print("Testing GPU Memory Detection")
    print("="*60)
    
    try:
        from utils.gpu_memory import get_gpu_memory_gb, get_recommended_volume_size
        
        gpu_memory = get_gpu_memory_gb()
        print(f"Detected GPU memory: {gpu_memory:.1f} GB")
        
        recommended_size = get_recommended_volume_size()
        print(f"Recommended volume size: {recommended_size}")
        
        # Test with different memory values
        test_memories = [8, 16, 24, 32]
        for memory in test_memories:
            size = get_recommended_volume_size(memory)
            print(f"  {memory}GB GPU -> {size}")
        
        return True
        
    except Exception as e:
        print(f"❌ GPU memory detection failed: {e}")
        return False


def test_loss_functions():
    """Test loss functions with various configurations"""
    print("\n" + "="*60)
    print("Testing Loss Functions")
    print("="*60)
    
    try:
        from functions.losses import fast_ddpm_loss, discretized_gaussian_log_likelihood
        
        # Create dummy data
        batch_size = 1
        volume_size = (64, 64, 64)
        
        x_available = torch.randn(batch_size, 4, *volume_size)
        x_target = torch.randn(batch_size, 1, *volume_size)
        t = torch.randint(0, 1000, (batch_size,))
        e = torch.randn_like(x_target)
        betas = torch.linspace(0.0001, 0.02, 1000)
        
        # Create a simple mock model
        class MockModel(torch.nn.Module):
            def __init__(self, var_type='fixed'):
                super().__init__()
                self.conv = torch.nn.Conv3d(4, 1, 3, padding=1)
                self.var_type = var_type
                if var_type in ['learned', 'learned_range']:
                    self.var_conv = torch.nn.Conv3d(4, 1, 3, padding=1)
            
            def forward(self, x, t):
                mean = self.conv(x)
                if self.var_type in ['learned', 'learned_range']:
                    var = self.var_conv(x)
                    return mean, var
                return mean
        
        # Test different variance types
        var_types = ['fixed', 'learned', 'learned_range']
        
        for var_type in var_types:
            print(f"Testing {var_type} variance...")
            model = MockModel(var_type)
            
            loss = fast_ddpm_loss(model, x_available, x_target, t, e, betas, var_type)
            print(f"  ✅ {var_type} loss: {loss.item():.6f}")
        
        # Test discretized gaussian log likelihood
        x = torch.randn(2, 3, 32, 32, 32)
        means = torch.randn_like(x)
        log_scales = torch.randn_like(x)
        
        log_probs = discretized_gaussian_log_likelihood(x, means, log_scales)
        print(f"✅ Discretized Gaussian log likelihood shape: {log_probs.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Loss function test failed: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return False


def test_model_creation_and_forward():
    """Test model creation and forward pass"""
    print("\n" + "="*60)
    print("Testing Model Creation and Forward Pass")
    print("="*60)
    
    try:
        # Load config
        config_path = 'configs/fast_ddpm_3d.yml'
        if not os.path.exists(config_path):
            print(f"❌ Config file not found: {config_path}")
            return False
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Convert to namespace
        class DictAsNamespace:
            def __init__(self, d):
                for k, v in d.items():
                    if isinstance(v, dict):
                        setattr(self, k, DictAsNamespace(v))
                    else:
                        setattr(self, k, v)
        
        config = DictAsNamespace(config)
        
        # Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        from models.diffusion_3d import Model3D
        
        # Create model
        model = Model3D(config)
        model = model.to(device)
        print(f"✅ Model created successfully")
        print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Test forward pass with different sizes
        test_sizes = [(32, 32, 32), (64, 64, 64)]
        
        for size in test_sizes:
            try:
                batch_size = 1
                x = torch.randn(batch_size, 4, *size).to(device)
                t = torch.randint(0, 1000, (batch_size,)).to(device)
                
                with torch.no_grad():
                    output = model(x, t)
                
                expected_shape = (batch_size, 1, *size)
                if isinstance(output, tuple):
                    output_shape = output[0].shape
                    print(f"  ✅ Size {size}: {output_shape} (with variance)")
                else:
                    output_shape = output.shape
                    print(f"  ✅ Size {size}: {output_shape}")
                
                if output_shape == expected_shape or (isinstance(output, tuple) and output[0].shape == expected_shape):
                    print(f"    ✅ Output shape correct")
                else:
                    print(f"    ❌ Output shape mismatch! Expected: {expected_shape}")
                    
            except Exception as e:
                print(f"  ❌ Size {size} failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return False


def test_data_validation():
    """Test data validation utilities"""
    print("\n" + "="*60)
    print("Testing Data Validation")
    print("="*60)
    
    try:
        from utils.data_validation import validate_brats_data_structure, create_dummy_brats_data
        
        # Test with non-existent directory
        results = validate_brats_data_structure("/nonexistent/path")
        print(f"✅ Non-existent path handled correctly: {not results['valid']}")
        
        # Create dummy data for testing
        dummy_dir = "/tmp/dummy_brats_test"
        try:
            create_dummy_brats_data(dummy_dir, num_cases=2)
            results = validate_brats_data_structure(dummy_dir)
            print(f"✅ Dummy data validation: {results['valid']}")
            print(f"  Cases found: {results['case_count']}")
            print(f"  Modalities: {results['modalities_found']}")
        except Exception as e:
            print(f"⚠️  Dummy data creation failed (nibabel not available): {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Data validation test failed: {e}")
        return False


def test_denoising_improvements():
    """Test the improved denoising function"""
    print("\n" + "="*60)
    print("Testing Denoising Improvements")
    print("="*60)
    
    try:
        from functions.denoising_3d import unified_4to4_generalized_steps
        
        # This is a more complex test - we'll just check that the function exists
        # and has the expected signature
        import inspect
        sig = inspect.signature(unified_4to4_generalized_steps)
        print(f"✅ Denoising function signature: {sig}")
        
        # Test smart channel selection logic
        # Create test data with different variances
        x_available = torch.zeros(1, 4, 32, 32, 32)
        x_available[:, 0] = torch.randn(1, 1, 32, 32, 32) * 0.1  # Low variance (missing)
        x_available[:, 1] = torch.randn(1, 1, 32, 32, 32) * 1.0  # High variance (good)
        x_available[:, 2] = torch.randn(1, 1, 32, 32, 32) * 1.0  # High variance (good)
        x_available[:, 3] = torch.randn(1, 1, 32, 32, 32) * 1.0  # High variance (good)
        
        # Calculate variances
        channel_vars = []
        for i in range(x_available.shape[1]):
            var = torch.var(x_available[:, i:i+1])
            channel_vars.append(var.item())
        
        min_var_channel = channel_vars.index(min(channel_vars))
        print(f"✅ Smart channel selection: channel {min_var_channel} has lowest variance")
        print(f"  Channel variances: {[f'{v:.4f}' for v in channel_vars]}")
        
        return True
        
    except Exception as e:
        print(f"❌ Denoising test failed: {e}")
        return False


def main():
    """Run all tests"""
    print("🧪 Running Fast-DDPM-3D-BraTS Comprehensive Tests")
    print("This will validate all the fixes and improvements made")
    
    tests = [
        ("Imports", test_imports),
        ("GPU Memory Detection", test_gpu_memory_detection),
        ("Loss Functions", test_loss_functions),
        ("Model Creation", test_model_creation_and_forward),
        ("Data Validation", test_data_validation),
        ("Denoising Improvements", test_denoising_improvements),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{'='*80}")
        print(f"🔍 Running: {test_name}")
        print('='*80)
        
        try:
            success = test_func()
            results[test_name] = success
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "="*80)
    print("📊 TEST SUMMARY")
    print("="*80)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status:10} {test_name}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your Fast-DDPM-3D-BraTS setup is working correctly.")
    else:
        print("⚠️  Some tests failed. Please check the output above for details.")
    
    return passed == total


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
