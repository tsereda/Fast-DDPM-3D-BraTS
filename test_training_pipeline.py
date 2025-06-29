#!/usr/bin/env python3
"""
Test script to verify the training pipeline works correctly
Run this before starting actual training
"""
import torch
import yaml
import sys
import traceback
from pathlib import Path

# Add current directory to path
sys.path.append('.')


def test_model_creation():
    """Test model creation and basic forward pass"""
    print("="*60)
    print("Testing Model Creation")
    print("="*60)
    
    # Load config
    config_path = 'configs/fast_ddpm_3d.yml'
    if not Path(config_path).exists():
        print(f"❌ Config file not found: {config_path}")
        return False
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Mock config namespace
    class Config:
        def __init__(self, d):
            for k, v in d.items():
                if isinstance(v, dict):
                    setattr(self, k, Config(v))
                else:
                    setattr(self, k, v)
    
    config = Config(config)
    
    # Ensure volume_size is tuple
    if hasattr(config.data, 'volume_size'):
        config.data.volume_size = tuple(config.data.volume_size)
    else:
        config.data.volume_size = (96, 96, 96)
    
    try:
        from models.diffusion_3d import Model3D
        model = Model3D(config)
        print("✅ Model created successfully")
        print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
        return True
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        traceback.print_exc()
        return False


def test_forward_pass():
    """Test complete forward pass with loss computation"""
    print("\n" + "="*60)
    print("Testing Forward Pass and Loss")
    print("="*60)
    
    # Load config
    with open('configs/fast_ddpm_3d.yml', 'r') as f:
        config = yaml.safe_load(f)
    
    class Config:
        def __init__(self, d):
            for k, v in d.items():
                if isinstance(v, dict):
                    setattr(self, k, Config(v))
                else:
                    setattr(self, k, v)
    
    config = Config(config)
    config.data.volume_size = tuple(config.data.volume_size)
    
    try:
        from models.diffusion_3d import Model3D
        from functions.losses import sg_noise_estimation_loss
        
        # Create model
        model = Model3D(config)
        model.eval()  # Set to eval mode for testing
        
        # Create dummy data
        batch_size = 1
        volume_size = config.data.volume_size
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        inputs = torch.randn(batch_size, 4, *volume_size).to(device)
        targets = torch.randn(batch_size, 1, *volume_size).to(device)
        t = torch.randint(0, 1000, (batch_size,)).to(device)
        e = torch.randn_like(targets).to(device)
        betas = torch.linspace(0.0001, 0.02, 1000).to(device)
        
        print(f"Device: {device}")
        print(f"Input shape: {inputs.shape}")
        print(f"Target shape: {targets.shape}")
        
        # Move model to device
        model = model.to(device)
        
        # Test model output directly
        print("\nTesting model output...")
        model_input = inputs.clone()
        model_input[:, 0:1] = targets
        
        with torch.no_grad():
            output = model(model_input, t)
        
        if isinstance(output, tuple):
            print(f"✅ Model output is tuple (variance learning)")
            print(f"  Mean shape: {output[0].shape}")
            print(f"  Variance shape: {output[1].shape}")
        else:
            print(f"✅ Model output shape: {output.shape}")
        
        # Test loss computation
        print("\nTesting loss computation...")
        try:
            loss = sg_noise_estimation_loss(model, inputs, targets, t, e, betas)
            print(f"✅ Loss computed successfully: {loss.item():.6f}")
            
            # Check if loss is reasonable
            if torch.isnan(loss):
                print("⚠️  Loss is NaN!")
                return False
            elif loss.item() > 100:
                print("⚠️  Loss seems very high!")
            
            return True
            
        except Exception as e:
            print(f"❌ Loss computation failed: {e}")
            traceback.print_exc()
            return False
            
    except Exception as e:
        print(f"❌ Forward pass test failed: {e}")
        traceback.print_exc()
        return False


def test_data_loading():
    """Test data loading (with dummy data)"""
    print("\n" + "="*60)
    print("Testing Data Loading")
    print("="*60)
    
    try:
        from data.brain_3d_unified import BraTS3DUnifiedDataset
        print("✅ Dataset import successful")
        
        # Create a dummy dataset path
        dummy_path = Path("./data/dummy_brats")
        dummy_path.mkdir(parents=True, exist_ok=True)
        
        # Try to create dataset (will likely fail with dummy path)
        try:
            dataset = BraTS3DUnifiedDataset(
                data_root=str(dummy_path),
                phase='train',
                volume_size=(64, 64, 64)
            )
            print(f"✅ Dataset created with {len(dataset)} cases")
        except ValueError as e:
            print(f"⚠️  Expected error with dummy data: {e}")
            return True  # This is expected
            
    except Exception as e:
        print(f"❌ Data loading test failed: {e}")
        traceback.print_exc()
        return False


def test_training_step():
    """Test a complete training step"""
    print("\n" + "="*60)
    print("Testing Complete Training Step")
    print("="*60)
    
    # Load config
    with open('configs/fast_ddpm_3d.yml', 'r') as f:
        config = yaml.safe_load(f)
    
    class Config:
        def __init__(self, d):
            for k, v in d.items():
                if isinstance(v, dict):
                    setattr(self, k, Config(v))
                else:
                    setattr(self, k, v)
    
    config = Config(config)
    config.data.volume_size = tuple(config.data.volume_size)
    
    try:
        from models.diffusion_3d import Model3D
        from functions.losses import sg_noise_estimation_loss
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create model and optimizer
        model = Model3D(config).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        
        # Create dummy batch
        batch_size = 1
        volume_size = config.data.volume_size
        
        inputs = torch.randn(batch_size, 4, *volume_size).to(device)
        targets = torch.randn(batch_size, 1, *volume_size).to(device)
        
        # Training parameters
        betas = torch.linspace(0.0001, 0.02, 1000).to(device)
        timesteps = 10
        
        # Get timestep schedule (Fast-DDPM style)
        t_intervals = torch.arange(0, 1000, 1000 // timesteps)
        
        # Simulate training step
        model.train()
        optimizer.zero_grad()
        
        # Sample timestep
        n = inputs.size(0)
        idx = torch.randint(0, len(t_intervals), size=(n,))
        t = t_intervals[idx].to(device)
        
        # Random noise
        e = torch.randn_like(targets)
        
        # Compute loss
        loss = sg_noise_estimation_loss(model, inputs, targets, t, e, betas)
        
        print(f"Loss: {loss.item():.6f}")
        
        # Backward pass
        loss.backward()
        
        # Check gradients
        total_grad_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                total_grad_norm += p.grad.data.norm(2).item() ** 2
        total_grad_norm = total_grad_norm ** 0.5
        
        print(f"Total gradient norm: {total_grad_norm:.6f}")
        
        if total_grad_norm == 0:
            print("⚠️  No gradients computed!")
            return False
        elif total_grad_norm > 1000:
            print("⚠️  Gradient norm very high!")
        
        # Optimizer step
        optimizer.step()
        
        print("✅ Training step completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Training step failed: {e}")
        traceback.print_exc()
        return False


def test_memory_usage():
    """Test memory usage with different volume sizes"""
    print("\n" + "="*60)
    print("Testing Memory Usage")
    print("="*60)
    
    if not torch.cuda.is_available():
        print("⚠️  CUDA not available, skipping memory test")
        return True
    
    try:
        from models.diffusion_3d import Model3D
        from utils.gpu_memory import get_gpu_memory_gb
        
        # Get GPU memory
        gpu_memory = get_gpu_memory_gb()
        print(f"GPU Memory: {gpu_memory:.1f} GB")
        
        # Test different volume sizes
        test_sizes = [(32, 32, 32), (64, 64, 64), (96, 96, 96)]
        
        for size in test_sizes:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            try:
                # Create minimal config
                class MinConfig:
                    class model:
                        ch = 48
                        out_ch = 1
                        ch_mult = [1, 2, 4, 8]
                        num_res_blocks = 2
                        attn_resolutions = [16]
                        dropout = 0.1
                        in_channels = 4
                        var_type = 'fixed'
                        resamp_with_conv = True
                    class data:
                        volume_size = size
                
                config = MinConfig()
                
                model = Model3D(config).cuda()
                x = torch.randn(1, 4, *size).cuda()
                t = torch.randint(0, 1000, (1,)).cuda()
                
                with torch.no_grad():
                    _ = model(x, t)
                
                peak_memory = torch.cuda.max_memory_allocated() / (1024**3)
                print(f"✅ Volume size {size}: Peak memory {peak_memory:.2f} GB")
                
                del model, x, t
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"❌ Volume size {size}: Out of memory")
                else:
                    raise
        
        return True
        
    except Exception as e:
        print(f"❌ Memory test failed: {e}")
        return False


def main():
    """Run all tests"""
    print("🧪 Fast-DDPM-3D-BraTS Training Pipeline Test")
    print("This will verify all components work correctly")
    
    tests = [
        ("Model Creation", test_model_creation),
        ("Forward Pass", test_forward_pass),
        ("Data Loading", test_data_loading),
        ("Training Step", test_training_step),
        ("Memory Usage", test_memory_usage),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            print(f"\n{'='*80}")
            results[test_name] = test_func()
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
        print("\n🎉 All tests passed! Your training pipeline is ready.")
        print("\nNext steps:")
        print("1. Prepare your BraTS data")
        print("2. Run: python scripts/train_3d.py --data_root /path/to/brats/data")
    else:
        print("\n⚠️  Some tests failed. Please check the output above.")
        print("Common issues:")
        print("- Import errors: Make sure all files are in the correct locations")
        print("- CUDA errors: Check your GPU and PyTorch installation")
        print("- Config errors: Verify configs/fast_ddpm_3d.yml exists")
    
    return passed == total


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)