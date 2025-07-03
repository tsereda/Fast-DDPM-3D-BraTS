#!/usr/bin/env python3
"""
Systematic debugging tests for gradient explosion in Fast-DDPM-3D-BraTS
Run this to isolate whether the problem is in the model or loss function.

Usage:
    python debug_gradient_explosion.py

This will run a series of tests to identify the root cause of gradient explosion.
"""

import torch
import sys
import os
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_model_only():
    """
    Test 1: Check if the model itself has gradient explosion without any diffusion math
    This isolates whether the problem is in model architecture vs loss computation
    """
    print("\n" + "="*60)
    print("TEST 1: Model Architecture Stability")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    try:
        # Import model
        from models.fast_ddpm_3d import FastDDPM3D
        
        # Create minimal config for testing with proper structure
        model_config = type('ModelConfig', (), {
            'ch': 32,                    # Reduced from typical 64
            'num_res_blocks': 1,         # Minimal blocks
            'attn_resolutions': [],      # No attention for now
            'ch_mult': [1, 2],          # Simple channel multipliers
            'resolution': 32,            # Small resolution for testing
            'in_channels': 4,            # BraTS input channels
            'out_ch': 1,                # Single output channel
            'dropout': 0.0,             # No dropout for cleaner test
            'resamp_with_conv': True,
            'num_timesteps': 10,        # Few timesteps for testing
        })()
        
        config = type('Config', (), {
            'model': model_config
        })()
        
        print("Creating model...")
        model = FastDDPM3D(config).to(device)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {total_params:,}")
        
        # Create simple input data (small values, similar to normalized medical data)
        B, C, H, W, D = 1, 4, 16, 16, 16  # Very small for testing
        print(f"Input shape: [{B}, {C}, {H}, {W}, {D}]")
        
        x = torch.randn(B, C, H, W, D).to(device) * 0.1  # Small values like medical data
        t = torch.zeros(B).to(device)  # Simple timestep
        
        print(f"Input data range: {x.min():.6f} to {x.max():.6f}")
        
        # Simple forward pass
        print("Running forward pass...")
        output = model(x, t)
        print(f"Output shape: {output.shape}")
        print(f"Output range: {output.min():.6f} to {output.max():.6f}")
        
        # Simple loss (just mean of output)
        loss = output.mean()
        print(f"Simple loss: {loss.item():.6f}")
        
        # Backward pass
        print("Running backward pass...")
        loss.backward()
        
        # Check gradients
        gradient_norms = []
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm(2).item()
                gradient_norms.append(grad_norm)
                if grad_norm > 10:
                    print(f"⚠️  Large gradient in {name}: {grad_norm:.3f}")
        
        total_norm = sum(gn**2 for gn in gradient_norms)**0.5
        max_norm = max(gradient_norms) if gradient_norms else 0
        
        print(f"Total gradient norm: {total_norm:.6f}")
        print(f"Max gradient norm: {max_norm:.6f}")
        
        # Verdict
        if total_norm > 100:
            print("🚨 MODEL ITSELF HAS GRADIENT EXPLOSION!")
            print("   Problem is likely in model architecture or initialization")
            return False
        else:
            print("✅ Model gradients are reasonable")
            print("   Problem is likely in the loss function")
            return True
            
    except Exception as e:
        print(f"❌ Error in model test: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_loss_components():
    """
    Test 2: Check each component of the loss computation for numerical issues
    """
    print("\n" + "="*60)
    print("TEST 2: Loss Function Components")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create test data
    B, C, H, W, D = 1, 4, 8, 8, 8
    
    print("Creating test data...")
    x_available = torch.rand(B, C, H, W, D).to(device) * 0.5  # [0, 0.5] range
    x_target = torch.rand(B, 1, H, W, D).to(device) * 0.5
    e = torch.randn_like(x_target)  # Standard Gaussian noise
    t = torch.randint(0, 10, (B,)).to(device)
    b = torch.linspace(0.0001, 0.02, 10).to(device)  # Beta schedule
    
    print(f"x_available range: {x_available.min():.6f} to {x_available.max():.6f}")
    print(f"x_target range: {x_target.min():.6f} to {x_target.max():.6f}")
    print(f"Original noise range: {e.min():.6f} to {e.max():.6f}")
    print(f"Timestep: {t.item()}")
    print(f"Beta schedule range: {b.min():.6f} to {b.max():.6f}")
    
    print("\n--- Alpha Computation ---")
    a = (1-b).cumprod(dim=0)
    print(f"Raw alpha_cumprod range: {a.min():.6f} to {a.max():.6f}")
    
    a = torch.clamp(a, min=1e-8, max=1.0)
    print(f"Clamped alpha_cumprod range: {a.min():.6f} to {a.max():.6f}")
    
    a_selected = a.index_select(0, t).view(-1, 1, 1, 1, 1)
    print(f"Selected alpha: {a_selected.item():.6f}")
    
    print("\n--- Noise Scaling Tests ---")
    noise_scales = [1.0, 0.1, 0.01, 0.001]
    
    for scale in noise_scales:
        e_scaled = e * scale
        print(f"\nNoise scale {scale}:")
        print(f"  Scaled noise range: {e_scaled.min():.6f} to {e_scaled.max():.6f}")
        
        # Compute noisy input
        alpha_coeff = a_selected.sqrt()
        noise_coeff = (1.0 - a_selected).sqrt()
        
        print(f"  Alpha coefficient: {alpha_coeff.item():.6f}")
        print(f"  Noise coefficient: {noise_coeff.item():.6f}")
        
        x_noisy = x_target * alpha_coeff + e_scaled * noise_coeff
        print(f"  Noisy input range: {x_noisy.min():.6f} to {x_noisy.max():.6f}")
        
        # Check for extreme values
        if x_noisy.abs().max() > 2:
            print(f"  🚨 NOISY INPUT TOO LARGE with scale {scale}")
        elif x_noisy.abs().max() > 1:
            print(f"  ⚠️  Noisy input somewhat large with scale {scale}")
        else:
            print(f"  ✅ Noisy input reasonable with scale {scale}")
    
    print("\n--- Recommended Fix ---")
    print("Based on the tests above, try noise scaling of 0.01 or 0.001")


def test_full_loss_function():
    """
    Test 3: Test the actual loss function with controlled inputs
    """
    print("\n" + "="*60)
    print("TEST 3: Full Loss Function Test")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    try:
        # Import the loss function
        from functions.losses import brats_4to1_loss
        
        # Create a minimal model for testing
        class SimpleModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv3d(4, 1, 3, padding=1)
            
            def forward(self, x, t):
                return self.conv(x)
        
        model = SimpleModel().to(device)
        
        # Create test data
        B, C, H, W, D = 1, 4, 8, 8, 8
        
        x_available = torch.rand(B, C, H, W, D).to(device) * 0.5
        x_target = torch.rand(B, 1, H, W, D).to(device) * 0.5
        e = torch.randn_like(x_target)
        t = torch.randint(0, 10, (B,)).to(device)
        b = torch.linspace(0.0001, 0.02, 10).to(device)
        
        print("Testing loss function...")
        loss = brats_4to1_loss(model, x_available, x_target, t, e, b, target_idx=0)
        
        print(f"Loss value: {loss.item():.6f}")
        
        if torch.isnan(loss) or torch.isinf(loss):
            print("🚨 LOSS IS NaN OR INFINITY!")
            return False
        elif loss.item() > 10:
            print("⚠️  Loss is quite large")
        else:
            print("✅ Loss value seems reasonable")
        
        # Test backward pass
        print("Testing backward pass...")
        loss.backward()
        
        # Check gradients
        total_norm = sum(p.grad.norm(2).item()**2 for p in model.parameters() if p.grad is not None)**0.5
        print(f"Gradient norm: {total_norm:.6f}")
        
        if total_norm > 100:
            print("🚨 GRADIENT EXPLOSION IN LOSS FUNCTION!")
            return False
        else:
            print("✅ Gradients from loss function are reasonable")
            return True
            
    except Exception as e:
        print(f"❌ Error in loss function test: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_hyperparameters():
    """
    Test 4: Check current hyperparameters for reasonableness
    """
    print("\n" + "="*60)
    print("TEST 4: Hyperparameter Analysis")
    print("="*60)
    
    # Test different learning rates
    print("Learning rate recommendations for 3D medical imaging:")
    print("  - Conservative: 1e-6")
    print("  - Moderate: 1e-5") 
    print("  - Aggressive: 1e-4")
    print("  - Too high: >1e-4")
    
    # Test beta schedule
    print("\nBeta schedule analysis:")
    b = torch.linspace(0.0001, 0.02, 1000)  # Typical schedule
    a = (1-b).cumprod(dim=0)
    
    print(f"  Beta range: {b.min():.6f} to {b.max():.6f}")
    print(f"  Alpha_cumprod range: {a.min():.6f} to {a.max():.6f}")
    print(f"  Final alpha (t=999): {a[-1]:.6f}")
    
    if a[-1] < 1e-6:
        print("  ⚠️  Alpha gets very small - may cause numerical issues")
    else:
        print("  ✅ Alpha schedule looks reasonable")


def run_all_tests():
    """Run all debugging tests in sequence"""
    print("🔍 Running Gradient Explosion Debug Tests")
    print("This will help identify the root cause of the gradient explosion issue.\n")
    
    # Test 1: Model stability
    model_ok = test_model_only()
    
    # Test 2: Loss components
    test_loss_components()
    
    # Test 3: Full loss function
    if model_ok:
        loss_ok = test_full_loss_function()
    else:
        print("\nSkipping loss function test since model has issues")
        loss_ok = False
    
    # Test 4: Hyperparameters
    test_hyperparameters()
    
    # Summary
    print("\n" + "="*60)
    print("DEBUGGING SUMMARY")
    print("="*60)
    
    if not model_ok:
        print("🎯 PRIMARY ISSUE: Model architecture has gradient explosion")
        print("   FIX: Check model initialization, reduce model size, or use gradient clipping")
    elif not loss_ok:
        print("🎯 PRIMARY ISSUE: Loss function causes gradient explosion")
        print("   FIX: Use smaller noise scaling (0.001) and stricter clamping")
    else:
        print("🤔 Both model and loss seem OK in isolation")
        print("   POSSIBLE CAUSES: Learning rate too high, mixed precision issues, or data problems")
    
    print("\n📋 RECOMMENDED IMMEDIATE FIXES:")
    print("1. Add gradient clipping: torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)")
    print("2. Reduce learning rate to 1e-6")
    print("3. Use noise scaling of 0.001 instead of 0.01")
    print("4. Add strict input clamping: torch.clamp(x_noisy, 0.0, 1.0)")


if __name__ == '__main__':
    run_all_tests()
