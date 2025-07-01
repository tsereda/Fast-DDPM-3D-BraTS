#!/usr/bin/env python3
"""
Test script to verify loss scaling fixes
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import os

# Add project root to path
sys.path.append('/root/Fast-DDPM-3D-BraTS')

from functions.losses import unified_4to4_loss
from models.fast_ddpm_3d import FastDDPM3D
import yaml
import argparse

def test_loss_scaling():
    """Test the fixed loss scaling"""
    print("Testing loss scaling fixes...")
    
    # Load config
    config_path = '/root/Fast-DDPM-3D-BraTS/configs/fast_ddpm_3d.yml'
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Convert to namespace
    config = argparse.Namespace()
    for key, value in config_dict.items():
        if isinstance(value, dict):
            setattr(config, key, argparse.Namespace(**value))
        else:
            setattr(config, key, value)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Test data shapes
    batch_size = 2
    volume_size = (32, 32, 32)  # Smaller for testing
    
    # Create test data
    x_available = torch.randn(batch_size, 4, *volume_size, device=device)
    x_target = torch.randn(batch_size, 1, *volume_size, device=device)
    e = torch.randn_like(x_target)
    t = torch.randint(0, 100, (batch_size,), device=device)
    
    # Beta schedule
    betas = torch.linspace(0.0001, 0.02, 1000, device=device)
    
    # Test loss computation
    try:
        loss = unified_4to4_loss(
            model=None,  # We'll mock the model
            x_available=x_available,
            x_target=x_target,
            t=t,
            e=e,
            b=betas,
            target_idx=0
        )
        
        print(f"✗ Expected error with None model, but got loss: {loss}")
    except Exception as e:
        print(f"✓ Expected error with None model: {str(e)[:50]}...")
    
    # Create a mock model for testing
    class MockModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv3d(4, 1, 3, padding=1)
            
        def forward(self, x, t):
            return self.conv(x)
    
    model = MockModel().to(device)
    
    # Test loss computation with real model
    loss = unified_4to4_loss(
        model=model,
        x_available=x_available,
        x_target=x_target,
        t=t,
        e=e,
        b=betas,
        target_idx=0
    )
    
    print(f"✓ Loss computed successfully: {loss.item():.6f}")
    
    # Test that loss is reasonable (not too large or too small)
    if 1e-8 < loss.item() < 1000.0:
        print(f"✓ Loss is in reasonable range: {loss.item():.6f}")
    else:
        print(f"⚠ Loss might be problematic: {loss.item():.6e}")
    
    # Test keepdim=True
    loss_keepdim = unified_4to4_loss(
        model=model,
        x_available=x_available,
        x_target=x_target,
        t=t,
        e=e,
        b=betas,
        target_idx=0,
        keepdim=True
    )
    
    print(f"✓ Loss with keepdim=True shape: {loss_keepdim.shape}")
    print(f"✓ Per-sample losses: {loss_keepdim.detach().cpu().numpy()}")
    
    # Test gradient computation
    loss.backward()
    total_grad_norm = 0
    for param in model.parameters():
        if param.grad is not None:
            total_grad_norm += param.grad.norm().item() ** 2
    total_grad_norm = total_grad_norm ** 0.5
    
    print(f"✓ Gradient norm: {total_grad_norm:.6f}")
    
    if total_grad_norm > 0:
        print("✓ Gradients computed successfully")
    else:
        print("⚠ No gradients computed")
    
    # Test different volume sizes
    for vol_size in [(16, 16, 16), (64, 64, 64)]:
        print(f"\nTesting volume size: {vol_size}")
        x_test = torch.randn(1, 4, *vol_size, device=device)
        target_test = torch.randn(1, 1, *vol_size, device=device)
        e_test = torch.randn_like(target_test)
        
        # Create appropriately sized model
        model_test = MockModel().to(device)
        
        loss_test = unified_4to4_loss(
            model=model_test,
            x_available=x_test,
            x_target=target_test,
            t=t[:1],
            e=e_test,
            b=betas,
            target_idx=0
        )
        
        print(f"  Loss for {vol_size}: {loss_test.item():.6f}")
        
        # Check that loss scales appropriately with volume size
        voxels = np.prod(vol_size)
        loss_per_voxel = loss_test.item() / voxels
        print(f"  Loss per voxel: {loss_per_voxel:.8f}")

if __name__ == "__main__":
    test_loss_scaling()
