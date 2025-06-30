#!/usr/bin/env python3
"""
Test script to verify loss function fixes
"""
import torch
import numpy as np
import sys
import os

# Add path
sys.path.append('.')
sys.path.append('..')

from functions.losses import sg_noise_estimation_loss, fast_ddpm_loss

def test_loss_functions():
    """Test that loss functions return reasonable values"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create dummy data
    batch_size = 1
    volume_size = (80, 80, 80)
    
    # Create mock inputs
    x_available = torch.randn(batch_size, 4, *volume_size, device=device) * 0.5  # Scale down
    x_target = torch.randn(batch_size, 1, *volume_size, device=device) * 0.5     # Scale down
    t = torch.randint(0, 100, (batch_size,), device=device)
    e = torch.randn_like(x_target, device=device)
    
    # Create dummy beta schedule
    betas = torch.linspace(0.0001, 0.02, 1000, device=device)
    
    # Mock model that returns random noise
    class MockModel:
        def __call__(self, x, t):
            return torch.randn_like(x[:, 0:1], device=x.device) * 0.1  # Small scale
    
    model = MockModel()
    
    # Test simple loss
    loss_simple = sg_noise_estimation_loss(model, x_available, x_target, t, e, betas)
    print(f"Simple loss: {loss_simple.item():.6f}")
    
    # Test fast ddpm loss with fixed variance
    loss_fast_fixed = fast_ddpm_loss(model, x_available, x_target, t, e, betas, var_type='fixed')
    print(f"Fast DDPM loss (fixed): {loss_fast_fixed.item():.6f}")
    
    print("Test completed!")

if __name__ == "__main__":
    test_loss_functions()
