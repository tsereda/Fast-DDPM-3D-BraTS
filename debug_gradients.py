#!/usr/bin/env python3
"""
Systematic debugging script for gradient explosion in 3D Fast-DDPM
Run this BEFORE training to identify the root cause
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import sys

# Add your project paths
sys.path.append('.')
sys.path.append('..')

# Your imports
from data.brain_3d_unified import BraTS3DUnifiedDataset
from models.fast_ddpm_3d import FastDDPM3D
from functions.losses import brats_4to1_loss

def debug_step_by_step(data_root, config_path=None):
    """
    Step-by-step debugging to isolate gradient explosion cause
    """
    print("🔍 SYSTEMATIC GRADIENT EXPLOSION DEBUGGING")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Step 1: Test data loading
    print("\n1️⃣ TESTING DATA LOADING...")
    try:
        dataset = BraTS3DUnifiedDataset(data_root=data_root, phase='train')
        batch = dataset[0]
        
        inputs = batch['input'].unsqueeze(0)  # Add batch dim
        targets = batch['target'].unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
        
        print(f"✅ Data loaded successfully")
        print(f"   Input shape: {inputs.shape}")
        print(f"   Target shape: {targets.shape}")
        print(f"   Input range: [{inputs.min():.3f}, {inputs.max():.3f}]")
        print(f"   Target range: [{targets.min():.3f}, {targets.max():.3f}]")
        
        # Check for invalid values
        if torch.isnan(inputs).any() or torch.isnan(targets).any():
            print("🚨 NaN values detected in data!")
            return False
            
        if torch.isinf(inputs).any() or torch.isinf(targets).any():
            print("🚨 Inf values detected in data!")
            return False
            
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        return False
    
    # Step 2: Test minimal model
    print("\n2️⃣ TESTING MINIMAL MODEL...")
    
    class MinimalTest3D(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv3d(4, 32, 3, padding=1)
            self.conv2 = nn.Conv3d(32, 1, 3, padding=1)
            self.relu = nn.ReLU()
            
            # Standard initialization
            for m in self.modules():
                if isinstance(m, nn.Conv3d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
        
        def forward(self, x, t):
            x = self.relu(self.conv1(x))
            x = self.conv2(x)
            return x
    
    minimal_model = MinimalTest3D().to(device)
    inputs = inputs.to(device)
    targets = targets.to(device)
    
    # Test forward pass
    try:
        t = torch.tensor([0]).to(device)
        output = minimal_model(inputs, t.float())
        print(f"✅ Minimal model forward pass successful")
        print(f"   Output shape: {output.shape}")
        print(f"   Output range: [{output.min():.3f}, {output.max():.3f}]")
        
        if torch.isnan(output).any() or torch.isinf(output).any():
            print("🚨 Model output contains NaN/Inf!")
            return False
            
    except Exception as e:
        print(f"❌ Minimal model forward failed: {e}")
        return False
    
    # Step 3: Test loss computation with different noise scales
    print("\n3️⃣ TESTING LOSS COMPUTATION...")
    
    # Create beta schedule
    betas = torch.linspace(0.0001, 0.02, 1000).to(device)
    
    noise_scales = [1.0, 0.1, 0.01, 0.001]
    
    for scale in noise_scales:
        print(f"\n   Testing noise scale: {scale}")
        
        # Generate noise
        e = torch.randn_like(targets) * scale
        t = torch.randint(0, 100, (1,)).to(device)
        
        # Test alpha computation
        a = (1-betas).cumprod(dim=0)
        a = torch.clamp(a, min=1e-6, max=0.9999)
        a_t = a.index_select(0, t).view(-1, 1, 1, 1, 1)
        
        print(f"     Alpha value: {a_t.item():.6f}")
        
        # Forward diffusion
        x_noisy = targets * a_t.sqrt() + e * (1.0 - a_t).sqrt()
        print(f"     Noisy input range: [{x_noisy.min():.3f}, {x_noisy.max():.3f}]")
        
        # Create model input
        model_input = inputs.clone()
        target_idx = 0
        model_input[:, target_idx:target_idx+1] = x_noisy
        
        # Forward pass
        minimal_model.zero_grad()
        try:
            output = minimal_model(model_input, t.float())
            loss = F.mse_loss(output, e)
            
            print(f"     Loss value: {loss.item():.6f}")
            
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"     🚨 Loss is NaN/Inf at scale {scale}!")
                continue
            
            # Backward pass
            loss.backward()
            
            # Check gradients
            total_norm = 0
            max_grad = 0
            for name, param in minimal_model.named_parameters():
                if param.grad is not None:
                    param_norm = param.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
                    max_grad = max(max_grad, param.grad.abs().max().item())
            
            total_norm = total_norm ** 0.5
            print(f"     Gradient norm: {total_norm:.3f}")
            print(f"     Max gradient: {max_grad:.6f}")
            
            if total_norm > 100:
                print(f"     🚨 Large gradients at scale {scale}!")
            else:
                print(f"     ✅ Gradients OK at scale {scale}")
                
        except Exception as e:
            print(f"     ❌ Error at scale {scale}: {e}")
    
    # Step 4: Test your actual model with minimal setup
    print("\n4️⃣ TESTING ACTUAL MODEL...")
    
    # Create a minimal config for testing
    class MinimalConfig:
        def __init__(self):
            self.model = self
            self.data = self
            self.ch = 32  # Much smaller than default
            self.out_ch = 1
            self.ch_mult = (1, 2)  # Much simpler
            self.num_res_blocks = 1  # Fewer blocks
            self.dropout = 0.1
            self.in_channels = 4
            self.crop_size = (64, 64, 64)
            self.resamp_with_conv = True
            self.attn_resolutions = []  # No attention for testing
    
    try:
        config = MinimalConfig()
        actual_model = FastDDPM3D(config).to(device)
        
        # Count parameters
        total_params = sum(p.numel() for p in actual_model.parameters())
        print(f"   Model parameters: {total_params:,}")
        
        # Test forward pass
        actual_model.zero_grad()
        t = torch.tensor([0]).to(device)
        output = actual_model(inputs, t.float())
        
        print(f"   ✅ Actual model forward pass successful")
        print(f"   Output range: [{output.min():.3f}, {output.max():.3f}]")
        
        # Test with actual loss function
        e = torch.randn_like(targets) * 0.01  # Conservative scaling
        t = torch.randint(0, 100, (1,)).to(device)
        
        loss = brats_4to1_loss(actual_model, inputs, targets, t, e, b=betas, target_idx=0)
        print(f"   Loss value: {loss.item():.6f}")
        
        if torch.isnan(loss) or torch.isinf(loss):
            print("   🚨 Actual loss is NaN/Inf!")
            return False
        
        loss.backward()
        
        # Check gradients
        total_norm = 0
        for param in actual_model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        
        print(f"   Gradient norm: {total_norm:.3f}")
        
        if total_norm > 100:
            print("   🚨 Large gradients with actual model!")
            
            # Debug which layers have large gradients
            print("\n   📊 Gradient analysis by layer:")
            for name, param in actual_model.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    if grad_norm > 1.0:
                        print(f"     {name}: {grad_norm:.3f}")
        else:
            print("   ✅ Gradients OK with actual model")
            
    except Exception as e:
        print(f"   ❌ Actual model test failed: {e}")
        return False
    
    # Step 5: Recommendations
    print("\n5️⃣ RECOMMENDATIONS")
    print("=" * 40)
    
    if total_norm > 100:
        print("🔧 FIXES TO TRY:")
        print("1. Reduce model complexity:")
        print("   - ch = 16 instead of default")
        print("   - ch_mult = (1, 2) instead of (1, 2, 4)")
        print("   - num_res_blocks = 1")
        print("   - Remove attention blocks initially")
        
        print("\n2. Adjust training parameters:")
        print("   - Learning rate: 1e-5 or 1e-6")
        print("   - Noise scaling: 0.001")
        print("   - Gradient clipping: 0.5")
        
        print("\n3. Try different approaches:")
        print("   - 2.5D instead of full 3D")
        print("   - Smaller crop sizes (32³ instead of 64³)")
        print("   - Progressive training")
    else:
        print("✅ Model appears stable with minimal test!")
        print("   The issue might be in the training loop or data loading")
    
    return total_norm <= 100


def create_minimal_config():
    """
    Create a minimal config file for stable training
    """
    config = {
        'model': {
            'ch': 16,
            'out_ch': 1,
            'ch_mult': [1, 2],
            'num_res_blocks': 1,
            'dropout': 0.1,
            'in_channels': 4,
            'resamp_with_conv': True,
            'attn_resolutions': []
        },
        'data': {
            'crop_size': [32, 32, 32],  # Smaller for testing
            'num_workers': 2
        },
        'training': {
            'batch_size': 1,
            'epochs': 10,
            'learning_rate': 1e-5,
            'weight_decay': 0
        },
        'diffusion': {
            'beta_schedule': 'linear',
            'beta_start': 0.0001,
            'beta_end': 0.02,
            'num_diffusion_timesteps': 1000
        }
    }
    
    return config


if __name__ == '__main__':
    import sys
    if len(sys.argv) != 2:
        print("Usage: python debug_gradients.py <data_root>")
        sys.exit(1)
    
    data_root = sys.argv[1]
    
    # Run debugging
    success = debug_step_by_step(data_root)
    
    if not success:
        print("\n❌ DEBUGGING REVEALED CRITICAL ISSUES")
        print("Fix the identified problems before training")
    else:
        print("\n✅ DEBUGGING COMPLETED SUCCESSFULLY")
        print("Model appears stable for training")
    
    # Save minimal config
    import yaml
    minimal_config = create_minimal_config()
    with open('minimal_config.yml', 'w') as f:
        yaml.dump(minimal_config, f, default_flow_style=False)
    print(f"\n💾 Saved minimal_config.yml for stable training")