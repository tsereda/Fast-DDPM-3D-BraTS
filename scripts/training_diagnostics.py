#!/usr/bin/env python3
"""
Training diagnostic script for 3D Fast-DDPM BraTS
Helps identify common training issues and provides solutions
"""

import torch
import numpy as np
import argparse
import sys
import os

# Add path
sys.path.append('.')
sys.path.append('..')

from data.brain_3d_unified import BraTS3DUnifiedDataset
from models.fast_ddpm_3d import FastDDPM3D
from training_utils import load_config


def check_data_normalization(dataset, num_samples=10):
    """Check if data normalization is working properly"""
    print("=== DATA NORMALIZATION CHECK ===")
    
    all_inputs = []
    all_targets = []
    
    for i in range(min(num_samples, len(dataset))):
        sample = dataset[i]
        all_inputs.append(sample['input'])
        all_targets.append(sample['target'])
    
    inputs = torch.stack(all_inputs)
    targets = torch.stack(all_targets)
    
    print(f"Input range: [{inputs.min():.4f}, {inputs.max():.4f}]")
    print(f"Target range: [{targets.min():.4f}, {targets.max():.4f}]")
    print(f"Input mean: {inputs.mean():.4f}, std: {inputs.std():.4f}")
    print(f"Target mean: {targets.mean():.4f}, std: {targets.std():.4f}")
    
    # Check for issues
    issues = []
    if inputs.min() < -0.1 or inputs.max() > 1.1:
        issues.append("❌ Input data not in [0,1] range")
    if targets.min() < -0.1 or targets.max() > 1.1:
        issues.append("❌ Target data not in [0,1] range")
    if inputs.mean() < 0.05 or inputs.mean() > 0.95:
        issues.append("❌ Input mean suggests poor normalization")
    if targets.mean() < 0.05 or targets.mean() > 0.95:
        issues.append("❌ Target mean suggests poor normalization")
    
    if issues:
        print("\nISSUES FOUND:")
        for issue in issues:
            print(f"  {issue}")
        print("\n🔧 SOLUTION: Run normalization statistics script:")
        print("  python scripts/compute_normalization_stats.py --data_root /path/to/your/data")
    else:
        print("✅ Data normalization looks good!")
    
    return len(issues) == 0


def check_model_outputs(model, dataset, device):
    """Check if model outputs are reasonable"""
    print("\n=== MODEL OUTPUT CHECK ===")
    
    model.eval()
    sample = dataset[0]
    
    with torch.no_grad():
        inputs = sample['input'].unsqueeze(0).to(device)
        target_idx = sample['target_idx']
        
        # Test with different timesteps
        timesteps = [0, 100, 500, 999]
        
        for t_val in timesteps:
            t = torch.tensor([t_val]).to(device)
            output = model(inputs, t.float())
            
            print(f"  t={t_val:3d}: output range [{output.min():.4f}, {output.max():.4f}], "
                  f"mean={output.mean():.4f}, std={output.std():.4f}")
    
    # Check for issues
    if output.std() < 0.01:
        print("❌ Model outputs have very low variance - possible dead neurons")
        return False
    elif output.std() > 2.0:
        print("❌ Model outputs have very high variance - possible training instability")
        return False
    else:
        print("✅ Model outputs look reasonable!")
        return True


def check_loss_behavior(losses):
    """Analyze loss behavior for common issues"""
    print("\n=== LOSS BEHAVIOR ANALYSIS ===")
    
    if len(losses) < 10:
        print("Not enough loss values to analyze")
        return
    
    recent_losses = losses[-50:]  # Last 50 losses
    loss_mean = np.mean(recent_losses)
    loss_std = np.std(recent_losses)
    loss_trend = np.polyfit(range(len(recent_losses)), recent_losses, 1)[0]
    
    print(f"Recent loss mean: {loss_mean:.4f}")
    print(f"Recent loss std: {loss_std:.4f}")
    print(f"Loss trend: {'📈 increasing' if loss_trend > 0 else '📉 decreasing'} ({loss_trend:.6f})")
    
    # Analyze issues
    issues = []
    if loss_mean > 1.0:
        issues.append("❌ Loss too high - possible learning rate or normalization issues")
    if loss_std > 0.5:
        issues.append("❌ Loss too volatile - consider gradient clipping or lower LR")
    if abs(loss_trend) < 1e-6:
        issues.append("❌ Loss plateaued - model not learning")
    if loss_mean < 0.001:
        issues.append("❌ Loss suspiciously low - check for bugs")
    
    if issues:
        print("\nISSUES FOUND:")
        for issue in issues:
            print(f"  {issue}")
    else:
        print("✅ Loss behavior looks healthy!")


def check_gradient_health(model):
    """Check gradient magnitudes for training health"""
    print("\n=== GRADIENT HEALTH CHECK ===")
    
    total_norm = 0.0
    param_count = 0
    zero_grad_count = 0
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
            param_count += 1
            
            if param_norm < 1e-7:
                zero_grad_count += 1
    
    total_norm = total_norm ** (1. / 2)
    
    print(f"Total gradient norm: {total_norm:.4f}")
    print(f"Parameters with gradients: {param_count}")
    print(f"Parameters with near-zero gradients: {zero_grad_count}")
    
    if total_norm < 1e-6:
        print("❌ Gradients too small - vanishing gradient problem")
        return False
    elif total_norm > 100:
        print("❌ Gradients too large - exploding gradient problem")
        return False
    elif zero_grad_count > param_count * 0.1:
        print("❌ Too many parameters have near-zero gradients")
        return False
    else:
        print("✅ Gradient health looks good!")
        return True


def print_recommendations():
    """Print training recommendations"""
    print("\n" + "="*60)
    print("TRAINING RECOMMENDATIONS")
    print("="*60)
    print("""
🔧 IMMEDIATE FIXES:
1. Run normalization stats: python scripts/compute_normalization_stats.py
2. Update learning rate to 0.0002 (done in config)
3. Consider reducing beta_end to 0.01 (done in config)

📊 MONITORING:
- Loss should decrease from ~0.7 to ~0.1-0.3 over time
- Good convergence: loss decreases smoothly with small oscillations
- Bad signs: loss plateaus above 0.5, or becomes too volatile

⚡ PERFORMANCE TIPS:
- Use mixed precision (already enabled)
- Monitor GPU memory usage
- Consider gradient accumulation for larger effective batch size

🎯 QUALITY INDICATORS:
- Loss converging to 0.1-0.3 range
- Generated samples showing anatomical structure
- Model learning different modality characteristics

🚨 RED FLAGS:
- Loss > 1.0 after 1000 steps → check normalization
- Loss oscillating wildly → reduce learning rate
- Loss plateauing quickly → check attention/model capacity
""")


def main():
    parser = argparse.ArgumentParser(description='Diagnose 3D Fast-DDPM training')
    parser.add_argument('--config', type=str, default='configs/fast_ddpm_3d.yml', 
                       help='Config file path')
    parser.add_argument('--data_root', type=str, required=True, 
                       help='Path to BraTS data')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to model checkpoint (optional)')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {device}")
    print(f"Data root: {args.data_root}")
    
    # Load dataset
    print("Loading dataset...")
    dataset = BraTS3DUnifiedDataset(
        data_root=args.data_root,
        phase='train',
        crop_size=config.data.crop_size
    )
    
    # Check data normalization
    data_ok = check_data_normalization(dataset)
    
    # Load model
    print("Loading model...")
    model = FastDDPM3D(config).to(device)
    
    if args.checkpoint and os.path.exists(args.checkpoint):
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from {args.checkpoint}")
    
    # Check model outputs
    model_ok = check_model_outputs(model, dataset, device)
    
    # Print recommendations
    print_recommendations()
    
    print("\n" + "="*60)
    print("DIAGNOSTIC COMPLETE")
    print("="*60)
    
    if data_ok and model_ok:
        print("✅ No major issues detected. Training should proceed normally.")
    else:
        print("❌ Issues detected. Please address the problems above.")


if __name__ == '__main__':
    main()
