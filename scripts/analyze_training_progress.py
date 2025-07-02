#!/usr/bin/env python3
"""
Analyze current training progress and provide recommendations
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add path
sys.path.append('.')
sys.path.append('..')

def analyze_loss_trend():
    """Analyze the loss trend from your training output"""
    # Based on your training log:
    losses = [0.880869, 0.715821, 0.634339, 0.618971, 0.607582, 0.614050, 
              0.709349, 0.590510, 0.593726, 0.590216]
    steps = [25, 50, 100, 150, 200, 250, 300, 350, 400, 450]
    
    print("=== CURRENT TRAINING ANALYSIS ===")
    print(f"Steps analyzed: {len(steps)}")
    print(f"Loss range: [{min(losses):.4f}, {max(losses):.4f}]")
    print(f"Current loss: {losses[-1]:.4f}")
    
    # Calculate trend
    trend = np.polyfit(steps, losses, 1)[0]
    print(f"Loss trend: {trend:.6f} per step")
    
    if trend < 0:
        print("✅ Loss is decreasing overall")
    else:
        print("❌ Loss is increasing or plateauing")
    
    # Calculate improvement rate
    initial_loss = losses[0]
    current_loss = losses[-1]
    improvement = (initial_loss - current_loss) / initial_loss * 100
    print(f"Improvement: {improvement:.1f}% from initial loss")
    
    # Volatility check
    recent_std = np.std(losses[-5:])
    print(f"Recent volatility (std): {recent_std:.4f}")
    
    print("\n=== PREDICTIONS ===")
    if trend < 0:
        # Estimate steps to reach target loss
        target_loss = 0.3
        if trend < 0:
            steps_to_target = (current_loss - target_loss) / abs(trend)
            print(f"Estimated steps to reach loss {target_loss}: {steps_to_target:.0f}")
    
    print("\n=== RECOMMENDATIONS ===")
    
    if current_loss > 0.6:
        print("🔧 Loss still high. Consider:")
        print("  - Verify normalization stats are applied (just updated)")
        print("  - Check if learning rate needs adjustment")
        print("  - Monitor for 1000+ more steps")
    
    if recent_std > 0.05:
        print("⚠️  Loss is volatile. Consider:")
        print("  - Reducing learning rate")
        print("  - Adding gradient clipping")
        print("  - Increasing batch size if possible")
    
    if trend >= 0:
        print("❌ Loss not improving. Check:")
        print("  - Learning rate (might be too high/low)")
        print("  - Model capacity")
        print("  - Data preprocessing")
    else:
        print("✅ Training progressing normally")
        print("  - Continue monitoring")
        print("  - Loss should reach 0.1-0.3 range")


def check_training_efficiency():
    """Check if training is efficient"""
    print("\n=== TRAINING EFFICIENCY CHECK ===")
    
    # Your current speed: 1.53-1.55 it/s
    current_speed = 1.54  # it/s
    batch_size = 1  # Based on your setup
    
    samples_per_hour = current_speed * 3600 * batch_size
    print(f"Current training speed: {current_speed:.2f} it/s")
    print(f"Samples per hour: {samples_per_hour:.0f}")
    
    # Estimate epoch time
    samples_per_epoch = 1251  # From your training
    epoch_time_hours = samples_per_epoch / samples_per_hour
    print(f"Estimated time per epoch: {epoch_time_hours:.2f} hours")
    
    print("\n💡 EFFICIENCY TIPS:")
    print("  - Consider increasing batch size if memory allows")
    print("  - Use gradient accumulation if batch size limited")
    print("  - Monitor GPU utilization")


def recommend_next_steps():
    """Recommend immediate next steps"""
    print("\n" + "="*60)
    print("IMMEDIATE NEXT STEPS")
    print("="*60)
    
    print("""
1. 🔄 RESTART TRAINING with updated normalization:
   - The normalization stats have been updated in your dataset
   - This should improve convergence
   
2. 📊 MONITOR these metrics for next 1000 steps:
   - Loss should trend downward
   - Target: reach ~0.4 in next 1000 steps
   - Watch for volatility (should be smooth)
   
3. 🎯 EARLY STOPPING criteria:
   - If loss plateaus above 0.5 for 500+ steps → adjust LR
   - If loss becomes very volatile → reduce LR or add grad clip
   - If loss explodes → much lower LR needed
   
4. 🔍 QUALITY CHECK after 2000 steps:
   - Generate sample images
   - Check if different modalities are being learned
   - Verify generated anatomy makes sense
   
5. 📈 EXPECTED TIMELINE:
   - Steps 0-1000: Loss 0.8 → 0.4-0.5
   - Steps 1000-5000: Loss 0.4 → 0.2-0.3  
   - Steps 5000+: Fine-tuning, loss 0.1-0.3
""")


if __name__ == '__main__':
    analyze_loss_trend()
    check_training_efficiency()
    recommend_next_steps()
