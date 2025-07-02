#!/usr/bin/env python3
"""
Quick fixes for current training issues
Apply these changes to improve training convergence
"""

def suggest_immediate_fixes():
    """Print immediate actionable fixes"""
    print("="*60)
    print("IMMEDIATE FIXES FOR CURRENT TRAINING")
    print("="*60)
    
    print("""
🔥 CRITICAL ISSUES IDENTIFIED:

1. ❌ LEARNING RATE MISMATCH:
   - Config has both lr=0.0001 and lr=0.00001
   - Fixed: Updated to consistent 0.0002

2. ❌ MISSING ATTENTION BLOCKS:
   - Config specifies attention but model didn't implement it
   - Fixed: Added 3D attention blocks at resolution 16

3. ❌ AGGRESSIVE BETA SCHEDULE:
   - beta_end=0.02 might be too high for [0,1] normalized data
   - Fixed: Reduced to beta_end=0.01

4. ❌ HARDCODED NORMALIZATION STATS:
   - Using generic BraTS stats, not your data-specific stats
   - Action needed: Run normalization computation script

📊 LOSS ANALYSIS (Your current training):
   - Loss: 0.59-0.71 range after 450 steps
   - This is reasonable for early training but should decrease
   - Expected: Should drop to 0.1-0.3 range with fixes

🔧 APPLY THESE FIXES:

1. RESTART TRAINING with updated config:
   ```bash
   python train.py --config configs/fast_ddpm_3d.yml --data_root /path/to/data
   ```

2. COMPUTE PROPER NORMALIZATION STATS:
   ```bash
   python scripts/compute_normalization_stats.py --data_root /path/to/data
   ```
   Then update the global_stats in brain_3d_unified.py

3. RUN DIAGNOSTICS:
   ```bash
   python scripts/training_diagnostics.py --data_root /path/to/data
   ```

⚡ EXPECTED IMPROVEMENTS:
   - Faster convergence due to higher LR (0.0002 vs 0.0001)
   - Better feature learning with attention blocks
   - More stable training with gentler diffusion schedule
   - Proper normalization will improve loss landscape

📈 MONITORING:
   - Loss should start decreasing more consistently
   - Watch for loss dropping below 0.5 within 1000 steps
   - Generated samples should show better anatomical structure
   
🚨 IF STILL NOT WORKING:
   - Check GPU memory usage
   - Verify data loading (no NaN values)
   - Consider smaller crop size if memory issues
   - Monitor gradient norms (should be 0.1-10 range)
""")

def print_config_changes():
    """Show what was changed in the config"""
    print("\n" + "="*60)
    print("CONFIG CHANGES MADE")
    print("="*60)
    
    changes = [
        ("training.learning_rate", "0.0001 → 0.0002", "Higher LR for better convergence"),
        ("optim.lr", "0.00001 → 0.0002", "Fixed LR mismatch"),
        ("diffusion.beta_end", "0.02 → 0.01", "Gentler diffusion schedule"),
        ("model attention", "Missing → Added", "3D attention blocks at resolution 16"),
        ("loss function", "Basic → Improved", "Better numerical stability and weighting")
    ]
    
    for param, change, reason in changes:
        print(f"✅ {param:20s}: {change:20s} - {reason}")

if __name__ == '__main__':
    suggest_immediate_fixes()
    print_config_changes()
    
    print("\n" + "="*60)
    print("NEXT STEPS")
    print("="*60)
    print("""
1. 🛑 STOP current training
2. 📊 Run: python scripts/training_diagnostics.py --data_root /your/data/path
3. 📈 Run: python scripts/compute_normalization_stats.py --data_root /your/data/path
4. 🔄 UPDATE normalization stats in data/brain_3d_unified.py
5. 🚀 RESTART training with fixed config
6. 📱 Monitor loss - should improve within 500-1000 steps

Expected result: Loss should converge to 0.1-0.3 range instead of staying at 0.6+
""")
