#!/usr/bin/env python3
"""
Quick test script for W&B integration
"""

import torch
import wandb
import numpy as np
from pathlib import Path

def test_wandb_minimal():
    """Test minimal W&B functionality"""
    print("Testing W&B integration...")
    
    # Check if wandb is installed
    try:
        import wandb
        print("✅ W&B is installed")
    except ImportError:
        print("❌ W&B not installed. Install with: pip install wandb")
        return False
    
    # Check if logged in
    try:
        wandb.ensure_configured()
        print("✅ W&B is configured")
    except:
        print("⚠️  W&B not configured. Run: wandb login")
        print("   Get your API key from: https://wandb.ai/authorize")
        return False
    
    # Test creating a minimal run
    try:
        # Initialize a test run
        run = wandb.init(
            project="fast-ddpm-3d-test",
            name="test-run",
            mode="offline",  # Use offline mode for testing
            config={
                "test": True,
                "volume_size": [64, 64, 64],
                "batch_size": 1
            }
        )
        
        # Log some dummy metrics
        for i in range(10):
            wandb.log({
                "loss": np.random.random() * 0.5 + 0.5,
                "learning_rate": 0.001 * (0.9 ** i),
                "step": i
            })
        
        # Log a summary metric
        wandb.run.summary["best_loss"] = 0.42
        
        # Finish the run
        wandb.finish()
        
        print("✅ W&B test run completed successfully!")
        print("   Check your W&B dashboard or look in ./wandb/ for offline runs")
        return True
        
    except Exception as e:
        print(f"❌ W&B test failed: {e}")
        return False

def quick_training_command():
    """Print quick commands to test W&B with actual training"""
    print("\n" + "="*60)
    print("Quick W&B Training Test Commands:")
    print("="*60)
    
    print("\n1. First, login to W&B (if not already done):")
    print("   wandb login")
    
    print("\n2. Test with minimal training (debug mode + W&B):")
    print("   python scripts/train_3d.py \\")
    print("     --data_root /path/to/your/brats/data \\")
    print("     --use_wandb \\")
    print("     --wandb_project fast-ddpm-test \\")
    print("     --debug \\")
    print("     --doc test-wandb-run")
    
    print("\n3. For full training with W&B:")
    print("   python scripts/train_3d.py \\")
    print("     --data_root /path/to/your/brats/data \\")
    print("     --use_wandb \\")
    print("     --wandb_project fast-ddpm-3d-brats \\")
    print("     --wandb_entity your-username \\")
    print("     --doc experiment-name")
    
    print("\n4. To disable W&B, simply omit the --use_wandb flag")
    
    print("\n" + "="*60)
    print("W&B Features Implemented:")
    print("="*60)
    print("✅ Training loss logging (every 10 steps)")
    print("✅ Validation loss logging")
    print("✅ Learning rate tracking")
    print("✅ GPU memory usage")
    print("✅ Gradient norm tracking")
    print("✅ Best model tracking")
    print("✅ Configuration logging")
    print("✅ Automatic run naming")
    
    print("\n" + "="*60)
    print("W&B Dashboard:")
    print("="*60)
    print("After running with --use_wandb, you can:")
    print("1. View runs at: https://wandb.ai/your-entity/your-project")
    print("2. Compare multiple runs")
    print("3. View loss curves in real-time")
    print("4. Track system metrics")
    print("5. Share results with collaborators")

if __name__ == "__main__":
    print("🧪 W&B Integration Test for Fast-DDPM-3D-BraTS")
    print("="*60)
    
    # Test W&B
    success = test_wandb_minimal()
    
    # Show commands
    quick_training_command()
    
    if not success:
        print("\n⚠️  Fix the issues above before using W&B with training")
    else:
        print("\n✅ W&B is ready to use with your training!")