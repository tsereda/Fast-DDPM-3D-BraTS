#!/usr/bin/env python3
"""
Test script for enhanced W&B visualization
"""

import torch
import numpy as np
import sys
import os
import yaml
import argparse
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append('.')

def create_dummy_batch():
    """Create a dummy batch for testing visualization"""
    batch_size = 1
    volume_size = (64, 64, 64)
    
    # Create realistic-looking dummy data
    inputs = torch.randn(batch_size, 4, *volume_size)
    targets = torch.randn(batch_size, *volume_size)
    target_idx = torch.tensor([2])  # T2w modality
    
    # Normalize to [-1, 1] range
    inputs = torch.tanh(inputs)
    targets = torch.tanh(targets)
    
    # Create some realistic structure (brain-like)
    center = [s // 2 for s in volume_size]
    for i in range(4):  # For each input modality
        for z in range(volume_size[2]):
            for y in range(volume_size[1]):
                for x in range(volume_size[0]):
                    # Distance from center
                    dist = ((x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2)**0.5
                    # Create brain-like structure
                    if dist < min(volume_size) * 0.3:  # Inner brain
                        inputs[0, i, x, y, z] = 0.8 + 0.2 * torch.randn(1)
                    elif dist < min(volume_size) * 0.4:  # Outer brain
                        inputs[0, i, x, y, z] = 0.4 + 0.2 * torch.randn(1)
                    else:  # Background
                        inputs[0, i, x, y, z] = -0.8 + 0.1 * torch.randn(1)
    
    # Make target similar to one of the inputs but slightly different
    targets = inputs[0, 2] + 0.1 * torch.randn_like(targets)
    
    batch = {
        'input': inputs,
        'target': targets,
        'target_idx': target_idx
    }
    
    return batch

def create_dummy_model():
    """Create a dummy model for testing"""
    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv3d(4, 1, 3, padding=1)
            
        def forward(self, x, t):
            # Return some noise-like output
            return self.conv(x) + 0.1 * torch.randn(x.shape[0], 1, *x.shape[2:])
    
    return DummyModel()

def test_wandb_offline():
    """Test W&B visualization in offline mode"""
    print("🧪 Testing W&B Visualization Enhancement")
    print("="*60)
    
    # Set up offline W&B
    os.environ['WANDB_MODE'] = 'offline'
    
    try:
        import wandb
        print("✅ W&B imported successfully")
    except ImportError:
        print("❌ W&B not available. Install with: pip install wandb")
        return False
    
    # Initialize W&B in offline mode
    wandb.init(
        project="test-visualization", 
        mode="offline",
        config={"test": True}
    )
    
    # Import the enhanced logging functions
    from scripts.train_3d import log_samples_to_wandb, log_multi_slice_comparison_to_wandb
    
    # Create test data
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = create_dummy_model().to(device)
    batch = create_dummy_batch()
    
    # Move batch to device
    for key in batch:
        if isinstance(batch[key], torch.Tensor):
            batch[key] = batch[key].to(device)
    
    # Create dummy diffusion parameters
    t_intervals = torch.arange(0, 1000, 100)  # [0, 100, 200, ..., 900]
    betas = torch.linspace(0.0001, 0.02, 1000).to(device)
    
    print("📊 Testing comprehensive comparison visualization...")
    try:
        log_samples_to_wandb(model, batch, t_intervals, betas, device, step=1000)
        print("✅ Comprehensive comparison logged successfully")
    except Exception as e:
        print(f"❌ Comprehensive comparison failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("📊 Testing multi-slice comparison visualization...")
    try:
        log_multi_slice_comparison_to_wandb(model, batch, t_intervals, betas, device, step=1000)
        print("✅ Multi-slice comparison logged successfully")
    except Exception as e:
        print(f"❌ Multi-slice comparison failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test that files are created in wandb/offline-run-*
    wandb_dir = "wandb"
    if os.path.exists(wandb_dir):
        run_dirs = [d for d in os.listdir(wandb_dir) if d.startswith('offline-run-')]
        if run_dirs:
            print(f"✅ W&B offline logs created in: {run_dirs[-1]}")
        else:
            print("⚠️  No offline run directories found")
    
    wandb.finish()
    print("🎉 All W&B visualization tests passed!")
    return True

def test_matplotlib_only():
    """Test just the matplotlib visualization without W&B"""
    print("\n🖼️  Testing matplotlib visualization components...")
    
    try:
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
        import matplotlib.patches as patches
        
        # Create test data
        batch = create_dummy_batch()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Move to device
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device)
        
        inputs = batch['input']
        targets = batch['target'].unsqueeze(1)
        target_idx = batch['target_idx'][0].item()
        
        # Get middle slice
        slice_idx = inputs.shape[-1] // 2
        modality_names = ['T1n', 'T1c', 'T2w', 'T2f']
        
        # Prepare data
        input_slices = []
        for i in range(4):
            slice_data = (inputs[0, i, :, :, slice_idx].cpu().numpy() + 1) / 2
            input_slices.append(slice_data)
        
        generated_slice = (targets[0, 0, :, :, slice_idx].cpu().numpy() + 1) / 2  # Using target as dummy generated
        target_slice = (targets[0, 0, :, :, slice_idx].cpu().numpy() + 1) / 2
        
        # Create the figure
        fig = plt.figure(figsize=(20, 12))
        gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.2)
        
        fig.suptitle(f'BraTS Modality Synthesis Test - Target: {modality_names[target_idx]}', 
                    fontsize=16, fontweight='bold')
        
        # Top row: Input modalities
        for i in range(4):
            ax = fig.add_subplot(gs[0, i])
            im = ax.imshow(input_slices[i], cmap='gray', vmin=0, vmax=1)
            
            if i == target_idx:
                rect = patches.Rectangle((0, 0), input_slices[i].shape[1]-1, input_slices[i].shape[0]-1, 
                                       linewidth=4, edgecolor='red', facecolor='none')
                ax.add_patch(rect)
                ax.set_title(f'{modality_names[i]}\n(TARGET)', fontsize=12, fontweight='bold', color='red')
            else:
                ax.set_title(f'{modality_names[i]}\n(Input)', fontsize=12, fontweight='bold')
            
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # Test save
        plt.savefig('test_visualization.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print("✅ Matplotlib visualization test successful")
        print("✅ Test image saved as 'test_visualization.png'")
        return True
        
    except Exception as e:
        print(f"❌ Matplotlib test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("🔍 Testing Enhanced W&B Visualization for Fast-DDPM-3D-BraTS")
    print("=" * 80)
    
    # Test matplotlib components first
    matplotlib_success = test_matplotlib_only()
    
    # Test W&B integration
    wandb_success = test_wandb_offline()
    
    print("\n" + "=" * 80)
    print("📊 TEST SUMMARY")
    print("=" * 80)
    
    if matplotlib_success:
        print("✅ Matplotlib visualization components work correctly")
    else:
        print("❌ Matplotlib visualization has issues")
    
    if wandb_success:
        print("✅ W&B integration works correctly")
        print("✅ Enhanced comparison views are ready for training")
    else:
        print("❌ W&B integration has issues")
    
    if matplotlib_success and wandb_success:
        print("\n🎉 ALL TESTS PASSED!")
        print("🚀 Ready to use enhanced W&B visualization in training")
        print("\nFeatures available:")
        print("  • Comprehensive comparison view with all 4 input modalities")
        print("  • Side-by-side generated vs target comparison")
        print("  • Difference maps with quantitative metrics")
        print("  • Multi-slice 3D visualization")
        print("  • Automatic PSNR and SSIM calculation")
        print("  • Clear modality labeling and highlighting")
    else:
        print("\n❌ Some tests failed. Check the errors above.")
    
    return matplotlib_success and wandb_success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
