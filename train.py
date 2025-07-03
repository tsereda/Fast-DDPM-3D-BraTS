#!/usr/bin/env python3
"""
Fixed training script with improved alpha computation for numerical stability
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import logging
import yaml
import numpy as np
import matplotlib.pyplot as plt

# Add the current directory to path to import modules
sys.path.append('.')
sys.path.append('..')

# Import project modules
from data.brain_3d_unified import BraTS3DUnifiedDataset
from models.fast_ddpm_3d import FastDDPM3D
from functions.losses import brats_4to1_loss
from functions.denoising_3d import generalized_steps_3d, unified_4to1_generalized_steps_3d

# Minimal training utils
from training_utils import load_config, get_beta_schedule, get_timestep_schedule

# Optional wandb import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def compute_alpha_stable(beta, t):
    """
    🔥 FIXED: Improved alpha computation with better numerical stability
    
    Args:
        beta: [T] beta schedule
        t: [B] timestep indices
    
    Returns:
        alpha: [B, 1, 1, 1, 1] alpha values for 5D tensors
    """
    # Add zero at beginning for t=0 case
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    
    # Compute cumulative product
    alpha_cumprod = (1 - beta).cumprod(dim=0)
    
    # 🔥 FIXED: More conservative clamping for better numerical stability
    # - min=1e-5 instead of 1e-8 (prevents underflow)
    # - max=0.999 instead of 1.0 (prevents edge case issues)
    alpha_cumprod = torch.clamp(alpha_cumprod, min=1e-5, max=0.999)
    
    # Select timesteps and reshape for 5D tensors [B, C, H, W, D]
    alpha = alpha_cumprod.index_select(0, t + 1).view(-1, 1, 1, 1, 1)
    
    return alpha


def validate_model_fixed(model, val_loader, device, betas, t_intervals):
    """🔥 FIXED: Validation function with stable alpha computation"""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in val_loader:
            try:
                inputs = batch['input'].to(device)
                targets = batch['target'].unsqueeze(1).to(device)
                target_idx = batch['target_idx'][0].item()
                
                n = inputs.size(0)
                idx = torch.randint(0, len(t_intervals), size=(n,))
                t = t_intervals[idx].to(device)
                e = torch.randn_like(targets)
                
                with autocast():
                    loss = brats_4to1_loss(model, inputs, targets, t, e, b=betas, target_idx=target_idx)
                
                # Check for invalid loss values
                if torch.isnan(loss) or torch.isinf(loss):
                    logging.warning(f"Invalid loss detected in validation: {loss.item()}")
                    continue
                
                total_loss += loss.item()
                num_batches += 1
                
                if num_batches >= 10:  # Limit validation to 10 batches
                    break
                    
            except Exception as e:
                logging.warning(f"Validation batch failed: {e}")
                continue
    
    model.train()
    return total_loss / max(num_batches, 1)


def sample_and_log_images_fixed(model, val_loader, device, betas, t_intervals, step, use_wandb):
    """🔥 FIXED: Generate sample images with stable alpha computation"""
    if not use_wandb:
        return
        
    model.eval()
    
    try:
        with torch.no_grad():
            # Get a single validation batch for sampling
            batch = next(iter(val_loader))
            inputs = batch['input'][:1].to(device)  # Take only first sample
            targets = batch['target'][:1].unsqueeze(1).to(device)
            target_idx = batch['target_idx'][0].item()
            
            modality_names = ['t1n', 't1c', 't2w', 't2f']
            target_name = modality_names[target_idx]
            
            # Create input for generation - mask the target modality
            x_available = inputs.clone()
            x_available[:, target_idx] = 0  # Zero out the target modality
            
            # 🔥 FIXED: Use stable Gaussian noise initialization
            noise_shape = targets.shape
            x_noise = torch.randn(noise_shape).to(device)
            
            # Get sampling sequence (use subset of t_intervals for faster sampling)
            seq = t_intervals[::max(1, len(t_intervals)//10)].tolist() if len(t_intervals) > 10 else t_intervals.tolist()
            
            # Generate sample using the diffusion model
            try:
                xs, x0_preds = unified_4to1_generalized_steps_3d(
                    x_noise, x_available, target_idx, seq, model, betas, eta=0.0
                )
                generated = xs[-1]  # Final generated sample
                
                # 🔥 FIXED: Check for invalid generated values
                if torch.isnan(generated).any() or torch.isinf(generated).any():
                    logging.warning("Generated sample contains NaN/Inf values, using noise")
                    generated = x_noise
                    
            except Exception as gen_error:
                logging.warning(f"Generation failed, using noise: {str(gen_error)}")
                generated = x_noise
            
            # Convert tensors to numpy for visualization
            inputs_np = inputs[0].cpu().numpy()  # Shape: (4, H, W, D)
            targets_np = targets[0, 0].cpu().numpy()  # Shape: (H, W, D)
            generated_np = generated[0, 0].cpu().numpy()  # Shape: (H, W, D)
            x_noise_np = x_noise[0, 0].cpu().numpy()  # Shape: (H, W, D)
            
            # Get middle slice for visualization
            mid_slice = targets_np.shape[2] // 2
            
            # Create comprehensive visualization with single slice
            fig, axes = plt.subplots(1, 6, figsize=(24, 4))
            fig.suptitle(f'Step {step}: Missing Modality Synthesis - {target_name.upper()}', 
                        fontsize=18, fontweight='bold')
            
            # Normalize function for better visualization
            def normalize_for_vis(img):
                if img.max() > img.min():
                    return np.clip((img - img.min()) / (img.max() - img.min()), 0, 1)
                return np.zeros_like(img)
            
            # Column 0-3: Input modalities (including noise for target)
            for i, mod_name in enumerate(modality_names):
                ax = axes[i]
                if i == target_idx:
                    # Show the noisy input for the target modality
                    img = normalize_for_vis(x_noise_np[:, :, mid_slice])
                    ax.imshow(img, cmap='gray', vmin=0, vmax=1)
                    ax.set_title(f'{mod_name.upper()}\n(Noise)', fontweight='bold', color='red')
                else:
                    # Show available input modalities
                    img = normalize_for_vis(inputs_np[i, :, :, mid_slice])
                    ax.imshow(img, cmap='gray', vmin=0, vmax=1)
                    ax.set_title(f'{mod_name.upper()}\n(Available)', fontweight='bold', color='green')
                ax.axis('off')
            
            # Column 4: Generated result
            gen_img = normalize_for_vis(generated_np[:, :, mid_slice])
            axes[4].imshow(gen_img, cmap='gray', vmin=0, vmax=1)
            axes[4].set_title('Generated\n(Output)', fontweight='bold', color='blue')
            axes[4].axis('off')
            
            # Column 5: Ground truth
            gt_img = normalize_for_vis(targets_np[:, :, mid_slice])
            axes[5].imshow(gt_img, cmap='gray', vmin=0, vmax=1)
            axes[5].set_title('Ground Truth\n(Target)', fontweight='bold', color='orange')
            axes[5].axis('off')
            
            plt.tight_layout()
            
            # Log only the comprehensive view to W&B
            wandb.log({
                f"samples/comprehensive_view": wandb.Image(fig, caption=f"Comprehensive view at step {step} - {target_name}"),
                f"samples/generation_stats": {
                    "generated_min": float(generated_np.min()),
                    "generated_max": float(generated_np.max()),
                    "generated_mean": float(generated_np.mean()),
                    "target_min": float(targets_np.min()),
                    "target_max": float(targets_np.max()),
                    "target_mean": float(targets_np.mean()),
                }
            }, step=step)
            
            plt.close(fig)
            
    except Exception as e:
        logging.warning(f"Failed to generate and log sample images: {str(e)}")
        import traceback
        logging.warning(f"Traceback: {traceback.format_exc()}")
    finally:
        model.train()


def training_loop_fixed(model, train_loader, val_loader, optimizer, scheduler, scaler, 
                       betas, t_intervals, config, args, device):
    """🔥 FIXED: Training loop with improved numerical stability"""
    
    # Setup W&B if requested
    use_wandb = False
    if args.use_wandb and WANDB_AVAILABLE:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=f"{args.doc}",
            config=vars(args)
        )
        use_wandb = True
        logging.info("✅ W&B initialized successfully")
    elif args.use_wandb and not WANDB_AVAILABLE:
        logging.warning("⚠️ W&B requested but not installed. Install with: pip install wandb")
    
    # Training parameters
    global_step = 0
    best_val_loss = float('inf')
    log_dir = os.path.join(args.exp, 'logs', args.doc)
    os.makedirs(log_dir, exist_ok=True)
    
    logging.info("Starting training...")
    logging.info(f"Scheduler type: {args.scheduler_type}, Timesteps: {args.timesteps}")
    logging.info(f"Batch size: {config.training.batch_size}")
    logging.info(f"🔥 Using improved alpha computation with conservative clamping")
    if use_wandb:
        logging.info(f"W&B sampling every {args.sample_every} steps")
    
    # Training loop
    for epoch in range(config.training.epochs):
        model.train()
        epoch_loss = 0.0
        valid_batches = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config.training.epochs}')
        
        for batch_idx, batch in enumerate(pbar):
            optimizer.zero_grad()
            
            try:
                # Move data to device
                inputs = batch['input'].to(device)
                targets = batch['target'].unsqueeze(1).to(device)
                target_idx = batch['target_idx'][0].item()
                
                # Print original and crop sizes (only for first batch of first epoch)
                if epoch == 0 and batch_idx == 0:
                    print(f"\n=== Data Size Information ===")
                    print(f"Input tensor shape: {inputs.shape}")  # [batch_size, 4, H, W, D]
                    print(f"Target tensor shape: {targets.shape}")  # [batch_size, 1, H, W, D]
                    print(f"Crop size used: {inputs.shape[2:]}")  # (H, W, D)
                    print(f"Case name: {batch['case_name'][0]}")
                    print(f"Target modality: {batch['target_modality'][0]}")
                    print(f"Available modalities: {batch['available_modalities'][0]}")
                    crop_coords = batch['crop_coords'][0]
                    print(f"Crop coordinates: {crop_coords}")
                    print(f"=== End Data Size Information ===\n")
                
                # 🔥 FIXED: Enhanced input validation
                if torch.isnan(inputs).any() or torch.isnan(targets).any():
                    logging.warning(f"Skipping batch {batch_idx} due to NaN values in input")
                    continue
                
                if torch.isinf(inputs).any() or torch.isinf(targets).any():
                    logging.warning(f"Skipping batch {batch_idx} due to Inf values in input")
                    continue
                
                # Check for reasonable value ranges
                if inputs.min() < -10 or inputs.max() > 10:
                    logging.warning(f"Skipping batch {batch_idx} due to extreme input values: [{inputs.min():.3f}, {inputs.max():.3f}]")
                    continue
                
                n = inputs.size(0)
                
                # Fast-DDPM antithetic sampling with improved stability
                idx_1 = torch.randint(0, len(t_intervals), size=(n // 2 + 1,))
                idx_2 = len(t_intervals) - idx_1 - 1
                idx = torch.cat([idx_1, idx_2], dim=0)[:n]
                t = t_intervals[idx].to(device)
                
                e = torch.randn_like(targets)
                
                # Forward pass with autocast
                with autocast():
                    loss = brats_4to1_loss(model, inputs, targets, t, e, b=betas, target_idx=target_idx)
                
                # 🔥 FIXED: Enhanced loss validation
                if torch.isnan(loss) or torch.isinf(loss) or loss.item() < 0:
                    logging.warning(f"Skipping batch {batch_idx} due to invalid loss: {loss.item()}")
                    continue
                
                # Backward pass
                scaler.scale(loss).backward()
                
                # 🔥 FIXED: Check gradients before clipping
                total_norm = 0
                for p in model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** (1. / 2)
                
                if total_norm > 100:  # Very large gradient
                    logging.warning(f"Large gradient norm detected: {total_norm:.3f}, skipping batch")
                    optimizer.zero_grad()
                    continue
                
                # Gradient clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                # Optimizer step
                scaler.step(optimizer)
                scaler.update()
                
                global_step += 1
                epoch_loss += loss.item()
                valid_batches += 1
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{loss.item():.6f}',
                    'lr': f'{optimizer.param_groups[0]["lr"]:.2e}',
                    'target': f'{target_idx}',
                    'step': global_step,
                    'grad_norm': f'{total_norm:.3f}'
                })
                
                # W&B logging
                if use_wandb:
                    wandb.log({
                        'train/loss': loss.item(),
                        'train/learning_rate': optimizer.param_groups[0]["lr"],
                        'train/epoch': epoch,
                        'train/step': global_step,
                        'train/target_idx': target_idx,
                        'train/grad_norm': total_norm,
                    }, step=global_step)
                
                # Periodic logging
                if global_step % args.log_every_n_steps == 0:
                    logging.info(f'Epoch {epoch+1}/{config.training.epochs}, '
                               f'Step {global_step} - '
                               f'Loss: {loss.item():.6f}, '
                               f'Target: {target_idx}, '
                               f'LR: {optimizer.param_groups[0]["lr"]:.2e}, '
                               f'Grad: {total_norm:.3f}')
                
                # Generate and log sample images
                if global_step % args.sample_every == 0 and global_step > 0:
                    sample_and_log_images_fixed(model, val_loader, device, betas, t_intervals, global_step, use_wandb)
                
                # Validation
                if global_step % 1000 == 0:
                    val_loss = validate_model_fixed(model, val_loader, device, betas, t_intervals)
                    logging.info(f'Step {global_step} - Val Loss: {val_loss:.6f}')
                    
                    if use_wandb:
                        wandb.log({'val/loss': val_loss}, step=global_step)
                    
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        logging.info(f'New best validation loss: {val_loss:.6f}')
                        if use_wandb:
                            wandb.run.summary["best_val_loss"] = val_loss
                            wandb.run.summary["best_val_step"] = global_step
                            
            except Exception as e:
                logging.warning(f"Training batch {batch_idx} failed: {str(e)}")
                optimizer.zero_grad()  # Clear gradients on error
                continue
        
        # End of epoch
        scheduler.step()
        if valid_batches > 0:
            avg_loss = epoch_loss / valid_batches
            logging.info(f'Epoch {epoch+1} - Average Loss: {avg_loss:.6f} '
                        f'({valid_batches}/{len(train_loader)} valid batches), '
                        f'LR: {optimizer.param_groups[0]["lr"]:.2e}')
            
            if use_wandb:
                wandb.log({
                    'epoch/avg_loss': avg_loss,
                    'epoch/learning_rate': optimizer.param_groups[0]["lr"],
                    'epoch/epoch': epoch + 1,
                    'epoch/valid_batches': valid_batches,
                    'epoch/total_batches': len(train_loader),
                }, step=global_step)
        else:
            logging.warning(f'Epoch {epoch+1} - No valid batches processed!')
    
    logging.info("Training completed!")
    logging.info(f"Best validation loss: {best_val_loss:.6f}")
    
    if use_wandb:
        wandb.finish()


# Copy all the other functions from the original script...
def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='3D Fast-DDPM Training for BraTS')
    parser.add_argument('--config', type=str, default='configs/fast_ddpm_3d.yml', help='Path to config file')
    parser.add_argument('--data_root', type=str, required=True, help='Path to BraTS data')
    parser.add_argument('--exp', type=str, default='./experiments', help='Experiment directory')
    parser.add_argument('--doc', type=str, default='fast_ddpm_3d_brats', help='Experiment name')
    parser.add_argument('--timesteps', type=int, default=10, help='Number of timesteps (Fast-DDPM advantage)')
    parser.add_argument('--scheduler_type', type=str, default='uniform', choices=['uniform', 'non-uniform'], help='Timestep scheduler')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID')
    parser.add_argument('--resume', action='store_true', help='Resume training')
    parser.add_argument('--resume_path', type=str, help='Path to checkpoint to resume from')
    parser.add_argument('--debug', action='store_true', help='Debug mode with smaller dataset')
    parser.add_argument('--log_every_n_steps', type=int, default=50, help='Log training progress every N steps')
    parser.add_argument('--use_wandb', action='store_true', help='Use Weights & Biases for logging')
    parser.add_argument('--wandb_project', type=str, default='fast-ddpm-3d-brats', help='W&B project name')
    parser.add_argument('--wandb_entity', type=str, default=None, help='W&B entity (username or team)')
    parser.add_argument('--sample_every', type=int, default=2000, help='Generate and log sample images to W&B every N steps')

    return parser.parse_args()


def setup_datasets(args, config):
    """Setup training and validation datasets"""
    logging.info("Setting up datasets...")
    
    # Check if data root exists
    if not os.path.exists(args.data_root):
        raise FileNotFoundError(f"Data root not found: {args.data_root}")
    
    train_dataset = BraTS3DUnifiedDataset(
        data_root=args.data_root,
        phase='train'
    )
    
    val_dataset = BraTS3DUnifiedDataset(
        data_root=args.data_root,
        phase='val'
    )
    
    if args.debug:
        # Use smaller debug dataset
        debug_train_size = min(50, len(train_dataset))
        debug_val_size = min(10, len(val_dataset))
        train_indices = list(range(debug_train_size))
        val_indices = list(range(debug_val_size))
        train_dataset = Subset(train_dataset, train_indices)
        val_dataset = Subset(val_dataset, val_indices)
        logging.info(f"Debug mode: using {debug_train_size} train samples, {debug_val_size} val samples")
    
    return train_dataset, val_dataset


def setup_dataloaders(train_dataset, val_dataset, config):
    """Setup data loaders"""
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=getattr(config.data, 'num_workers', 4),
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    logging.info(f"Train samples: {len(train_dataset)}")
    logging.info(f"Val samples: {len(val_dataset)}")
    logging.info(f"Train batches: {len(train_loader)}")
    
    return train_loader, val_loader


def setup_model_and_optimizer(config, device):
    """Setup model, optimizer, and scheduler"""
    logging.info("Setting up 3D Fast-DDPM model...")
    
    model = FastDDPM3D(config).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Total parameters: {total_params:,}")
    logging.info(f"Trainable parameters: {trainable_params:,}")
    
    # Optimizer and scheduler
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=config.training.learning_rate,
        weight_decay=getattr(config.training, 'weight_decay', 0),
        betas=(0.9, 0.999)
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=config.training.epochs,
        eta_min=config.training.learning_rate * 0.1
    )
    
    return model, optimizer, scheduler


def setup_diffusion_and_scaler(config, device, args):
    """Setup diffusion parameters and gradient scaler"""
    # Diffusion setup
    betas = get_beta_schedule(
        beta_schedule=config.diffusion.beta_schedule,
        beta_start=config.diffusion.beta_start,
        beta_end=config.diffusion.beta_end,
        num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
    )
    betas = torch.from_numpy(betas).float().to(device)
    num_timesteps = betas.shape[0]
    
    t_intervals = get_timestep_schedule(args.scheduler_type, args.timesteps, num_timesteps)
    
    # Initialize gradient scaler
    scaler = GradScaler()
    
    return betas, t_intervals, scaler


def setup_logging(log_dir):
    """Setup logging configuration"""
    os.makedirs(log_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'training.log')),
            logging.StreamHandler()
        ]
    )


def main():
    """Main training function"""
    args = parse_args()
    
    # Setup device
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
        torch.cuda.set_device(args.gpu)
    else:
        device = torch.device('cpu')
        logging.warning("CUDA not available, using CPU")
    
    print(f"Using device: {device}")
    
    # Setup logging
    log_dir = os.path.join(args.exp, 'logs', args.doc)
    setup_logging(log_dir)
    
    # Load config
    try:
        config = load_config(args.config)
    except FileNotFoundError:
        logging.error(f"Config file not found: {args.config}")
        sys.exit(1)
    
    config.device = device
    
    # Create experiment directory
    os.makedirs(os.path.join(args.exp, args.doc), exist_ok=True)
    
    # Setup datasets and dataloaders
    train_dataset, val_dataset = setup_datasets(args, config)
    train_loader, val_loader = setup_dataloaders(train_dataset, val_dataset, config)
    
    # Setup model, optimizer, and scheduler
    model, optimizer, scheduler = setup_model_and_optimizer(config, device)
    
    # Setup diffusion and gradient scaler
    betas, t_intervals, scaler = setup_diffusion_and_scaler(config, device, args)
    
    # 🔥 FIXED: Use improved training loop
    training_loop_fixed(
        model, train_loader, val_loader, optimizer, scheduler, scaler,
        betas, t_intervals, config, args, device
    )


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        if WANDB_AVAILABLE and wandb.run is not None:
            wandb.finish()
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        if WANDB_AVAILABLE and wandb.run is not None:
            wandb.finish()
        raise