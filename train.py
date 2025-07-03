#!/usr/bin/env python3
"""
Debug-enhanced training script to identify gradient explosion root cause
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


def debug_timestep_sampling(t_intervals, batch_size, method="antithetic"):
    """
    Debug function to analyze timestep sampling behavior
    """
    print(f"\n🔍 DEBUG: Timestep Sampling Analysis")
    print(f"Method: {method}")
    print(f"t_intervals shape: {t_intervals.shape}")
    print(f"t_intervals range: [{t_intervals.min()}, {t_intervals.max()}]")
    print(f"t_intervals values: {t_intervals.cpu().numpy()}")
    
    if method == "antithetic":
        # Your current method
        n = batch_size
        idx_1 = torch.randint(0, len(t_intervals), size=(n // 2 + 1,))
        idx_2 = len(t_intervals) - idx_1 - 1
        idx = torch.cat([idx_1, idx_2], dim=0)[:n]
        t = t_intervals[idx]
        
        print(f"Antithetic sampling:")
        print(f"  idx_1: {idx_1.cpu().numpy()}")
        print(f"  idx_2: {idx_2.cpu().numpy()}")
        print(f"  final idx: {idx.cpu().numpy()}")
        print(f"  final t values: {t.cpu().numpy()}")
        print(f"  t range: [{t.min()}, {t.max()}]")
        
    elif method == "simple":
        # Simple random sampling
        idx = torch.randint(0, len(t_intervals), size=(batch_size,))
        t = t_intervals[idx]
        
        print(f"Simple sampling:")
        print(f"  idx: {idx.cpu().numpy()}")
        print(f"  t values: {t.cpu().numpy()}")
        print(f"  t range: [{t.min()}, {t.max()}]")
    
    return t


def debug_loss_components_detailed(model, inputs, targets, t, e, betas, target_idx):
    """
    Detailed debugging of loss computation to match the debug script
    """
    print(f"\n🔍 DEBUG: Loss Computation Details")
    print(f"Inputs shape: {inputs.shape}, range: [{inputs.min():.3f}, {inputs.max():.3f}]")
    print(f"Targets shape: {targets.shape}, range: [{targets.min():.3f}, {targets.max():.3f}]")
    print(f"Timesteps: {t.cpu().numpy()}")
    print(f"Noise range: [{e.min():.3f}, {e.max():.3f}]")
    print(f"Target idx: {target_idx}")
    
    # Step-by-step loss computation (mirroring the actual loss function)
    a = (1-betas).cumprod(dim=0)
    print(f"Alpha cumprod range: [{a.min():.6f}, {a.max():.6f}]")
    
    a = torch.clamp(a, min=1e-8, max=1.0)
    a = a.index_select(0, t).view(-1, 1, 1, 1, 1)
    print(f"Selected alpha: {a.squeeze().cpu().numpy()}")
    
    # Check noise scaling (this is critical)
    e_scaled = e * 0.01  # Current scaling in loss function
    print(f"Scaled noise range: [{e_scaled.min():.6f}, {e_scaled.max():.6f}]")
    
    # Forward diffusion
    x_noisy = targets * a.sqrt() + e_scaled * (1.0 - a).sqrt()
    print(f"Noisy input range: [{x_noisy.min():.3f}, {x_noisy.max():.3f}]")
    
    # Model input preparation
    model_input = inputs.clone()
    model_input[:, target_idx:target_idx+1] = x_noisy
    print(f"Model input range: [{model_input.min():.3f}, {model_input.max():.3f}]")
    
    return model_input, e_scaled


def generate_and_log_samples(model, val_loader, betas, t_intervals, device, global_step, num_samples=4):
    """Generate samples and log them to W&B"""
    model.eval()
    
    with torch.no_grad():
        # Get a batch from validation set
        sample_batch = next(iter(val_loader))
        inputs = sample_batch['input'][:num_samples].to(device)
        targets = sample_batch['target'][:num_samples].unsqueeze(1).to(device)
        target_idx = sample_batch['target_idx'][0].item()
        
        # Generate samples using the reverse process
        try:
            # Create initial noise for the target modality
            x_target_noise = torch.randn_like(targets)  # [B, 1, H, W, D] 
            
            # Call with correct parameter order
            generated_sequence, x0_preds = unified_4to1_generalized_steps_3d(
                x=x_target_noise,
                x_available=inputs,
                target_idx=target_idx, 
                seq=t_intervals.cpu().numpy(),
                model=model,
                b=betas
            )
            
            # Get the final generated sample
            generated = generated_sequence[-1] if generated_sequence else None
            
            # Debug the generated output
            if generated is not None:
                logging.info(f"Generated samples shape: {generated.shape}")
                logging.info(f"Generated range: [{generated.min():.3f}, {generated.max():.3f}]")
            else:
                logging.warning("Generated samples is None")
            
            # Create comparison images (take middle slice for visualization)
            middle_slice = inputs.size(-1) // 2
            
            # Prepare images for logging
            images_to_log = []
            
            for i in range(min(num_samples, 4)):  # Log up to 4 samples
                # Original input (take first channel that's not the target)
                available_channels = [j for j in range(4) if j != target_idx]
                input_img = inputs[i, available_channels[0], :, :, middle_slice].cpu().numpy()
                
                # Ground truth target
                target_img = targets[i, 0, :, :, middle_slice].cpu().numpy()
                
                # Generated sample
                if generated is not None:
                    gen_img = generated[i, 0, :, :, middle_slice].cpu().numpy()
                else:
                    gen_img = np.zeros_like(target_img)
                
                # Normalize for display
                input_img = (input_img - input_img.min()) / (input_img.max() - input_img.min() + 1e-8)
                target_img = (target_img - target_img.min()) / (target_img.max() - target_img.min() + 1e-8)
                gen_img = (gen_img - gen_img.min()) / (gen_img.max() - gen_img.min() + 1e-8)
                
                # Create side-by-side comparison
                comparison = np.concatenate([input_img, target_img, gen_img], axis=1)
                
                images_to_log.append(wandb.Image(
                    comparison,
                    caption=f"Sample {i+1}: Input | Target | Generated (Target: {sample_batch['target_modality'][i]})"
                ))
            
            # Log to W&B
            wandb.log({
                "samples/generated_images": images_to_log,
                "samples/step": global_step
            }, step=global_step)
            
            logging.info(f"Generated and logged {len(images_to_log)} samples to W&B")
            
        except Exception as e:
            logging.warning(f"Failed to generate samples: {str(e)}")
            import traceback
            logging.warning(f"Traceback: {traceback.format_exc()}")
    
    model.train()


def training_loop_debug(model, train_loader, val_loader, optimizer, scheduler, scaler, 
                       betas, t_intervals, config, args, device):
    """🔥 DEBUG: Enhanced training loop with debugging steps"""
    
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
    
    logging.info("Starting DEBUG training...")
    logging.info(f"Scheduler type: {args.scheduler_type}, Timesteps: {args.timesteps}")
    logging.info(f"Batch size: {config.training.batch_size}")
    logging.info(f"Use mixed precision: {not args.no_mixed_precision}")
    logging.info(f"Timestep sampling method: {args.timestep_method}")
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
                
                # 🔍 DEBUG STEP 1: First batch detailed analysis
                if epoch == 0 and batch_idx == 0:
                    print(f"\n=== FIRST BATCH DETAILED DEBUG ===")
                    print(f"Input tensor shape: {inputs.shape}")
                    print(f"Target tensor shape: {targets.shape}")
                    print(f"Input range: [{inputs.min():.3f}, {inputs.max():.3f}]")
                    print(f"Target range: [{targets.min():.3f}, {targets.max():.3f}]")
                    print(f"Target modality: {batch['target_modality'][0]}")
                    print(f"Available modalities: {batch['available_modalities'][0]}")
                    
                    print(f"\n=== TRAINING BATCH DEBUG ===")
                    print(f"Batch case name: {batch['case_name'][0]}")
                    print(f"Successful modalities should be: ['t1n', 't1c', 't2w', 't2f']")
                    print(f"But available_modalities shows: {batch['available_modalities'][0]}")
                    print(f"This suggests only {len(batch['available_modalities'][0])} modalities loaded for training")
                    print(f"Target modality: {batch['target_modality'][0]}")
                    print(f"Target idx: {batch['target_idx'][0].item()}")
                    
                    print(f"\nInput tensor analysis:")
                    for i in range(4):
                        channel_data = inputs[0, i]
                        non_zero_count = torch.sum(channel_data > 0).item()
                        total_voxels = channel_data.numel()
                        print(f"  Channel {i} ({['t1n', 't1c', 't2w', 't2f'][i]}): {non_zero_count}/{total_voxels} non-zero voxels")
                        if non_zero_count == 0:
                            print(f"    -> Channel {i} is all zeros (target or missing)")
                        else:
                            print(f"    -> Channel {i} has data: range [{channel_data.min():.3f}, {channel_data.max():.3f}]")
                
                # Enhanced input validation
                if torch.isnan(inputs).any() or torch.isnan(targets).any():
                    logging.warning(f"Skipping batch {batch_idx} due to NaN values in input")
                    continue
                
                if torch.isinf(inputs).any() or torch.isinf(targets).any():
                    logging.warning(f"Skipping batch {batch_idx} due to Inf values in input")
                    continue
                
                n = inputs.size(0)
                
                # 🔍 DEBUG STEP 2: Timestep sampling analysis
                if epoch == 0 and batch_idx <= 2:  # Debug first few batches
                    t = debug_timestep_sampling(t_intervals, n, method=args.timestep_method)
                    t = t.to(device)
                else:
                    # Use specified timestep sampling method
                    if args.timestep_method == "simple":
                        # Simple random sampling (like debug script)
                        idx = torch.randint(0, len(t_intervals), size=(n,))
                        t = t_intervals[idx].to(device)
                    else:
                        # Fast-DDPM antithetic sampling (original method)
                        idx_1 = torch.randint(0, len(t_intervals), size=(n // 2 + 1,))
                        idx_2 = len(t_intervals) - idx_1 - 1
                        idx = torch.cat([idx_1, idx_2], dim=0)[:n]
                        t = t_intervals[idx].to(device)
                
                e = torch.randn_like(targets)
                
                # 🔍 DEBUG STEP 3: Detailed loss computation analysis
                if epoch == 0 and batch_idx == 0:
                    model_input_debug, e_scaled_debug = debug_loss_components_detailed(
                        model, inputs, targets, t, e, betas, target_idx
                    )
                
                # 🔍 DEBUG STEP 4: Test with and without mixed precision
                if args.no_mixed_precision:
                    # No mixed precision (like debug script)
                    loss = brats_4to1_loss(model, inputs, targets, t, e, b=betas, target_idx=target_idx)
                else:
                    # With mixed precision (original)
                    with autocast():
                        loss = brats_4to1_loss(model, inputs, targets, t, e, b=betas, target_idx=target_idx)
                
                # Enhanced loss validation
                if torch.isnan(loss) or torch.isinf(loss) or loss.item() < 0:
                    logging.warning(f"Skipping batch {batch_idx} due to invalid loss: {loss.item()}")
                    continue
                
                # 🔍 DEBUG STEP 5: Log loss value for first few batches
                if epoch == 0 and batch_idx <= 5:
                    print(f"Batch {batch_idx}: Loss = {loss.item():.6f}")
                
                # Backward pass
                if args.no_mixed_precision:
                    loss.backward()
                else:
                    scaler.scale(loss).backward()
                
                # Check gradients before clipping
                total_norm = 0
                max_grad = 0
                for p in model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                        max_grad = max(max_grad, p.grad.abs().max().item())
                total_norm = total_norm ** (1. / 2)
                
                # 🔍 DEBUG STEP 6: Detailed gradient analysis for first few batches
                if epoch == 0 and batch_idx <= 5:
                    print(f"Batch {batch_idx}: Gradient norm = {total_norm:.3f}, Max grad = {max_grad:.6f}")
                    
                    if total_norm > 10:  # Lower threshold for debugging
                        print(f"🚨 Large gradients detected in batch {batch_idx}!")
                        # Print which layers have large gradients
                        for name, param in model.named_parameters():
                            if param.grad is not None:
                                grad_norm = param.grad.norm().item()
                                if grad_norm > 1.0:
                                    print(f"  {name}: {grad_norm:.3f}")
                
                # Gradient explosion check with configurable threshold
                gradient_threshold = getattr(args, 'gradient_threshold', 50)
                if total_norm > gradient_threshold:
                    logging.warning(f"Large gradient norm detected: {total_norm:.3f}, skipping batch")
                    optimizer.zero_grad()
                    continue
                
                # Gradient clipping
                clip_norm = getattr(args, 'clip_norm', 1.0)
                if args.no_mixed_precision:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
                else:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
                
                # Optimizer step
                if args.no_mixed_precision:
                    optimizer.step()
                else:
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
                        'train/max_grad': max_grad,
                        'train/timesteps_mean': t.float().mean().item(),
                        'train/timesteps_max': t.max().item(),
                    }, step=global_step)
                    
                    # Generate and log samples
                    if global_step % args.sample_every == 0 and global_step > 0:
                        logging.info(f"Generating samples at step {global_step}")
                        generate_and_log_samples(
                            model, val_loader, betas, t_intervals, device, global_step
                        )
                
                # Periodic logging
                if global_step % args.log_every_n_steps == 0:
                    logging.info(f'Epoch {epoch+1}/{config.training.epochs}, '
                               f'Step {global_step} - '
                               f'Loss: {loss.item():.6f}, '
                               f'Target: {target_idx}, '
                               f'LR: {optimizer.param_groups[0]["lr"]:.2e}, '
                               f'Grad: {total_norm:.3f}')
                
                # Early stopping for debugging (process only first few batches)
                if args.debug_early_stop and batch_idx >= args.debug_early_stop:
                    logging.info(f"Early stopping for debugging after {batch_idx+1} batches")
                    break
                    
            except Exception as e:
                logging.warning(f"Training batch {batch_idx} failed: {str(e)}")
                import traceback
                logging.warning(f"Traceback: {traceback.format_exc()}")
                optimizer.zero_grad()
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
        
        # Generate and log samples periodically
        if use_wandb and global_step % args.sample_every == 0:
            generate_and_log_samples(model, val_loader, betas, t_intervals, device, global_step, num_samples=4)
        
        # Early exit for debugging
        if args.debug_early_stop:
            logging.info("Early exit for debugging after 1 epoch")
            break
    
    logging.info("Training completed!")
    logging.info(f"Best validation loss: {best_val_loss:.6f}")
    
    if use_wandb:
        wandb.finish()


def parse_args():
    """Parse command line arguments with debug options"""
    parser = argparse.ArgumentParser(description='3D Fast-DDPM Training for BraTS with Debug')
    parser.add_argument('--config', type=str, default='configs/fast_ddpm_3d.yml', help='Path to config file')
    parser.add_argument('--data_root', type=str, required=True, help='Path to BraTS data')
    parser.add_argument('--exp', type=str, default='./experiments', help='Experiment directory')
    parser.add_argument('--doc', type=str, default='fast_ddpm_3d_brats_debug', help='Experiment name')
    parser.add_argument('--timesteps', type=int, default=10, help='Number of timesteps (Fast-DDPM advantage)')
    parser.add_argument('--scheduler_type', type=str, default='uniform', choices=['uniform', 'non-uniform'], help='Timestep scheduler')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID')
    parser.add_argument('--resume', action='store_true', help='Resume training')
    parser.add_argument('--resume_path', type=str, help='Path to checkpoint to resume from')
    parser.add_argument('--debug', action='store_true', help='Debug mode with smaller dataset')
    parser.add_argument('--log_every_n_steps', type=int, default=1, help='Log training progress every N steps')
    parser.add_argument('--use_wandb', action='store_true', help='Use Weights & Biases for logging')
    parser.add_argument('--wandb_project', type=str, default='fast-ddpm-3d-brats-debug', help='W&B project name')
    parser.add_argument('--wandb_entity', type=str, default=None, help='W&B entity (username or team)')
    parser.add_argument('--sample_every', type=int, default=2000, help='Generate and log sample images to W&B every N steps')
    
    # 🔍 DEBUG-specific arguments
    parser.add_argument('--no_mixed_precision', action='store_true', help='Disable mixed precision training (like debug script)')
    parser.add_argument('--timestep_method', type=str, default='antithetic', choices=['antithetic', 'simple'], 
                       help='Timestep sampling method: antithetic (original) or simple (like debug)')
    parser.add_argument('--gradient_threshold', type=float, default=50.0, help='Gradient explosion threshold')
    parser.add_argument('--clip_norm', type=float, default=1.0, help='Gradient clipping norm')
    parser.add_argument('--debug_early_stop', type=int, default=None, help='Stop after N batches for debugging')

    return parser.parse_args()


def setup_datasets(args, config):
    """Setup training and validation datasets"""
    logging.info("Setting up datasets...")
    
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
        debug_train_size = min(10, len(train_dataset))  # Very small for debugging
        debug_val_size = min(5, len(val_dataset))
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
        num_workers=getattr(config.data, 'num_workers', 2),  # Reduced for debugging
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
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
    betas = get_beta_schedule(
        beta_schedule=config.diffusion.beta_schedule,
        beta_start=config.diffusion.beta_start,
        beta_end=config.diffusion.beta_end,
        num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
    )
    betas = torch.from_numpy(betas).float().to(device)
    num_timesteps = betas.shape[0]
    
    t_intervals = get_timestep_schedule(args.scheduler_type, args.timesteps, num_timesteps)
    
    # Initialize gradient scaler (only if using mixed precision)
    scaler = GradScaler() if not args.no_mixed_precision else None
    
    return betas, t_intervals, scaler


def setup_logging(log_dir):
    """Setup logging configuration"""
    os.makedirs(log_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'training_debug.log')),
            logging.StreamHandler()
        ]
    )


def main():
    """Main training function with debug enhancements"""
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
    
    # 🔍 DEBUG: Print diffusion setup
    print(f"\n🔍 DIFFUSION SETUP DEBUG:")
    print(f"Beta schedule: {config.diffusion.beta_schedule}")
    print(f"Beta range: [{config.diffusion.beta_start}, {config.diffusion.beta_end}]")
    print(f"Diffusion timesteps: {config.diffusion.num_diffusion_timesteps}")
    print(f"Fast-DDPM timesteps: {args.timesteps}")
    print(f"t_intervals: {t_intervals.cpu().numpy()}")
    
    # Use debug training loop
    training_loop_debug(
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