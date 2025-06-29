#!/usr/bin/env python3
"""
Complete 3D Fast-DDPM training script for BraTS
"""

import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import logging
import numpy as np
from collections import OrderedDict

# Add the current directory to path to import modules
sys.path.append('.')
sys.path.append('..')

try:
    from models.diffusion_3d import Model3D
    from functions.losses import sg_noise_estimation_loss, combined_loss
    from data.brain_3d_unified import BraTS3DUnifiedDataset
    from utils.gpu_memory import get_recommended_volume_size, check_memory_usage
    from utils.data_validation import validate_brats_data_structure, print_validation_results
    print("✓ Successfully imported 3D components")
except ImportError as e:
    print(f"✗ Import error: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)

def parse_args():
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
    return parser.parse_args()

def dict2namespace(config):
    """Convert dictionary to namespace object"""
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace

def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    """Beta schedule for diffusion - same as Fast-DDPM"""
    if beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "cosine":
        # Cosine schedule as in improved DDPM
        steps = num_diffusion_timesteps + 1
        s = 0.008  # small offset
        x = np.linspace(0, num_diffusion_timesteps, steps)
        alphas_cumprod = np.cos(((x / num_diffusion_timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        betas = np.clip(betas, 0, 0.999)
    else:
        raise NotImplementedError(f"Beta schedule {beta_schedule} not implemented")
    
    assert betas.shape == (num_diffusion_timesteps,)
    return betas

def setup_logging(log_dir, args):
    """Setup logging configuration"""
    os.makedirs(log_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'train.log')),
            logging.StreamHandler()
        ]
    )
    
    # Log arguments
    logging.info("Training arguments:")
    for arg, value in vars(args).items():
        logging.info(f"  {arg}: {value}")

def get_timestep_schedule(scheduler_type, timesteps, num_timesteps):
    """Get timestep schedule for Fast-DDPM"""
    if scheduler_type == 'uniform':
        skip = num_timesteps // timesteps
        t_intervals = torch.arange(0, num_timesteps, skip)
    elif scheduler_type == 'non-uniform':
        # Non-uniform schedule from Fast-DDPM paper (optimized for 10 steps)
        if timesteps == 10:
            t_intervals = torch.tensor([0, 199, 399, 599, 699, 799, 849, 899, 949, 999])
        else:
            # Adaptive non-uniform schedule
            num_1 = int(timesteps * 0.4)
            num_2 = int(timesteps * 0.6)
            stage_1 = torch.linspace(0, 699, num_1 + 1)[:-1]
            stage_2 = torch.linspace(699, 999, num_2)
            stage_1 = torch.ceil(stage_1).long()
            stage_2 = torch.ceil(stage_2).long()
            t_intervals = torch.cat((stage_1, stage_2))
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
    
    return t_intervals

def save_checkpoint(model, optimizer, scaler, step, epoch, loss, config, path):
    """Save training checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'loss': loss,
        'config': config
    }
    torch.save(checkpoint, path)
    logging.info(f'Saved checkpoint at step {step}')

def load_checkpoint(path, model, optimizer, scaler, device):
    """Load training checkpoint"""
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if 'scaler_state_dict' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
    
    return checkpoint['epoch'], checkpoint['step'], checkpoint['loss']

def validate_model(model, val_loader, device, betas, timesteps):
    """Validation loop"""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in val_loader:
            inputs = batch['input'].to(device)
            targets = batch['target'].to(device).unsqueeze(1)
            
            n = inputs.size(0)
            t = torch.randint(0, len(betas), (n,), device=device)
            e = torch.randn_like(targets)
            
            loss = sg_noise_estimation_loss(model, inputs, targets, t, e, betas)
            total_loss += loss.item()
            num_batches += 1
    
    model.train()
    return total_loss / num_batches if num_batches > 0 else 0

def main():
    args = parse_args()
    
    # Setup device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Setup logging
    log_dir = os.path.join(args.exp, 'logs', args.doc)
    setup_logging(log_dir, args)
    
    # Load config
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found: {args.config}")
        
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    config = dict2namespace(config)
    config.device = device
    
    # Override config with args
    if hasattr(config, 'diffusion'):
        config.diffusion.num_diffusion_timesteps = getattr(config.diffusion, 'timesteps', 1000)
    
    # Create experiment directory
    exp_dir = os.path.join(args.exp, args.doc)
    os.makedirs(exp_dir, exist_ok=True)
    
    # Save config
    with open(os.path.join(exp_dir, 'config.yaml'), 'w') as f:
        yaml.dump(vars(args), f, default_flow_style=False)
    
    # Dataset and DataLoader
    logging.info("Setting up datasets...")
    
    # Validate data structure first
    validation_results = validate_brats_data_structure(args.data_root)
    print_validation_results(validation_results)
    
    if not validation_results['valid']:
        raise ValueError("Invalid data structure. Please check your BraTS data directory.")
    
    # Auto-adjust volume size based on GPU memory
    recommended_size = get_recommended_volume_size()
    if hasattr(config.data, 'volume_size'):
        original_size = tuple(config.data.volume_size)
        print(f"Config volume size: {original_size}")
        print(f"Recommended volume size: {recommended_size}")
        
        # Use the smaller of the two (more conservative)
        final_size = tuple(min(o, r) for o, r in zip(original_size, recommended_size))
        config.data.volume_size = final_size
        print(f"Using volume size: {final_size}")
    else:
        config.data.volume_size = recommended_size
    
    # Check if data root exists
    if not os.path.exists(args.data_root):
        raise FileNotFoundError(f"Data root not found: {args.data_root}")
    
    try:
        train_dataset = BraTS3DUnifiedDataset(
            data_root=args.data_root,
            phase='train',
            volume_size=tuple(config.data.volume_size)
        )
        
        # For validation, we can use a subset of training data or separate val data
        val_dataset = BraTS3DUnifiedDataset(
            data_root=args.data_root,
            phase='train',  # Use same for now, can change to 'val' if available
            volume_size=tuple(config.data.volume_size)
        )
        
        # For debugging, use smaller datasets
        if args.debug:
            from torch.utils.data import Subset
            train_indices = list(range(min(10, len(train_dataset))))
            val_indices = list(range(min(5, len(val_dataset))))
            train_dataset = Subset(train_dataset, train_indices)
            val_dataset = Subset(val_dataset, val_indices)
            logging.info("Debug mode: using smaller datasets")
        
    except Exception as e:
        logging.error(f"Failed to create datasets: {e}")
        raise
    
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
        batch_size=1,  # Smaller batch for validation
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    logging.info(f"Train samples: {len(train_dataset)}")
    logging.info(f"Val samples: {len(val_dataset)}")
    logging.info(f"Train batches: {len(train_loader)}")
    
    # Model
    logging.info("Setting up 3D Fast-DDPM model...")
    try:
        model = Model3D(config).to(device)
        if torch.cuda.device_count() > 1 and not args.debug:
            model = torch.nn.DataParallel(model)
            logging.info(f"Using {torch.cuda.device_count()} GPUs")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logging.info(f"Total parameters: {total_params:,}")
        logging.info(f"Trainable parameters: {trainable_params:,}")
        
    except Exception as e:
        logging.error(f"Failed to create model: {e}")
        raise
    
    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=config.training.learning_rate,
        weight_decay=getattr(config.training, 'weight_decay', 0),
        betas=(0.9, 0.999)
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=config.training.epochs,
        eta_min=config.training.learning_rate * 0.01
    )
    
    # Diffusion setup
    betas = get_beta_schedule(
        beta_schedule=config.diffusion.beta_schedule,
        beta_start=config.diffusion.beta_start,
        beta_end=config.diffusion.beta_end,
        num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
    )
    betas = torch.from_numpy(betas).float().to(device)
    num_timesteps = betas.shape[0]
    
    # Get timestep schedule
    t_intervals = get_timestep_schedule(args.scheduler_type, args.timesteps, num_timesteps)
    
    # Mixed precision for memory efficiency
    scaler = GradScaler()
    
    # Resume training if requested
    start_epoch = 0
    start_step = 0
    if args.resume and args.resume_path and os.path.exists(args.resume_path):
        logging.info(f"Resuming from checkpoint: {args.resume_path}")
        start_epoch, start_step, _ = load_checkpoint(
            args.resume_path, model, optimizer, scaler, device
        )
        logging.info(f"Resumed from epoch {start_epoch}, step {start_step}")
    
    # Training loop
    logging.info("Starting 3D Fast-DDPM training...")
    logging.info(f"Scheduler type: {args.scheduler_type}, Timesteps: {args.timesteps}")
    logging.info(f"Volume size: {config.data.volume_size}")
    logging.info(f"Batch size: {config.training.batch_size}")
    
    step = start_step
    best_val_loss = float('inf')
    
    for epoch in range(start_epoch, config.training.epochs):
        model.train()
        epoch_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config.training.epochs}')
        for batch_idx, batch in enumerate(pbar):
            try:
                optimizer.zero_grad()
                
                # Unified 4→4 training data
                inputs = batch['input'].to(device)  # [B, 4, H, W, D]
                targets = batch['target'].to(device)  # [B, H, W, D]
                targets = targets.unsqueeze(1)  # [B, 1, H, W, D]
                
                n = inputs.size(0)
                step += 1
                
                # Fast-DDPM timestep selection with antithetic sampling
                idx_1 = torch.randint(0, len(t_intervals), size=(n // 2 + 1,))
                idx_2 = len(t_intervals) - idx_1 - 1
                idx = torch.cat([idx_1, idx_2], dim=0)[:n]
                t = t_intervals[idx].to(device)
                
                # Random noise
                e = torch.randn_like(targets)
                
                with autocast():
                    # 3D unified loss
                    loss = sg_noise_estimation_loss(model, inputs, targets, t, e, betas)
                
                scaler.scale(loss).backward()
                
                # Gradient clipping
                if hasattr(config.training, 'gradient_clip'):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.gradient_clip)
                
                scaler.step(optimizer)
                scaler.update()
                
                epoch_loss += loss.item()
                pbar.set_postfix({
                    'loss': f'{loss.item():.6f}',
                    'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'
                })
                
                # Save checkpoint
                if step % getattr(config.training, 'save_every', 1000) == 0:
                    save_checkpoint(
                        model, optimizer, scaler, step, epoch, loss.item(), 
                        config, os.path.join(log_dir, f'ckpt_{step}.pth')
                    )
                
                # Validation
                if step % getattr(config.training, 'validate_every', 500) == 0:
                    val_loss = validate_model(model, val_loader, device, betas, args.timesteps)
                    logging.info(f'Step {step} - Val Loss: {val_loss:.6f}')
                    
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        save_checkpoint(
                            model, optimizer, scaler, step, epoch, val_loss,
                            config, os.path.join(log_dir, 'best_model.pth')
                        )
                        logging.info(f'New best validation loss: {val_loss:.6f}')
                
            except Exception as e:
                logging.error(f"Error in training step {step}: {e}")
                if args.debug:
                    raise
                continue
        
        # End of epoch
        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)
        logging.info(f'Epoch {epoch+1} - Average Loss: {avg_loss:.6f}, LR: {optimizer.param_groups[0]["lr"]:.2e}')
        
        # Save epoch checkpoint
        save_checkpoint(
            model, optimizer, scaler, step, epoch, avg_loss,
            config, os.path.join(log_dir, 'ckpt.pth')
        )
    
    logging.info("Training completed!")
    logging.info(f"Best validation loss: {best_val_loss:.6f}")
    logging.info(f"Final model saved at: {os.path.join(log_dir, 'ckpt.pth')}")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        raise