#!/usr/bin/env python3
"""
Training script for 3D Fast-DDPM on BraTS dataset
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

# Add the current directory to path to import modules
sys.path.append('.')
sys.path.append('..')

# Import project modules
from data.brain_3d_unified import BraTS3DUnifiedDataset
from models.fast_ddpm_3d import FastDDPM3D
from functions.losses import brats_4to1_enhanced_loss, Simple3DPerceptualNet
from functions.denoising_3d import generalized_steps_3d, unified_4to1_generalized_steps_3d

# Minimal training utils
from training_utils import load_config, get_beta_schedule, get_timestep_schedule

# Optional wandb import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def sample_timesteps(t_intervals, batch_size):
    """Simple timestep sampling"""
    idx = torch.randint(0, len(t_intervals), size=(batch_size,))
    return t_intervals[idx]



def generate_and_log_samples(model, val_loader, betas, t_intervals, device, global_step, num_samples=6):
    """Generate and log samples to W&B"""
    model.eval()
    
    with torch.no_grad():
        try:
            val_iter = iter(val_loader)
            sample_batches = []
            
            # Collect samples
            for _ in range(min(num_samples, len(val_loader))):
                try:
                    batch = next(val_iter)
                    sample_batches.append(batch)
                except StopIteration:
                    break
            
            if not sample_batches:
                return
            
            images_to_log = []
            
            for i, batch in enumerate(sample_batches):
                inputs = batch['input'][:1].to(device)
                targets = batch['target'][:1].unsqueeze(1).to(device)
                target_idx = batch['target_idx'][0].item()
                
                # Generate samples
                x_target_noise = torch.randn_like(targets)
                
                generated_sequence, x0_preds = unified_4to1_generalized_steps_3d(
                    x=x_target_noise,
                    x_available=inputs,
                    target_idx=target_idx, 
                    seq=t_intervals.cpu().numpy(),
                    model=model,
                    b=betas
                )
                
                generated = generated_sequence[-1] if generated_sequence else None
                
                if generated is None:
                    continue
                
                # Take middle slice for visualization
                middle_slice = inputs.size(-1) // 2
                
                # Extract all modality images and target
                modality_names = ['t1n', 't1c', 't2w', 't2f']
                all_images = []
                caption_parts = []
                
                # Normalize for display
                def safe_normalize(img):
                    img_min, img_max = img.min(), img.max()
                    if img_max > img_min and not np.isnan(img_min) and not np.isnan(img_max):
                        normalized = (img - img_min) / (img_max - img_min)
                        return np.clip(normalized, 0, 1)
                    else:
                        return np.zeros_like(img)
                
                # Replace the target channel in inputs with the noise used for generation for visualization
                inputs_with_noise = inputs.clone()
                inputs_with_noise[0, target_idx] = x_target_noise[0, 0]

                # Add all 4 input modalities (use inputs_with_noise for visualization)
                for j in range(4):
                    input_img = inputs_with_noise[0, j, :, :, middle_slice].cpu().numpy()
                    input_img_norm = safe_normalize(input_img)
                    all_images.append(input_img_norm)
                    if j == target_idx:
                        caption_parts.append(f"{modality_names[j]} (noise)")
                    else:
                        caption_parts.append(modality_names[j])
                
                # Add generated target (second to last)
                gen_img = generated[0, 0, :, :, middle_slice].cpu().numpy()
                gen_img_norm = safe_normalize(gen_img)
                all_images.append(gen_img_norm)
                target_mod = batch['target_modality'][0] if 'target_modality' in batch else modality_names[target_idx]
                caption_parts.append(f"{target_mod} Gen")
                
                # Add ground truth target (last)
                target_img = targets[0, 0, :, :, middle_slice].cpu().numpy()
                target_img_norm = safe_normalize(target_img)
                all_images.append(target_img_norm)
                caption_parts.append(f"{target_mod} GT")
                
                # Create side-by-side comparison (6 images total)
                comparison = np.concatenate(all_images, axis=1)
                comparison_uint8 = (comparison * 255).astype(np.uint8)
                
                # Create caption
                caption = f"Sample {i+1}: " + " | ".join(caption_parts)
                
                # Create W&B image
                wandb_img = wandb.Image(
                    comparison_uint8,
                    caption=caption
                )
                images_to_log.append(wandb_img)
                
                if len(images_to_log) >= num_samples:
                    break
            
            # Log to W&B
            if images_to_log:
                wandb.log({
                    "samples/generated_images": images_to_log,
                    "samples/step": global_step,
                    "samples/count": len(images_to_log)
                }, step=global_step)
            
        except Exception as e:
            logging.error(f"Failed to generate samples: {str(e)}")
    
    model.train()


def training_loop(model, train_loader, val_loader, optimizer, scheduler, scaler, 
                 betas, t_intervals, config, args, device):
    """Simplified training loop"""
    
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
        logging.info("W&B initialized successfully")
    elif args.use_wandb and not WANDB_AVAILABLE:
        logging.warning("W&B requested but not installed. Install with: pip install wandb")
    
    # Training parameters
    global_step = 0
    log_dir = os.path.join(args.exp, 'logs', args.doc)
    os.makedirs(log_dir, exist_ok=True)
    
    logging.info("Starting training...")
    logging.info(f"Batch size: {config.training.batch_size}")
    logging.info(f"Log every: {config.training.log_every_n_steps} steps")
    logging.info(f"Mixed precision: {not args.no_mixed_precision}")

    perceptual_net = Simple3DPerceptualNet(input_channels=1).to(device)
    
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
                
                # Input validation
                if torch.isnan(inputs).any() or torch.isnan(targets).any():
                    raise ValueError(f"NaN detected in batch {batch_idx}")
                
                if torch.isinf(inputs).any() or torch.isinf(targets).any():
                    raise ValueError(f"Inf detected in batch {batch_idx}")
                
                n = inputs.size(0)
                
                # Timestep sampling
                t = sample_timesteps(t_intervals, n).to(device)
                e = torch.randn_like(targets)
                
                # Compute loss with mixed precision

                with autocast(enabled=not args.no_mixed_precision):
                    loss = brats_4to1_enhanced_loss(model, inputs, targets, t, e, b=betas, target_idx=target_idx,
                        perceptual_net=perceptual_net)
                
                # Loss validation
                if torch.isnan(loss) or torch.isinf(loss) or loss.item() < 0:
                    raise ValueError(f"Invalid loss: {loss.item()}")
                
                # Backward pass
                if args.no_mixed_precision:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_norm)
                    optimizer.step()
                else:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_norm)
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
                    'step': global_step
                })
                
                # W&B logging
                if use_wandb:
                    wandb.log({
                        'train/loss': loss.item(),
                        'train/learning_rate': optimizer.param_groups[0]["lr"],
                        'train/step': global_step,
                        'train/target_idx': target_idx,
                    }, step=global_step)
                    
                    # Generate and log samples for debugging
                    if global_step % args.sample_every == 0 and global_step > 0:
                        generate_and_log_samples(
                            model, val_loader, betas, t_intervals, device, global_step, num_samples=6
                        )
                
                # Periodic logging
                if global_step % config.training.log_every_n_steps == 0:
                    logging.info(f'Epoch {epoch+1}, Step {global_step} - Loss: {loss.item():.6f}, '
                               f'LR: {optimizer.param_groups[0]["lr"]:.2e}, Target: {target_idx}')
                    
            except Exception as e:
                logging.error(f"Training batch {batch_idx} failed: {str(e)}")
                raise  # Fail fast instead of continuing
        
        # End of epoch
        scheduler.step()
        if valid_batches > 0:
            avg_loss = epoch_loss / valid_batches
            logging.info(f'Epoch {epoch+1} - Average Loss: {avg_loss:.6f} '
                        f'({valid_batches}/{len(train_loader)} valid batches)')
            
            if use_wandb:
                wandb.log({
                    'epoch/avg_loss': avg_loss,
                    'epoch/learning_rate': optimizer.param_groups[0]["lr"],
                    'epoch/epoch': epoch + 1,
                    'epoch/valid_batches': valid_batches,
                }, step=global_step)
        else:
            logging.warning(f'Epoch {epoch+1} - No valid batches processed!')
    
    logging.info("Training completed!")
    
    if use_wandb:
        wandb.finish()


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='3D Fast-DDPM Training for BraTS')
    parser.add_argument('--config', type=str, default='configs/fast_ddpm_3d.yml', help='Path to config file')
    parser.add_argument('--data_root', type=str, required=True, help='Path to BraTS data')
    parser.add_argument('--exp', type=str, default='./experiments', help='Experiment directory')
    parser.add_argument('--doc', type=str, default='fast_ddpm_3d_brats', help='Experiment name')
    parser.add_argument('--timesteps', type=int, default=10, help='Number of timesteps')
    parser.add_argument('--scheduler_type', type=str, default='uniform', choices=['uniform', 'non-uniform'], help='Timestep scheduler')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID')
    parser.add_argument('--resume', action='store_true', help='Resume training')
    parser.add_argument('--resume_path', type=str, help='Path to checkpoint to resume from')
    parser.add_argument('--debug', action='store_true', help='Debug mode with smaller dataset')
    parser.add_argument('--log_every_n_steps', type=int, default=None, help='Log training progress every N steps (overrides config)')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size (overrides config)')
    parser.add_argument('--use_wandb', action='store_true', help='Use Weights & Biases for logging')
    parser.add_argument('--wandb_project', type=str, default='fast-ddpm-3d-brats', help='W&B project name')
    parser.add_argument('--wandb_entity', type=str, default=None, help='W&B entity')
    parser.add_argument('--sample_every', type=int, default=2000, help='Generate samples every N steps')
    
    # Training options
    parser.add_argument('--no_mixed_precision', action='store_true', help='Disable mixed precision training')
    parser.add_argument('--clip_norm', type=float, default=1.0, help='Gradient clipping norm')
    
    # Data processing options
    parser.add_argument('--use_full_volumes', action='store_true', 
                       help='Use full volumes instead of random patches')
    parser.add_argument('--input_size', nargs=3, type=int, default=[80, 80, 80],
                       help='Input size for full volumes (default: 80 80 80)')
    parser.add_argument('--patch_size', nargs=3, type=int, default=[80, 80, 80],
                       help='Patch size for patch-based training (default: 80 80 80)')
    parser.add_argument('--crops_per_volume', type=int, default=4,
                       help='Number of crops per volume for patch-based training (default: 4)')

    return parser.parse_args()


def setup_datasets(args, config):
    """Setup training and validation datasets"""
    logging.info("Setting up datasets...")
    
    if not os.path.exists(args.data_root):
        raise FileNotFoundError(f"Data root not found: {args.data_root}")
    
    # Determine sizes based on mode
    if args.use_full_volumes:
        input_size = tuple(args.input_size)
        crop_size = (64, 64, 64)  # Not used in full volume mode
        mode_info = f"full volumes ({input_size})"
    else:
        input_size = tuple(args.patch_size)
        crop_size = tuple(args.patch_size)
        mode_info = f"patches ({crop_size})"
    
    logging.info(f"Data processing mode: {mode_info}")
    
    train_dataset = BraTS3DUnifiedDataset(
        data_root=args.data_root,
        phase='train',
        crop_size=crop_size,
        use_full_volumes=args.use_full_volumes,
        input_size=input_size,
        crops_per_volume=args.crops_per_volume
    )
    
    val_dataset = BraTS3DUnifiedDataset(
        data_root=args.data_root,
        phase='val',
        crop_size=crop_size,
        use_full_volumes=args.use_full_volumes,
        input_size=input_size,
        crops_per_volume=1  # Always use 1 crop per volume for validation
    )
    
    if args.debug:
        debug_train_size = min(10, len(train_dataset))
        debug_val_size = min(5, len(val_dataset))
        train_indices = list(range(debug_train_size))
        val_indices = list(range(debug_val_size))
        train_dataset = Subset(train_dataset, train_indices)
        val_dataset = Subset(val_dataset, val_indices)
        logging.info(f"Debug mode: using {debug_train_size} train samples, {debug_val_size} val samples")
    
    return train_dataset, val_dataset


def custom_collate_fn(batch):
    """Custom collate function to handle string lists correctly"""
    # Default collate for most fields
    collated = {}
    
    # Handle each key separately
    for key in batch[0].keys():
        if key in ['available_modalities', 'successfully_loaded_modalities']:
            # Keep list fields as lists of lists
            collated[key] = [item[key] for item in batch]
        elif key in ['case_name', 'target_modality', 'processing_mode']:
            # String fields - keep as list of strings
            collated[key] = [item[key] for item in batch]
        elif key in ['target_idx']:
            # Integer fields - convert to tensor
            collated[key] = torch.tensor([item[key] for item in batch])
        elif key == 'crop_coords':
            # Keep as list of tuples (may be None for full volumes)
            collated[key] = [item[key] for item in batch]
        else:
            # Tensor fields - use default stacking
            collated[key] = torch.stack([item[key] for item in batch])
    
    return collated


def setup_dataloaders(train_dataset, val_dataset, config):
    """Setup data loaders"""
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=getattr(config.data, 'num_workers', 2),
        pin_memory=True,
        drop_last=True,
        collate_fn=custom_collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,  # Keep batch size 1 for validation
        shuffle=False,
        num_workers=1,
        pin_memory=True,
        collate_fn=custom_collate_fn
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
    
    scaler = GradScaler() if not args.no_mixed_precision else None
    
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
    
    # Override config values with command line arguments
    if args.batch_size is not None:
        config.training.batch_size = args.batch_size
        logging.info(f"Overriding batch size to: {args.batch_size}")
    
    if args.log_every_n_steps is not None:
        config.training.log_every_n_steps = args.log_every_n_steps
        logging.info(f"Overriding log frequency to every: {args.log_every_n_steps} steps")
    
    # Log data processing configuration
    if args.use_full_volumes:
        logging.info(f"Using full volume mode with input size: {tuple(args.input_size)}")
    else:
        logging.info(f"Using patch mode with patch size: {tuple(args.patch_size)}")
    
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
    
    # Run training loop
    training_loop(
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