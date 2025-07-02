#!/usr/bin/env python3
"""
Main training script for 3D Fast-DDPM on BraTS data
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from tqdm import tqdm
import logging
import time

# Add the current directory to path to import modules
sys.path.append('.')
sys.path.append('..')

# Import project modules
try:
    from data.brain_3d_unified import BraTS3DUnifiedDataset
    from models.fast_ddpm_3d import FastDDPM3D
    from functions.losses import streamlined_4to1_loss
except ImportError as e:
    logging.error(f"Failed to import modules: {e}")
    sys.exit(1)

# Import our utility modules
from memory_utils import MemoryManager, DistributedMemoryManager, safe_batch_processing, optimize_model_for_memory, dynamic_batch_size_adjustment
from distributed_utils import setup_distributed, cleanup_distributed, is_main_process, get_effective_batch_size, launch_distributed_training
from training_utils import (
    load_config, setup_logging, setup_wandb, save_checkpoint, load_checkpoint,
    validate_model, monitor_scaler_health, get_beta_schedule, get_timestep_schedule,
    log_samples_to_wandb, create_experiment_directory, setup_model_for_training,
    get_fixed_validation_batch
)

# Optional wandb import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='3D Fast-DDPM Training for BraTS')
    parser.add_argument('--config', type=str, default='configs/fast_ddpm_3d.yml', help='Path to config file')
    parser.add_argument('--data_root', type=str, required=True, help='Path to BraTS data')
    parser.add_argument('--exp', type=str, default='./experiments', help='Experiment directory')
    parser.add_argument('--doc', type=str, default='fast_ddpm_3d_brats', help='Experiment name')
    parser.add_argument('--timesteps', type=int, default=10, help='Number of timesteps (Fast-DDPM advantage)')
    parser.add_argument('--scheduler_type', type=str, default='uniform', choices=['uniform', 'non-uniform'], help='Timestep scheduler')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID (for single GPU) or start GPU ID (for multi-GPU)')
    parser.add_argument('--resume', action='store_true', help='Resume training')
    parser.add_argument('--resume_path', type=str, help='Path to checkpoint to resume from')
    parser.add_argument('--debug', action='store_true', help='Debug mode with smaller dataset')
    parser.add_argument('--log_every_n_steps', type=int, default=50, help='Log training progress every N steps')
    parser.add_argument('--use_wandb', action='store_true', help='Use Weights & Biases for logging')
    parser.add_argument('--wandb_project', type=str, default='fast-ddpm-3d-brats', help='W&B project name')
    parser.add_argument('--wandb_entity', type=str, default=None, help='W&B entity (username or team)')
    parser.add_argument('--sample_every', type=int, default=2000, help='Generate samples for W&B every N steps')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4, help='Gradient accumulation steps')
    
    # Multi-GPU arguments
    parser.add_argument('--multi_gpu', action='store_true', help='Enable multi-GPU training')
    parser.add_argument('--distributed', action='store_true', help='Use DistributedDataParallel (recommended for multi-node)')
    parser.add_argument('--world_size', type=int, default=-1, help='Number of processes for distributed training')
    parser.add_argument('--rank', type=int, default=-1, help='Process rank for distributed training')
    parser.add_argument('--dist_url', type=str, default='env://', help='URL for distributed training setup')
    parser.add_argument('--dist_backend', type=str, default='nccl', help='Distributed backend')
    parser.add_argument('--local_rank', type=int, default=-1, help='Local rank for distributed training')

    return parser.parse_args()


def setup_datasets(args, config, rank):
    """Setup training and validation datasets"""
    if is_main_process(rank):
        logging.info("Setting up datasets...")
    
    # Use provided volume size or default
    volume_size = tuple(config.data.volume_size) if hasattr(config.data, 'volume_size') else (80, 80, 80)
    if is_main_process(rank):
        print(f"Using volume size: {volume_size}")
    
    # Check if data root exists
    if not os.path.exists(args.data_root):
        raise FileNotFoundError(f"Data root not found: {args.data_root}")
    
    try:
        train_dataset = BraTS3DUnifiedDataset(
            data_root=args.data_root,
            phase='train',
            volume_size=volume_size
        )
        
        val_dataset = BraTS3DUnifiedDataset(
            data_root=args.data_root,
            phase='val',
            volume_size=volume_size
        )
        
        if args.debug:
            # Use larger debug dataset: ~10% of total for meaningful epochs
            debug_train_size = min(125, len(train_dataset))  # ~125 cases = ~125 batches per epoch
            debug_val_size = min(25, len(val_dataset))       # ~25 cases for validation
            train_indices = list(range(debug_train_size))
            val_indices = list(range(debug_val_size))
            train_dataset = Subset(train_dataset, train_indices)
            val_dataset = Subset(val_dataset, val_indices)
            if is_main_process(rank):
                logging.info(f"Debug mode: using {debug_train_size} train samples, {debug_val_size} val samples")
        
        return train_dataset, val_dataset
        
    except Exception as e:
        if is_main_process(rank):
            logging.error(f"Failed to create datasets: {e}")
        raise


def setup_dataloaders(train_dataset, val_dataset, config, args, rank, world_size):
    """Setup data loaders with proper distributed sampling"""
    # Setup distributed sampler if using distributed training
    train_sampler = None
    if args.distributed:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=True
        )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.training.batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),  # Don't shuffle if using DistributedSampler
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

    if is_main_process(rank):
        logging.info(f"Train samples: {len(train_dataset)}")
        logging.info(f"Val samples: {len(val_dataset)}")
        logging.info(f"Train batches: {len(train_loader)}")
    
    return train_loader, val_loader, train_sampler


def setup_model_and_optimizer(config, device, rank, world_size, args):
    """Setup model, optimizer, and scheduler"""
    if is_main_process(rank):
        logging.info("Setting up 3D Fast-DDPM model...")
    
    try:
        model = FastDDPM3D(config).to(device)
        
        # Setup for distributed/multi-GPU training
        model = setup_model_for_training(model, args, rank, world_size)
        
        if is_main_process(rank):
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logging.info(f"Total parameters: {total_params:,}")
            logging.info(f"Trainable parameters: {trainable_params:,}")
        
    except Exception as e:
        if is_main_process(rank):
            logging.error(f"Failed to create model: {e}")
        raise
    
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
    
    # Initialize gradient scaler with config settings
    init_scale = getattr(config.training, 'loss_scale_init', 2048.0)
    growth_interval = getattr(config.training, 'loss_scale_growth', 2000)
    backoff_factor = getattr(config.training, 'loss_scale_backoff', 0.5)
    
    scaler = GradScaler(
        init_scale=init_scale,
        growth_factor=2.0,
        backoff_factor=backoff_factor,
        growth_interval=growth_interval
    )
    
    logging.info(f"Initialized GradScaler with init_scale={init_scale}, "
                 f"growth_interval={growth_interval}, backoff_factor={backoff_factor}")
    
    return betas, t_intervals, scaler


def training_loop(model, train_loader, val_loader, optimizer, scheduler, scaler, 
                 betas, t_intervals, config, args, rank, world_size, device):
    """Main training loop"""
    
    # Setup memory manager
    if args.distributed:
        memory_manager = DistributedMemoryManager(device, rank, world_size)
    else:
        memory_manager = MemoryManager(device)
    
    # Apply model optimizations for memory efficiency
    optimize_model_for_memory(model, enable_gradient_checkpointing=True)
    
    # Log initial memory state (only on main process)
    if is_main_process(rank):
        memory_manager.log_memory_usage("Initial: ")
    
    # Setup W&B and fixed validation batch
    use_wandb = False
    fixed_val_batch = None
    if is_main_process(rank):
        use_wandb = setup_wandb(args, config)
        if use_wandb:
            logging.info("✅ W&B initialized successfully")
            gradient_accumulation_steps = args.gradient_accumulation_steps
            effective_batch_size = get_effective_batch_size(
                config.training.batch_size, world_size, gradient_accumulation_steps
            )
            wandb.config.update({
                'gradient_accumulation_steps': gradient_accumulation_steps,
                'effective_batch_size': effective_batch_size,
                'world_size': world_size,
                'distributed': args.distributed,
                'multi_gpu': args.multi_gpu
            }, allow_val_change=True)
        elif args.use_wandb and not WANDB_AVAILABLE:
            logging.warning("⚠️ W&B requested but not installed. Install with: pip install wandb")
        
        fixed_val_batch = get_fixed_validation_batch(val_loader)
    
    # Resume training if requested
    start_epoch = 0
    start_step = 0
    if args.resume and args.resume_path and os.path.exists(args.resume_path):
        logging.info(f"Resuming from checkpoint: {args.resume_path}")
        try:
            start_epoch, start_step, _ = load_checkpoint(
                args.resume_path, model, optimizer, scaler, device
            )
            logging.info(f"Resumed from epoch {start_epoch}, step {start_step}")
        except Exception as e:
            logging.error(f"Failed to load checkpoint: {e}")
            raise
    
    # Training parameters
    global_step = start_step
    best_val_loss = float('inf')
    gradient_accumulation_steps = args.gradient_accumulation_steps
    log_dir = os.path.join(args.exp, 'logs', args.doc)
    os.makedirs(log_dir, exist_ok=True)
    
    logging.info("Starting training...")
    logging.info(f"Scheduler type: {args.scheduler_type}, Timesteps: {args.timesteps}")
    logging.info(f"Batch size: {config.training.batch_size}")
    if is_main_process(rank):
        logging.info(f"Gradient accumulation steps: {gradient_accumulation_steps}")
        effective_batch_size = get_effective_batch_size(
            config.training.batch_size, world_size, gradient_accumulation_steps
        )
        logging.info(f"Effective batch size: {effective_batch_size}")
    
    # Training loop
    for epoch in range(start_epoch, config.training.epochs):
        model.train()
        
        # Set epoch for distributed sampler
        train_sampler = getattr(train_loader, 'sampler', None)
        if args.distributed and train_sampler is not None and hasattr(train_sampler, 'set_epoch'):
            train_sampler.set_epoch(epoch)
        
        epoch_loss = 0.0
        
        # Adaptive memory cleanup at start of epoch
        if args.distributed:
            memory_manager.sync_cleanup_across_processes()
        else:
            memory_manager.adaptive_cleanup()
            
        if is_main_process(rank):
            memory_manager.log_memory_usage(f"Epoch {epoch+1} start: ")
        
        # Only show progress bar on main process
        if is_main_process(rank):
            pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config.training.epochs}')
        else:
            pbar = train_loader
        
        # Initialize accumulation variables
        accumulated_loss = 0.0
        accumulation_steps = 0
        
        for batch_idx, batch in enumerate(pbar):
            try:
                step_start_time = time.time()
                
                # Periodic memory management
                if batch_idx % 10 == 0:
                    if args.distributed:
                        memory_manager.sync_cleanup_across_processes()
                    else:
                        memory_manager.adaptive_cleanup()
                
                # Zero gradients at the start of each accumulation cycle
                if accumulation_steps == 0:
                    optimizer.zero_grad()
                
                # Safe batch processing with enhanced memory management
                inputs = safe_batch_processing(batch, device, memory_manager)
                if inputs is None:
                    # Skip this batch if processing failed
                    if is_main_process(rank):
                        logging.warning(f"Skipping batch {batch_idx} due to processing failure")
                    continue
                
                targets = inputs['target'].unsqueeze(1)
                target_idx = inputs['target_idx'][0].item()
                
                n = inputs['input'].size(0)
                
                # Fast-DDPM antithetic sampling
                idx_1 = torch.randint(0, len(t_intervals), size=(n // 2 + 1,))
                idx_2 = len(t_intervals) - idx_1 - 1
                idx = torch.cat([idx_1, idx_2], dim=0)[:n]
                t = t_intervals[idx].to(device)
                
                e = torch.randn_like(targets)
                
                # Enhanced loss computation with memory management
                with autocast():
                    loss = streamlined_4to1_loss(model, inputs['input'], targets, t, e, b=betas, target_idx=target_idx)
                
                # Enhanced loss validation
                if torch.isnan(loss) or torch.isinf(loss):
                    if is_main_process(rank):
                        logging.error(f"Invalid loss detected: {loss.item()}")
                        logging.error(f"Input stats: min={inputs['input'].min():.3f}, max={inputs['input'].max():.3f}")
                        logging.error(f"Target stats: min={targets.min():.3f}, max={targets.max():.3f}")
                    
                    # Clear problematic tensors
                    del inputs, targets, e, loss
                    memory_manager.cleanup_gpu_memory(force=True)
                    
                    if args.debug:
                        raise ValueError("NaN/Inf loss - stopping training")
                    continue  # Skip this batch
                
                # Scale loss for gradient accumulation BEFORE backward pass
                scaled_loss = loss / gradient_accumulation_steps
                scaler.scale(scaled_loss).backward()
                accumulated_loss += loss.item()  # Accumulate the original (unscaled) loss for logging
                accumulation_steps += 1
                
                # Clear intermediate tensors to save memory
                del inputs, targets, e, loss, scaled_loss
                
                # Step optimizer when we've accumulated enough gradients
                if accumulation_steps >= gradient_accumulation_steps:
                    # Unscale gradients for gradient clipping
                    scaler.unscale_(optimizer)
                    
                    # Apply gradient clipping
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), 
                        max_norm=getattr(config.training, 'gradient_clip', float('inf'))
                    )
                    
                    # Step optimizer and update scaler
                    scaler.step(optimizer)
                    scaler.update()
                    
                    # Zero gradients for next accumulation cycle
                    optimizer.zero_grad()
                    
                    # Monitor gradient scaler health
                    if global_step % 100 == 0:  # Check every 100 steps
                        monitor_scaler_health(scaler, global_step)
                    
                    # Now we've completed one effective training step
                    global_step += 1
                    
                    # Calculate average loss over accumulation steps
                    average_loss = accumulated_loss / gradient_accumulation_steps
                    epoch_loss += average_loss
                    
                    step_time = time.time() - step_start_time
                    
                    # Enhanced progress bar with memory info
                    if is_main_process(rank):
                        memory_stats = memory_manager.get_memory_stats()
                        memory_gb = memory_stats.get('allocated_gb', 0) if memory_stats else 0
                        
                        pbar.set_postfix({
                            'loss': f'{average_loss:.6f}',
                            'lr': f'{optimizer.param_groups[0]["lr"]:.2e}',
                            'target': f'{target_idx}',
                            'step': global_step,
                            'mem_gb': f'{memory_gb:.1f}',
                            'step_time': f'{step_time:.2f}s'
                        })
                    
                    # W&B logging
                    if use_wandb:
                        wandb.log({
                            'train/loss': average_loss,
                            'train/learning_rate': optimizer.param_groups[0]["lr"],
                            'train/epoch': epoch,
                            'train/step': global_step,
                            'train/target_idx': target_idx,
                            'train/grad_norm': grad_norm.item() if grad_norm > 0 else 0,
                            'train/loss_scale': scaler.get_scale(),
                            'system/gpu_memory_mb': torch.cuda.max_memory_allocated() / 1024 / 1024,
                            'system/step_time': step_time,
                        }, step=global_step)
                    
                    # Reset accumulation variables for next cycle
                    accumulated_loss = 0.0
                    accumulation_steps = 0
                    
                    # Logging
                    if global_step % args.log_every_n_steps == 0 and is_main_process(rank):
                        logging.info(f'Epoch {epoch+1}/{config.training.epochs}, '
                                     f'Step {global_step} - '
                                     f'Loss: {average_loss:.6f}, '
                                     f'Target: {target_idx}, '
                                     f'LR: {optimizer.param_groups[0]["lr"]:.2e}, '
                                     f'Time: {step_time:.2f}s')
                    
                    # Save and validate periodically (only on main process)
                    save_every = getattr(config.training, 'save_every', 2000)
                    validate_every = getattr(config.training, 'validate_every', 1000)
                    
                    if global_step % save_every == 0:
                        save_checkpoint(
                            model, optimizer, scaler, global_step, epoch, average_loss, 
                            config, os.path.join(log_dir, f'ckpt_{global_step}.pth'), rank
                        )
                    
                    if global_step % validate_every == 0:
                        val_loss = validate_model(model, val_loader, device, betas, t_intervals)
                        if is_main_process(rank):
                            logging.info(f'Step {global_step} - Val Loss: {val_loss:.6f}')
                        
                        if use_wandb and is_main_process(rank):
                            wandb.log({'val/loss': val_loss, 'val/epoch': epoch}, step=global_step)
                        
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            save_checkpoint(
                                model, optimizer, scaler, global_step, epoch, val_loss,
                                config, os.path.join(log_dir, 'best_model.pth'), rank
                            )
                            if is_main_process(rank):
                                logging.info(f'New best validation loss: {val_loss:.6f}')
                                if use_wandb:
                                    wandb.run.summary["best_val_loss"] = val_loss
                                    wandb.run.summary["best_val_step"] = global_step
                    
                    # Sample logging (only on main process)
                    if (use_wandb and is_main_process(rank) and fixed_val_batch and 
                        global_step % args.sample_every == 0):
                        actual_model = model.module if isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel)) else model
                        log_samples_to_wandb(
                            actual_model, fixed_val_batch, t_intervals, betas, device, global_step
                        )
                
                # Enhanced periodic memory cleanup
                if (batch_idx + 1) % 5 == 0:  # More frequent cleanup
                    memory_manager.adaptive_cleanup()
            
            except Exception as e:
                if "out of memory" in str(e).lower():
                    logging.error(f"OOM error in batch {batch_idx}: {e}")
                    memory_manager.cleanup_gpu_memory(force=True)
                    # Reset optimizer and accumulation state
                    optimizer.zero_grad()
                    accumulated_loss = 0.0
                    accumulation_steps = 0
                    continue
                else:
                    logging.error(f"Error in training step: {e}")
                    if args.debug:
                        raise
                    # Reset optimizer and accumulation state
                    optimizer.zero_grad()
                    accumulated_loss = 0.0
                    accumulation_steps = 0
                    continue
        
        # End of epoch processing
        scheduler.step()
        
        # Log epoch summary
        avg_loss = epoch_loss / max(1, global_step - start_step) if global_step > start_step else 0
        if is_main_process(rank):
            memory_manager.log_memory_usage(f"Epoch {epoch+1} end: ")
            logging.info(f'Epoch {epoch+1} - Average Loss: {avg_loss:.6f}, '
                        f'LR: {optimizer.param_groups[0]["lr"]:.2e}')
        
        if use_wandb and is_main_process(rank):
            wandb.log({
                'epoch/avg_loss': avg_loss,
                'epoch/learning_rate': optimizer.param_groups[0]["lr"],
                'epoch/epoch': epoch + 1,
            }, step=global_step)
        
        # Comprehensive memory cleanup at end of epoch
        if args.distributed:
            memory_manager.sync_cleanup_across_processes()
        else:
            memory_manager.cleanup_gpu_memory(force=True)
        
        # Save checkpoint at end of epoch (every 10 epochs)
        if (epoch + 1) % 10 == 0:
            save_checkpoint(
                model, optimizer, scaler, global_step, epoch, avg_loss,
                config, os.path.join(log_dir, f'ckpt_epoch_{epoch+1}.pth'), rank
            )
    
    if is_main_process(rank):
        logging.info("Training completed!")
        logging.info(f"Best validation loss: {best_val_loss:.6f}")
        
        if use_wandb:
            wandb.finish()


def main(args=None):
    """Main training function"""
    if args is None:
        args = parse_args()
    
    print(f"=== MAIN FUNCTION START ===")
    print(f"Process PID: {os.getpid()}")
    print(f"Args distributed: {args.distributed}")
    print(f"Environment variables:")
    for key in ['RANK', 'WORLD_SIZE', 'LOCAL_RANK', 'MASTER_ADDR', 'MASTER_PORT', 'CUDA_VISIBLE_DEVICES']:
        print(f"  {key}: {os.environ.get(key, 'not set')}")
    
    # Setup distributed training
    rank, world_size, device = setup_distributed(args)
    
    # Only setup logging on main process to avoid duplicate logs
    if is_main_process(rank):
        print(f"Using device: {device}")
        print(f"World size: {world_size}, Rank: {rank}")
        
        # Setup logging
        log_dir = os.path.join(args.exp, 'logs', args.doc)
        setup_logging(log_dir, args)
        
        # Load config
        try:
            config = load_config(args.config)
        except FileNotFoundError:
            logging.error(f"Config file not found: {args.config}")
            sys.exit(1)
            
        # Create experiment directory
        create_experiment_directory(args, rank)
    else:
        # Load config on non-main processes too
        config = load_config(args.config)
    
    config.device = device
    
    # Synchronize all processes before continuing
    if args.distributed:
        dist.barrier()
    
    # Setup datasets and dataloaders
    train_dataset, val_dataset = setup_datasets(args, config, rank)
    train_loader, val_loader, train_sampler = setup_dataloaders(
        train_dataset, val_dataset, config, args, rank, world_size
    )
    
    # Setup model, optimizer, and scheduler
    model, optimizer, scheduler = setup_model_and_optimizer(config, device, rank, world_size, args)
    
    # Setup diffusion and gradient scaler
    betas, t_intervals, scaler = setup_diffusion_and_scaler(config, device, args)
    
    # Run training loop
    training_loop(
        model, train_loader, val_loader, optimizer, scheduler, scaler,
        betas, t_intervals, config, args, rank, world_size, device
    )
    
    # Clean up distributed training
    cleanup_distributed()


if __name__ == '__main__':
    args = parse_args()
    
    try:
        # Check if we're already in a distributed process (launched by train_multi_gpu.py)
        if args.distributed and ('RANK' in os.environ and 'WORLD_SIZE' in os.environ):
            # We're already in a distributed process, call main directly
            print(f"Process launched with RANK={os.environ.get('RANK')}, LOCAL_RANK={os.environ.get('LOCAL_RANK')}")
            main(args)
        else:
            # Use internal distributed launcher
            launch_distributed_training(args, main)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        if WANDB_AVAILABLE and wandb.run is not None:
            wandb.finish()
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        if WANDB_AVAILABLE and wandb.run is not None:
            wandb.finish()
        raise