#!/usr/bin/env python3
"""
3D Fast-DDPM training script for BraTS - Updated to work with actual Fast-DDPM
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

# Add the current directory to path to import modules
sys.path.append('.')

# Import 3D components (create these files)
from models.diffusion_3d import Model3D
from functions.denoising_3d import generalized_steps_3d, unified_4to4_generalized_steps
from data.brain_3d_unified import BraTS3DUnifiedDataset
from functions.losses import sg_noise_estimation_loss

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/basic_3d.yml', help='Path to config file')
    parser.add_argument('--data_root', type=str, required=True, help='Path to BraTS data')
    parser.add_argument('--exp', type=str, default='./experiments', help='Experiment directory')
    parser.add_argument('--doc', type=str, default='fast_ddpm_3d_brats', help='Experiment name')
    parser.add_argument('--timesteps', type=int, default=10, help='Number of timesteps (Fast-DDPM advantage)')
    parser.add_argument('--scheduler_type', type=str, default='uniform', help='uniform or non-uniform')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID')
    parser.add_argument('--resume', action='store_true', help='Resume training')
    return parser.parse_args()

def dict2namespace(config):
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
    import numpy as np
    
    if beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    else:
        raise NotImplementedError(beta_schedule)
    
    assert betas.shape == (num_diffusion_timesteps,)
    return betas

def main():
    args = parse_args()
    
    # Setup device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Setup logging
    log_dir = os.path.join(args.exp, 'logs', args.doc)
    os.makedirs(log_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'train.log')),
            logging.StreamHandler()
        ]
    )
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    config = dict2namespace(config)
    config.device = device
    
    # Override data root
    config.data.train_dataroot = args.data_root
    
    # Dataset and DataLoader
    logging.info("Setting up dataset...")
    train_dataset = BraTS3DUnifiedDataset(
        data_root=args.data_root,
        phase='train',
        volume_size=tuple(config.data.volume_size)
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=True
    )
    
    logging.info(f"Train samples: {len(train_dataset)}")
    
    # Model
    logging.info("Setting up 3D Fast-DDPM model...")
    model = Model3D(config).to(device)
    model = torch.nn.DataParallel(model)
    
    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=config.optim.lr,
        weight_decay=config.optim.weight_decay,
        betas=(config.optim.beta1, 0.999)
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
    
    # Mixed precision for memory efficiency
    scaler = GradScaler()
    
    # Training loop
    logging.info("Starting 3D Fast-DDPM training...")
    logging.info(f"Scheduler type: {args.scheduler_type}, Timesteps: {args.timesteps}")
    
    step = 0
    for epoch in range(config.training.n_epochs):
        model.train()
        epoch_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}')
        for batch_idx, batch in enumerate(pbar):
            optimizer.zero_grad()
            
            # Unified 4→4 training data
            inputs = batch['input'].to(device)  # [B, 4, H, W, D] - available modalities
            targets = batch['target'].to(device)  # [B, H, W, D] - target modality
            targets = targets.unsqueeze(1)  # [B, 1, H, W, D]
            
            n = inputs.size(0)
            step += 1
            
            # Fast-DDPM timestep selection
            if args.scheduler_type == 'uniform':
                skip = num_timesteps // args.timesteps
                t_intervals = torch.arange(0, num_timesteps, skip)
            elif args.scheduler_type == 'non-uniform':
                # Non-uniform schedule from Fast-DDPM paper
                t_intervals = torch.tensor([0, 199, 399, 599, 699, 799, 849, 899, 949, 999])
                if args.timesteps != 10:
                    num_1 = int(args.timesteps * 0.4)
                    num_2 = int(args.timesteps * 0.6)
                    stage_1 = torch.linspace(0, 699, num_1 + 1)[:-1]
                    stage_2 = torch.linspace(699, 999, num_2)
                    stage_1 = torch.ceil(stage_1).long()
                    stage_2 = torch.ceil(stage_2).long()
                    t_intervals = torch.cat((stage_1, stage_2))
            
            # Antithetic sampling for stability
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
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
            # Save checkpoint
            if step % config.training.snapshot_freq == 0:
                checkpoint = {
                    'epoch': epoch,
                    'step': step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss.item(),
                    'config': config
                }
                torch.save(checkpoint, os.path.join(log_dir, f'ckpt_{step}.pth'))
                torch.save(checkpoint, os.path.join(log_dir, 'ckpt.pth'))
                logging.info(f'Saved checkpoint at step {step}')
        
        avg_loss = epoch_loss / len(train_loader)
        logging.info(f'Epoch {epoch+1} - Average Loss: {avg_loss:.6f}')
    
    logging.info("Training completed!")

if __name__ == '__main__':
    main()