"""
Simplified training utilities for 3D Fast-DDPM
"""
import os
import torch
import logging
import yaml
import numpy as np
import argparse

# Optional wandb import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


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


def load_config(config_path):
    """Load configuration file"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return dict2namespace(config)


def save_checkpoint(model, optimizer, scaler, step, epoch, loss, config, path):
    """Save training checkpoint"""
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'step': step,
        'epoch': epoch,
        'loss': loss,
        'config': config
    }
    
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(checkpoint, path)
    logging.info(f"Checkpoint saved: {path}")


def load_checkpoint(path, model, optimizer, scaler, device):
    """Load training checkpoint"""
    checkpoint = torch.load(path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scaler.load_state_dict(checkpoint['scaler_state_dict'])
    
    return checkpoint['epoch'], checkpoint['step'], checkpoint['loss']


def validate_model(model, val_loader, device, betas, t_intervals):
    """Run validation and return average loss"""
    from functions.losses import brats_4to1_loss
    
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in val_loader:
            try:
                # Move to device
                inputs = batch['input'].to(device)
                targets = batch['target'].unsqueeze(1).to(device)
                target_idx = batch['target_idx'][0].item()
                
                # Random timestep
                n = inputs.size(0)
                t_idx = torch.randint(0, len(t_intervals), size=(n,))
                t = t_intervals[t_idx].to(device)
                
                # Random noise
                e = torch.randn_like(targets)
                
                # Compute loss
                loss = brats_4to1_loss(
                    model, inputs, targets, t, e, 
                    b=betas, target_idx=target_idx
                )
                
                total_loss += loss.item()
                num_batches += 1
                
                # Limit validation batches for speed
                if num_batches >= 10:
                    break
                    
            except Exception as e:
                logging.warning(f"Validation batch failed: {e}")
                continue
    
    model.train()
    return total_loss / max(1, num_batches)


def get_beta_schedule(beta_schedule, beta_start, beta_end, num_diffusion_timesteps):
    """Get beta schedule for diffusion, supports linear and cosine."""
    if beta_schedule == "linear":
        betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "cosine":
        s = 0.008
        steps = num_diffusion_timesteps + 1
        x = np.linspace(0, num_diffusion_timesteps, steps, dtype=np.float64)
        alphas_cumprod = np.cos(((x / num_diffusion_timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        betas = np.clip(betas, 0, 0.999)
    else:
        raise ValueError(f"Unknown beta schedule: {beta_schedule}")
    return betas


def get_timestep_schedule(scheduler_type, num_timesteps, max_timesteps):
    """Get timestep schedule for Fast-DDPM"""
    if scheduler_type == "uniform":
        # Uniform spacing
        t_intervals = torch.linspace(0, max_timesteps - 1, num_timesteps, dtype=torch.long)
    elif scheduler_type == "non-uniform":
        # Quadratic spacing (more timesteps at the beginning)
        t_normalized = torch.linspace(0, 1, num_timesteps) ** 2
        t_intervals = (t_normalized * (max_timesteps - 1)).long()
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
    
    return t_intervals