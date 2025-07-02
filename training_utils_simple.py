"""
Simplified training utilities for 3D Fast-DDPM
"""
import os
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
import logging
import yaml
import numpy as np
import argparse
from tqdm import tqdm

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


def setup_logging(log_dir, args):
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


def setup_wandb(args, config):
    """Setup Weights & Biases logging"""
    if not args.use_wandb or not WANDB_AVAILABLE:
        return False
    
    try:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.doc,
            config={
                'model': dict(config.model.__dict__) if hasattr(config, 'model') else {},
                'training': dict(config.training.__dict__) if hasattr(config, 'training') else {},
                'diffusion': dict(config.diffusion.__dict__) if hasattr(config, 'diffusion') else {},
                'data': dict(config.data.__dict__) if hasattr(config, 'data') else {},
                'args': vars(args)
            }
        )
        return True
    except Exception as e:
        logging.warning(f"Failed to initialize W&B: {e}")
        return False


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
                inputs = {
                    'input': batch['input'].to(device),
                    'target': batch['target'].to(device),
                    'target_idx': batch['target_idx']
                }
                
                targets = inputs['target'].unsqueeze(1)
                target_idx = inputs['target_idx'][0].item()
                
                # Random timestep
                n = inputs['input'].size(0)
                t_idx = torch.randint(0, len(t_intervals), size=(n,))
                t = t_intervals[t_idx].to(device)
                
                # Random noise
                e = torch.randn_like(targets)
                
                # Compute loss
                loss = brats_4to1_loss(
                    model, inputs['input'], targets, t, e, 
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


def monitor_scaler_health(scaler, step):
    """Monitor gradient scaler health"""
    scale = scaler.get_scale()
    if scale < 1.0:
        logging.warning(f"Low gradient scale detected: {scale:.2e} at step {step}")


def get_beta_schedule(beta_schedule, beta_start, beta_end, num_diffusion_timesteps):
    """Get beta schedule for diffusion"""
    if beta_schedule == "linear":
        betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "cosine":
        # Cosine schedule
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


def get_fixed_validation_batch(val_loader):
    """Get a fixed validation batch for consistent sample logging"""
    try:
        fixed_val_batch = next(iter(val_loader))
        logging.info("Got fixed validation batch for sample logging")
        return fixed_val_batch
    except StopIteration:
        logging.warning("Validation loader is empty")
        return None


def log_samples_to_wandb(model, fixed_val_batch, t_intervals, betas, device, step):
    """Log sample images to W&B (simplified version)"""
    if not WANDB_AVAILABLE:
        return
    
    try:
        model.eval()
        with torch.no_grad():
            # This is a placeholder - implement actual sampling if needed
            logging.info(f"Sample logging to W&B at step {step}")
        model.train()
        
    except Exception as e:
        logging.warning(f"Sample logging failed: {e}")
        model.train()


def create_experiment_directory(args):
    """Create experiment directory and save config"""
    exp_dir = os.path.join(args.exp, args.doc)
    os.makedirs(exp_dir, exist_ok=True)
    
    # Save config
    with open(os.path.join(exp_dir, 'config.yaml'), 'w') as f:
        yaml.dump(vars(args), f, default_flow_style=False)
    
    return exp_dir
