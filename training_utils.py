"""
Training utilities for 3D Fast-DDPM
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
from distributed_utils import is_main_process

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
            logging.FileHandler(os.path.join(log_dir, 'train.log')),
            logging.StreamHandler()
        ]
    )
    
    # Log arguments
    logging.info("Training arguments:")
    for arg, value in vars(args).items():
        logging.info(f"  {arg}: {value}")


def setup_wandb(args, config):
    """Initialize W&B if requested"""
    if args.use_wandb and WANDB_AVAILABLE:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.doc,
            config={
                **vars(args),
                'model_config': config.model.__dict__ if hasattr(config, 'model') else {},
                'data_config': config.data.__dict__ if hasattr(config, 'data') else {},
                'training_config': config.training.__dict__ if hasattr(config, 'training') else {},
                'diffusion_config': config.diffusion.__dict__ if hasattr(config, 'diffusion') else {},
            }
        )
        return True
    return False


def save_checkpoint(model, optimizer, scaler, step, epoch, loss, config, path, rank=0):
    """Save training checkpoint (only on main process for distributed training)"""
    if is_main_process(rank):
        # Extract the actual model state dict (handle DDP/DataParallel wrapper)
        if hasattr(model, 'module'):
            model_state_dict = model.module.state_dict()
        else:
            model_state_dict = model.state_dict()
            
        checkpoint = {
            'epoch': epoch,
            'step': step,
            'model_state_dict': model_state_dict,
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


def validate_model(model, val_loader, device, betas, t_intervals):
    """Validation loop"""
    from functions.losses import unified_4to1_loss
    
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating", leave=False):
            inputs = batch['input'].to(device)
            targets = batch['target'].to(device).unsqueeze(1)
            target_idx = batch['target_idx'][0].item()
            
            n = inputs.size(0)
            # Use Fast-DDPM timestep schedule for validation too
            idx = torch.randint(0, len(t_intervals), (n,))
            t = t_intervals[idx].to(device)
            e = torch.randn_like(targets)
            
            loss = unified_4to1_loss(model, inputs, targets, t, e, b=betas, target_idx=target_idx)
            total_loss += loss.item()
            num_batches += 1
    
    model.train()
    return total_loss / num_batches if num_batches > 0 else 0


def monitor_scaler_health(scaler, step):
    """Monitor gradient scaler health and adjust if needed"""
    scale = scaler.get_scale()
    
    # Log warnings for problematic scaling
    if scale < 1.0:
        logging.warning(f"Low gradient scale detected: {scale:.2e} at step {step}")
    elif scale > 65536.0:  # 2^16
        logging.warning(f"High gradient scale detected: {scale:.2e} at step {step}")
    
    # Force scale adjustment if it's too extreme
    if scale < 0.5:
        logging.info(f"Forcing gradient scale increase from {scale:.2e}")
        scaler.update(new_scale=2.0)
    elif scale > 131072.0:  # 2^17
        logging.info(f"Forcing gradient scale decrease from {scale:.2e}")
        scaler.update(new_scale=16384.0)  # 2^14


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    """Beta schedule for diffusion - same as Fast-DDPM"""
    if beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "cosine":
        steps = num_diffusion_timesteps + 1
        s = 0.008
        x = np.linspace(0, num_diffusion_timesteps, steps)
        alphas_cumprod = np.cos(((x / num_diffusion_timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        betas = np.clip(betas, 0, 0.999)
    else:
        raise NotImplementedError(f"Beta schedule {beta_schedule} not implemented")
    
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


def get_timestep_schedule(scheduler_type, timesteps, num_timesteps):
    """Get timestep schedule for Fast-DDPM"""
    if scheduler_type == 'uniform':
        skip = num_timesteps // timesteps
        t_intervals = torch.arange(0, num_timesteps, skip)
    elif scheduler_type == 'non-uniform':
        if timesteps == 10:
            t_intervals = torch.tensor([0, 199, 399, 599, 699, 799, 849, 899, 949, 999])
        else:
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


@torch.no_grad()
def log_samples_to_wandb(model, batch, t_intervals, betas, device, step):
    """Simple W&B logging with all modalities in one compact view"""
    if not WANDB_AVAILABLE:
        return
        
    try:
        import matplotlib.pyplot as plt
        
        model.eval()
        
        # Get batch data
        inputs = batch['input'].to(device)    # [1, 4, H, W, D]
        targets = batch['target'].to(device)  # [1, H, W, D]
        targets = targets.unsqueeze(1)        # [1, 1, H, W, D]
        target_idx = batch['target_idx'][0].item()
        
        # Generate sample using simplified reverse process
        shape = targets.shape
        img = torch.randn(shape, device=device)
        
        # Use the unified 4->1 sampling approach
        model_input = inputs.clone()
        model_input[:, target_idx:target_idx+1] = img
        
        # Simple reverse diffusion (just a few steps for speed)
        seq = t_intervals[-5:].cpu().numpy()  # Use last 5 timesteps for quick sampling
        
        for i in reversed(seq):
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            model_input[:, target_idx:target_idx+1] = img
            
            # Predict noise
            et = model(model_input, t.float())
            if isinstance(et, tuple):
                et = et[0]
            
            # Simple DDIM update (eta=0)
            alpha_cumprod_t = (1 - betas).cumprod(dim=0)[i].view(1, 1, 1, 1, 1)
            x0_t = (img - et * (1 - alpha_cumprod_t).sqrt()) / alpha_cumprod_t.sqrt()
            img = x0_t.clamp(-1.0, 1.0)
        
        # Get middle slice for all modalities
        slice_idx = img.shape[-1] // 2
        modality_names = ['T1n', 'T1c', 'T2w', 'T2f']
        
        # Collect all slices: 4 inputs + generated + target
        all_slices = []
        titles = []
        
        # Input modalities
        for i in range(4):
            slice_data = (inputs[0, i, :, :, slice_idx].cpu().numpy() + 1) / 2
            all_slices.append(slice_data)
            marker = " ⭐" if i == target_idx else ""
            titles.append(f'{modality_names[i]}{marker}')
        
        # Generated
        generated_slice = (img[0, 0, :, :, slice_idx].cpu().numpy() + 1) / 2
        all_slices.append(generated_slice)
        titles.append(f'Generated')
        
        # Target
        target_slice = (targets[0, 0, :, :, slice_idx].cpu().numpy() + 1) / 2
        all_slices.append(target_slice)
        titles.append(f'Ground Truth')
        
        # Create single row figure with all 6 images
        fig, axes = plt.subplots(1, 6, figsize=(18, 3))
        fig.suptitle(f'Step {step} | Target: {modality_names[target_idx]}', fontsize=14)
        
        # Plot all slices
        for i, (slice_data, title) in enumerate(zip(all_slices, titles)):
            axes[i].imshow(slice_data, cmap='gray', vmin=0, vmax=1)
            axes[i].set_title(title, fontsize=10)
            axes[i].axis('off')
        
        # Use subplots_adjust instead of tight_layout to avoid warning
        plt.subplots_adjust(top=0.8, bottom=0.1, left=0.05, right=0.95, wspace=0.1)
        
        # Log to wandb as a single image
        wandb.log({
            "samples": wandb.Image(fig),
            "step": step,
            "target_modality": modality_names[target_idx]
        }, step=step)
        
        plt.close(fig)
        model.train()
        
    except Exception as e:
        logging.warning(f"Sample logging failed: {e}")
        import traceback
        logging.warning(f"Traceback: {traceback.format_exc()}")
        model.train()


def create_experiment_directory(args, rank):
    """Create experiment directory and save config (only on main process)"""
    if is_main_process(rank):
        exp_dir = os.path.join(args.exp, args.doc)
        os.makedirs(exp_dir, exist_ok=True)
        
        # Save config
        with open(os.path.join(exp_dir, 'config.yaml'), 'w') as f:
            yaml.dump(vars(args), f, default_flow_style=False)
        
        return exp_dir
    return None


def setup_model_for_training(model, args, rank, world_size):
    """Setup model for distributed/multi-GPU training"""
    from torch.nn.parallel import DistributedDataParallel as DDP
    
    if args.distributed:
        # Use DistributedDataParallel for distributed training
        model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)
        if is_main_process(rank):
            logging.info(f"Using DistributedDataParallel with {world_size} GPUs")
    elif args.multi_gpu and torch.cuda.device_count() > 1 and not args.debug:
        # Use DataParallel for single-node multi-GPU
        model = torch.nn.DataParallel(model)
        if is_main_process(rank):
            logging.info(f"Using DataParallel with {torch.cuda.device_count()} GPUs")
    
    return model


def get_fixed_validation_batch(val_loader):
    """Get a fixed validation batch for consistent sample logging"""
    try:
        fixed_val_batch = next(iter(val_loader))
        logging.info("Got fixed validation batch for sample logging")
        return fixed_val_batch
    except StopIteration:
        logging.warning("Validation loader is empty")
        return None