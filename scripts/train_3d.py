import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import logging
import numpy as np
import time

# Add the current directory to path to import modules
sys.path.append('.')
sys.path.append('..')

# Import wandb - will be None if not installed
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    wandb = None
    WANDB_AVAILABLE = False

# Import required modules
try:
    from models.fast_ddpm_3d import FastDDPM3D
    from functions.losses import unified_4to1_loss, calculate_psnr
    from functions.denoising_3d import unified_4to1_generalized_steps_3d
    from data.brain_3d_unified import BraTS3DUnifiedDataset
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
    parser.add_argument('--log_every_n_steps', type=int, default=50, help='Log training progress every N steps')
    parser.add_argument('--use_wandb', action='store_true', help='Use Weights & Biases for logging')
    parser.add_argument('--wandb_project', type=str, default='fast-ddpm-3d-brats', help='W&B project name')
    parser.add_argument('--wandb_entity', type=str, default=None, help='W&B entity (username or team)')
    parser.add_argument('--sample_every', type=int, default=2000, help='Generate samples for W&B every N steps')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4, help='Gradient accumulation steps')

    return parser.parse_args()


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


def validate_model(model, val_loader, device, betas, t_intervals):
    """Validation loop"""
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


def load_config(config_path):
    """Load configuration file"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return dict2namespace(config)


def main():
    args = parse_args()
    
    # Setup device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Setup logging
    log_dir = os.path.join(args.exp, 'logs', args.doc)
    setup_logging(log_dir, args)
    
    # Load config
    try:
        config = load_config(args.config)
    except FileNotFoundError:
        logging.error(f"Config file not found: {args.config}")
        sys.exit(1)
    
    config.device = device
    
    # Debug: Log config training values
    logging.info(f"Config training section: {vars(config.training) if hasattr(config, 'training') else 'No training section'}")
    if hasattr(config, 'training'):
        logging.info(f"save_every in config: {getattr(config.training, 'save_every', 'NOT FOUND')}")
        logging.info(f"validate_every in config: {getattr(config.training, 'validate_every', 'NOT FOUND')}")
    
    # Gradient accumulation settings
    gradient_accumulation_steps = args.gradient_accumulation_steps
    effective_batch_size = config.training.batch_size * gradient_accumulation_steps
    
    logging.info(f"Gradient accumulation steps: {gradient_accumulation_steps}")
    logging.info(f"Effective batch size: {effective_batch_size}")
    
    # Setup W&B
    use_wandb = setup_wandb(args, config)
    if use_wandb:
        logging.info("✅ W&B initialized successfully")
        wandb.config.update({
            'gradient_accumulation_steps': gradient_accumulation_steps,
            'effective_batch_size': effective_batch_size
        })
    elif args.use_wandb and not WANDB_AVAILABLE:
        logging.warning("⚠️ W&B requested but not installed. Install with: pip install wandb")
    
    # Create experiment directory
    exp_dir = os.path.join(args.exp, args.doc)
    os.makedirs(exp_dir, exist_ok=True)
    
    # Save config
    with open(os.path.join(exp_dir, 'config.yaml'), 'w') as f:
        yaml.dump(vars(args), f, default_flow_style=False)
    
    # Dataset and DataLoader
    logging.info("Setting up datasets...")
    
    # Use provided volume size or default
    volume_size = tuple(config.data.volume_size) if hasattr(config.data, 'volume_size') else (80, 80, 80)
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
            from torch.utils.data import Subset
            # Use larger debug dataset: ~10% of total for meaningful epochs
            debug_train_size = min(125, len(train_dataset))  # ~125 cases = ~125 batches per epoch
            debug_val_size = min(25, len(val_dataset))       # ~25 cases for validation
            train_indices = list(range(debug_train_size))
            val_indices = list(range(debug_val_size))
            train_dataset = Subset(train_dataset, train_indices)
            val_dataset = Subset(val_dataset, val_indices)
            logging.info(f"Debug mode: using {debug_train_size} train samples, {debug_val_size} val samples")
        
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
        batch_size=1,
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
        model = FastDDPM3D(config).to(device)
        if torch.cuda.device_count() > 1 and not args.debug:
            model = torch.nn.DataParallel(model)
            logging.info(f"Using {torch.cuda.device_count()} GPUs")
        
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
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=config.training.epochs,
        eta_min=config.training.learning_rate * 0.1
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
    
    # Get fixed batch for sample logging
    fixed_val_batch = None
    if use_wandb and len(val_loader) > 0:
        try:
            fixed_val_batch = next(iter(val_loader))
            logging.info("Got fixed validation batch for sample logging")
        except StopIteration:
            fixed_val_batch = None
            logging.warning("Validation loader is empty")
    
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
    
    logging.info("Starting training...")
    logging.info(f"Scheduler type: {args.scheduler_type}, Timesteps: {args.timesteps}")
    logging.info(f"Volume size: {volume_size}")
    logging.info(f"Batch size: {config.training.batch_size}")
    logging.info(f"Gradient accumulation steps: {gradient_accumulation_steps}")
    logging.info(f"Effective batch size: {effective_batch_size}")
    
    global_step = start_step  # This tracks actual optimizer steps
    best_val_loss = float('inf')
    
    for epoch in range(start_epoch, config.training.epochs):
        model.train()
        epoch_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config.training.epochs}')
        
        # Initialize accumulation variables
        accumulated_loss = 0.0
        accumulation_steps = 0
        
        for batch_idx, batch in enumerate(pbar):
            try:
                step_start_time = time.time()
                
                # Zero gradients at the start of each accumulation cycle
                if accumulation_steps == 0:
                    optimizer.zero_grad()
                
                inputs = batch['input'].to(device)
                targets = batch['target'].to(device).unsqueeze(1)
                target_idx = batch['target_idx'][0].item()
                
                n = inputs.size(0)
                
                # Fast-DDPM antithetic sampling
                idx = torch.randint(0, len(t_intervals), size=(n,))
                t = t_intervals[idx].to(device)
                
                e = torch.randn_like(targets)
                
                # with autocast():
                #     loss = unified_4to1_loss(model, inputs, targets, t, e, b=betas, target_idx=target_idx)
                    # Don't scale loss here - let gradient accumulation handle it naturally
                loss = unified_4to1_loss(model, inputs, targets, t, e, b=betas, target_idx=target_idx)
                
                # Check for NaN/Inf loss
                if torch.isnan(loss) or torch.isinf(loss):
                    logging.error(f"Invalid loss detected: {loss.item()}")
                    logging.error(f"Input stats: min={inputs.min():.3f}, max={inputs.max():.3f}")
                    logging.error(f"Target stats: min={targets.min():.3f}, max={targets.max():.3f}")
                    if args.debug:
                        raise ValueError("NaN/Inf loss - stopping training")
                    continue  # Skip this batch
                
                # Additional safety check for extreme loss values
                loss_value = loss.item()
                if loss_value > 1000.0:
                    logging.warning(f"Very large loss detected: {loss_value:.6f} - possible scaling issue")
                elif loss_value < 1e-8:
                    logging.warning(f"Very small loss detected: {loss_value:.6e} - possible underflow")
                
                scaler.scale(loss).backward()
                accumulated_loss += loss.item()
                accumulation_steps += 1
                
                # Step optimizer when we've accumulated enough gradients
                if accumulation_steps >= gradient_accumulation_steps:
                    scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), 
                        max_norm=getattr(config.training, 'gradient_clip', float('inf'))
                    )
                    
                    scaler.step(optimizer)
                    scaler.update()
                    
                    # Monitor gradient scaler health
                    if global_step % 100 == 0:  # Check every 100 steps
                        monitor_scaler_health(scaler, global_step)
                    
                    # Move scheduler step here to avoid the warning
                    # scheduler.step()  # Comment out - we'll do this at end of epoch instead
                    
                    # Now we've completed one effective training step
                    global_step += 1
                    
                    # Calculate average loss over accumulation steps
                    average_loss = accumulated_loss / gradient_accumulation_steps
                    epoch_loss += average_loss
                    
                    step_time = time.time() - step_start_time
                    
                    pbar.set_postfix({
                        'loss': f'{average_loss:.6f}',
                        'lr': f'{optimizer.param_groups[0]["lr"]:.2e}',
                        'target': f'{target_idx}',
                        'step': global_step,
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
                    if global_step % args.log_every_n_steps == 0:
                        logging.info(f'Epoch {epoch+1}/{config.training.epochs}, '
                                     f'Step {global_step} - '
                                     f'Loss: {average_loss:.6f}, '
                                     f'Target: {target_idx}, '
                                     f'LR: {optimizer.param_groups[0]["lr"]:.2e}, '
                                     f'Time: {step_time:.2f}s')
                    
                    # Save and validate periodically
                    save_every = getattr(config.training, 'save_every', 2000)
                    validate_every = getattr(config.training, 'validate_every', 1000)
                    
                    if global_step % save_every == 0:
                        save_checkpoint(
                            model, optimizer, scaler, global_step, epoch, average_loss, 
                            config, os.path.join(log_dir, f'ckpt_{global_step}.pth')
                        )
                        logging.info(f'Saved checkpoint at step {global_step} (save_every={save_every})')
                    
                    if global_step % validate_every == 0:
                        val_loss = validate_model(model, val_loader, device, betas, t_intervals)
                        logging.info(f'Step {global_step} - Val Loss: {val_loss:.6f}')
                        
                        if use_wandb:
                            wandb.log({
                                'val/loss': val_loss,
                                'val/epoch': epoch,
                            }, step=global_step)
                        
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            save_checkpoint(
                                model, optimizer, scaler, global_step, epoch, val_loss,
                                config, os.path.join(log_dir, 'best_model.pth')
                            )
                            logging.info(f'New best validation loss: {val_loss:.6f}')
                            
                            if use_wandb:
                                wandb.run.summary["best_val_loss"] = val_loss
                                wandb.run.summary["best_val_step"] = global_step
                    
                    # Sample logging
                    if use_wandb and fixed_val_batch and global_step % args.sample_every == 0:
                        log_samples_to_wandb(
                            model.module if isinstance(model, nn.DataParallel) else model,
                            fixed_val_batch,
                            t_intervals,
                            betas,
                            device,
                            global_step
                        )
                        # Log samples to W&B with all modalities in one compact view
                        if fixed_val_batch is not None:
                            log_samples_to_wandb(
                                model.module if isinstance(model, nn.DataParallel) else model,
                                fixed_val_batch,
                                t_intervals,
                                betas,
                                device,
                                global_step
                            )
                
                # Clear GPU memory cache periodically
                if (batch_idx + 1) % 10 == 0:
                    torch.cuda.empty_cache()
            
            except RuntimeError as e:
                if "out of memory" in str(e):
                    logging.error("GPU OOM - clearing cache and skipping batch")
                    torch.cuda.empty_cache()
                    # Reset accumulation if OOM occurred
                    accumulated_loss = 0.0
                    accumulation_steps = 0
                    continue
                else:
                    logging.error(f"Runtime error in training step: {e}")
                    if args.debug:
                        raise
                    continue
            except Exception as e:
                logging.error(f"Error in training step: {e}")
                if args.debug:
                    raise
                continue
        
        # End of epoch
        scheduler.step()
        avg_loss = epoch_loss / max(1, global_step - start_step) if global_step > start_step else 0
        logging.info(f'Epoch {epoch+1} - Average Loss: {avg_loss:.6f}, LR: {optimizer.param_groups[0]["lr"]:.2e}')
        
        if use_wandb:
            wandb.log({
                'epoch/avg_loss': avg_loss,
                'epoch/learning_rate': optimizer.param_groups[0]["lr"],
                'epoch/epoch': epoch + 1,
            }, step=global_step)
        
        # Only save checkpoint at end of epoch if it's a multiple of 10 epochs (much less frequent)
        if (epoch + 1) % 10 == 0:
            save_checkpoint(
                model, optimizer, scaler, global_step, epoch, avg_loss,
                config, os.path.join(log_dir, f'ckpt_epoch_{epoch+1}.pth')
            )
            logging.info(f'Saved epoch checkpoint at epoch {epoch+1}')
    
    logging.info("Training completed!")
    logging.info(f"Best validation loss: {best_val_loss:.6f}")
    logging.info(f"Final model saved at: {os.path.join(log_dir, 'ckpt.pth')}")
    
    if use_wandb:
        wandb.finish()


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