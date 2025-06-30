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

# Import wandb - will be None if not installed
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    wandb = None
    WANDB_AVAILABLE = False

try:
    from models.fast_ddpm_3d import FastDDPM3D
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
    parser.add_argument('--log_every_n_steps', type=int, default=50, help='Log training progress every N steps')
    parser.add_argument('--use_wandb', action='store_true', help='Use Weights & Biases for logging')
    parser.add_argument('--wandb_project', type=str, default='fast-ddpm-3d-brats', help='W&B project name')
    parser.add_argument('--wandb_entity', type=str, default=None, help='W&B entity (username or team)')

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

# --- START: New functions for W&B sample logging ---

def get_diffusion_variables(betas):
    """Pre-compute diffusion variables for sampling."""
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)
    
    return {
        'alphas': alphas,
        'alphas_cumprod': alphas_cumprod,
    }

@torch.no_grad()
def log_sample_slices_to_wandb(model, batch, t_intervals, diffusion_vars, device, step):
    """
    Generates an image sample from the model using DDIM and logs 2D slices to W&B
    for an image-to-image translation task.
    """
    logging.info(f"Starting W&B sample logging at step {step}")
    model.eval()

    # Use the provided fixed batch
    inputs = batch['input'].to(device)    # [1, 4, H, W, D]
    targets = batch['target'].to(device)  # [1, H, W, D]
    targets = targets.unsqueeze(1)        # [1, 1, H, W, D]
    
    logging.info(f"Input shape: {inputs.shape}, Target shape: {targets.shape}")

    # Generate a sample using the reverse diffusion process (DDIM)
    shape = targets.shape
    img = torch.randn(shape, device=device) # Start with pure noise

    # Create the reverse timestep sequence from the training schedule
    seq = t_intervals.cpu().numpy()
    seq_next = [-1] + list(seq[:-1])
    
    logging.info(f"Diffusion sequence length: {len(seq)}")

    # Reverse process
    for i, j in tqdm(reversed(list(zip(seq, seq_next))), desc="Generating sample for W&B", total=len(seq), leave=False):
        t = (torch.ones(shape[0]) * i).to(device).long()
        
        # Create model input by replacing first channel with noisy image
        model_input = inputs.clone()
        model_input[:, 0:1] = img
        
        # Predict noise using the model
        et = model(model_input, t.float())

        # DDIM update rule - convert i,j to int for proper indexing
        alpha_cumprod_t = diffusion_vars['alphas_cumprod'][int(i)]
        if j >= 0:
            alpha_cumprod_next = diffusion_vars['alphas_cumprod'][int(j)]
        else:
            alpha_cumprod_next = torch.tensor(1.0, device=device)
        
        # Reshape for broadcasting
        alpha_cumprod_t = alpha_cumprod_t.view(1, 1, 1, 1, 1)
        alpha_cumprod_next = alpha_cumprod_next.view(1, 1, 1, 1, 1)
        
        # Predicted x0
        x0_t = (img - et * (1 - alpha_cumprod_t).sqrt()) / alpha_cumprod_t.sqrt()
        
        # Direction pointing to x_t
        c1 = (1 - alpha_cumprod_next).sqrt()
        
        # Update
        img = alpha_cumprod_next.sqrt() * x0_t + c1 * et
    
    # Clamp the final sample to a valid image range
    generated_sample = img.clamp(0.0, 1.0)
    logging.info(f"Generated sample shape: {generated_sample.shape}, range: [{generated_sample.min():.3f}, {generated_sample.max():.3f}]")

    # --- Prepare slices for logging ---
    # We'll take the middle slice from the depth axis
    slice_idx = generated_sample.shape[-1] // 2

    # Get slices (convert to numpy for visualization)
    # Using FLAIR as the example input modality for context
    input_flair_slice = inputs[0, 3, :, :, slice_idx].cpu().numpy()

    # Ground truth and generated image sample (no longer treated as masks)
    target_image_slice = targets[0, 0, :, :, slice_idx].cpu().numpy()
    generated_image_slice = generated_sample[0, 0, :, :, slice_idx].cpu().numpy()

    # Normalize slices to 0-1 range for better visualization
    def normalize_slice(slice_array):
        slice_min, slice_max = slice_array.min(), slice_array.max()
        if slice_max > slice_min:
            return (slice_array - slice_min) / (slice_max - slice_min)
        return slice_array

    input_flair_slice = normalize_slice(input_flair_slice)
    target_image_slice = normalize_slice(target_image_slice)
    generated_image_slice = normalize_slice(generated_image_slice)

    # Log to W&B as a gallery of images for side-by-side comparison
    if not WANDB_AVAILABLE:
        logging.error("W&B is not available, cannot log images")
        return
    
    log_dict = {
        "samples/image_comparison": [
            wandb.Image(input_flair_slice, caption=f"Input: FLAIR (Step {step})"),
            wandb.Image(generated_image_slice, caption=f"Model Output (Step {step})"),
            wandb.Image(target_image_slice, caption=f"Ground Truth (Step {step})")
        ]
    }

    wandb.log(log_dict, step=step)
    logging.info(f"Successfully logged sample image slices to W&B at step {step}")
    logging.info(f"Slice shapes - Input: {input_flair_slice.shape}, Generated: {generated_image_slice.shape}, Target: {target_image_slice.shape}")
    
    model.train() # Set model back to training mode

# --- END: New functions for W&B sample logging ---

def validate_model(model, val_loader, device, betas, timesteps):
    """Validation loop"""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating", leave=False):
            inputs = batch['input'].to(device)
            targets = batch['target'].to(device).unsqueeze(1)
            
            n = inputs.size(0)
            # Use full timestep range for validation loss for consistency
            t = torch.randint(0, len(betas), (n,), device=device)
            e = torch.randn_like(targets)
            
            loss = sg_noise_estimation_loss(model, inputs, targets, t, e, betas)
            total_loss += loss.item()
            num_batches += 1
    
    model.train()
    return total_loss / num_batches if num_batches > 0 else 0

def setup_wandb(args, config):
    """Initialize W&B if requested"""
    if args.use_wandb and WANDB_AVAILABLE:
        # Check if we should use W&B from config too
        use_wandb = args.use_wandb
        if hasattr(config, 'logging') and hasattr(config.logging, 'use_wandb'):
            use_wandb = use_wandb or config.logging.use_wandb
        
        if use_wandb:
            # Get project name from config if available
            project_name = args.wandb_project
            if hasattr(config, 'logging') and hasattr(config.logging, 'project_name'):
                project_name = config.logging.project_name
            
            # Initialize W&B
            wandb.init(
                project=project_name,
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
    
    # Override W&B settings from command line if provided
    if args.use_wandb:
        if not hasattr(config, 'logging'):
            config.logging = argparse.Namespace()
        config.logging.use_wandb = True
    
    # Setup W&B
    use_wandb = setup_wandb(args, config)
    if use_wandb:
        logging.info("✅ W&B initialized successfully")
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
        
        val_dataset = BraTS3DUnifiedDataset(
            data_root=args.data_root,
            phase='train',  # Use same for now, can change to 'val' if available
            volume_size=tuple(config.data.volume_size)
        )
        
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
    
    # --- START: Modified section for W&B sample logging ---
    # Get a fixed batch from the validation set for consistent logging
    fixed_val_batch = None
    if use_wandb and len(val_loader) > 0:
        try:
            fixed_val_batch = next(iter(val_loader))
            logging.info("Grabbed a fixed validation batch for W&B logging.")
        except StopIteration:
            fixed_val_batch = None
            logging.warning("Validation loader is empty, cannot log sample slices.")
    # --- END: Modified section ---

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
    
    t_intervals = get_timestep_schedule(args.scheduler_type, args.timesteps, num_timesteps)
    
    # --- START: Added diffusion vars for W&B logging ---
    diffusion_vars = get_diffusion_variables(betas) if use_wandb else None
    # --- END: Added diffusion vars ---

    scaler = GradScaler()
    
    start_epoch = 0
    start_step = 0
    if args.resume and args.resume_path and os.path.exists(args.resume_path):
        logging.info(f"Resuming from checkpoint: {args.resume_path}")
        start_epoch, start_step, _ = load_checkpoint(
            args.resume_path, model, optimizer, scaler, device
        )
        logging.info(f"Resumed from epoch {start_epoch}, step {start_step}")
    
    logging.info("Starting 3D Fast-DDPM training...")
    logging.info(f"Scheduler type: {args.scheduler_type}, Timesteps: {args.timesteps}")
    logging.info(f"Volume size: {config.data.volume_size}")
    logging.info(f"Batch size: {config.training.batch_size}")
    
    step = start_step
    best_val_loss = float('inf')
    
    # Log initial sample to W&B if enabled
    if use_wandb and fixed_val_batch and diffusion_vars:
        logging.info("Logging initial sample to W&B...")
        try:
            log_sample_slices_to_wandb(
                model.module if isinstance(model, nn.DataParallel) else model,
                fixed_val_batch,
                t_intervals,
                diffusion_vars,
                device,
                step=0
            )
            logging.info("Successfully logged initial sample to W&B")
        except Exception as e:
            logging.error(f"Failed to log initial sample to W&B: {e}")
    
    for epoch in range(start_epoch, config.training.epochs):
        model.train()
        epoch_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config.training.epochs}')
        for batch_idx, batch in enumerate(pbar):
            try:
                optimizer.zero_grad()
                
                inputs = batch['input'].to(device)
                targets = batch['target'].to(device).unsqueeze(1)
                
                n = inputs.size(0)
                step += 1
                
                idx_1 = torch.randint(0, len(t_intervals), size=(n // 2 + 1,))
                idx_2 = len(t_intervals) - idx_1 - 1
                idx = torch.cat([idx_1, idx_2], dim=0)[:n]
                t = t_intervals[idx].to(device)
                
                e = torch.randn_like(targets)
                
                with autocast():
                    loss = sg_noise_estimation_loss(model, inputs, targets, t, e, betas)
                
                scaler.scale(loss).backward()
                
                if hasattr(config.training, 'gradient_clip'):
                    scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.gradient_clip)
                else:
                    grad_norm = 0.0
                
                scaler.step(optimizer)
                scaler.update()
                
                epoch_loss += loss.item()
                
                pbar.set_postfix({
                    'loss': f'{loss.item():.6f}',
                    'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'
                })
                
                if use_wandb and step % 10 == 0:
                    log_dict = {
                        'train/loss': loss.item(),
                        'train/learning_rate': optimizer.param_groups[0]["lr"],
                        'train/epoch': epoch,
                        'train/step': step,
                    }
                    if grad_norm > 0:
                        log_dict['train/grad_norm'] = grad_norm
                    if torch.cuda.is_available():
                        log_dict['system/gpu_memory_allocated'] = torch.cuda.memory_allocated() / 1e9
                        log_dict['system/gpu_memory_reserved'] = torch.cuda.memory_reserved() / 1e9
                    wandb.log(log_dict, step=step)
                
                if (batch_idx + 1) % args.log_every_n_steps == 0:
                    logging.info(f'Epoch {epoch+1}/{config.training.epochs}, '
                                 f'Batch {batch_idx+1}/{len(train_loader)} - '
                                 f'Step Loss: {loss.item():.6f}, '
                                 f'LR: {optimizer.param_groups[0]["lr"]:.2e}')
                
                if step % getattr(config.training, 'save_every', 1000) == 0:
                    save_checkpoint(
                        model, optimizer, scaler, step, epoch, loss.item(), 
                        config, os.path.join(log_dir, f'ckpt_{step}.pth')
                    )
                
                if step % getattr(config.training, 'validate_every', 100) == 0:
                    val_loss = validate_model(model, val_loader, device, betas, args.timesteps)
                    logging.info(f'Step {step} - Val Loss: {val_loss:.6f}')
                    
                    if use_wandb:
                        wandb.log({
                            'val/loss': val_loss,
                            'val/epoch': epoch,
                        }, step=step)
                        
                        # --- START: Call to the W&B sample logging function ---
                        if fixed_val_batch and diffusion_vars:
                            log_sample_slices_to_wandb(
                                # unwrap model from DataParallel if used
                                model.module if isinstance(model, nn.DataParallel) else model,
                                fixed_val_batch,
                                t_intervals,
                                diffusion_vars,
                                device,
                                step
                            )
                        # --- END: Call to the W&B sample logging function ---
                    
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        save_checkpoint(
                            model, optimizer, scaler, step, epoch, val_loss,
                            config, os.path.join(log_dir, 'best_model.pth')
                        )
                        logging.info(f'New best validation loss: {val_loss:.6f}')
                        
                        if use_wandb:
                            wandb.run.summary["best_val_loss"] = val_loss
                            wandb.run.summary["best_val_step"] = step
            
            except Exception as e:
                logging.error(f"Error in training step {step}: {e}")
                if args.debug:
                    raise
                continue
        
        # End of epoch
        scheduler.step()
        avg_loss = epoch_loss / len(train_loader) if len(train_loader) > 0 else 0
        logging.info(f'Epoch {epoch+1} - Average Loss: {avg_loss:.6f}, LR: {optimizer.param_groups[0]["lr"]:.2e}')
        
        if use_wandb:
            wandb.log({
                'epoch/avg_loss': avg_loss,
                'epoch/learning_rate': optimizer.param_groups[0]["lr"],
                'epoch/epoch': epoch + 1,
            }, step=step)
        
        save_checkpoint(
            model, optimizer, scaler, step, epoch, avg_loss,
            config, os.path.join(log_dir, 'ckpt.pth')
        )
    
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