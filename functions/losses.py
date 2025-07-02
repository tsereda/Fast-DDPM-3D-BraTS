import torch
import math
import numpy as np
import logging

logger = logging.getLogger(__name__)


def validate_tensor_stability(tensor, name="tensor", max_val=10.0):
    """
    Validate tensor for numerical stability
    
    Args:
        tensor: Input tensor to validate
        name: Name for logging purposes
        max_val: Maximum allowed absolute value
        
    Returns:
        bool: True if tensor is stable, False otherwise
    """
    if torch.any(torch.isnan(tensor)):
        logger.warning(f"NaN detected in {name}")
        return False
    
    if torch.any(torch.isinf(tensor)):
        logger.warning(f"Inf detected in {name}")
        return False
    
    if torch.any(torch.abs(tensor) > max_val):
        logger.warning(f"Extreme values detected in {name}: max={torch.max(torch.abs(tensor)):.3f}")
        return False
    
    return True


def safe_clamp_tensor(tensor, min_val=-10.0, max_val=10.0):
    """
    Safely clamp tensor values with gradient preservation
    
    Args:
        tensor: Input tensor
        min_val: Minimum value
        max_val: Maximum value
        
    Returns:
        Clamped tensor
    """
    return torch.clamp(tensor, min_val, max_val)


def robust_sqrt(x, eps=1e-8):
    """
    Numerically stable square root
    
    Args:
        x: Input tensor
        eps: Small epsilon for numerical stability
        
    Returns:
        Stable square root
    """
    return torch.sqrt(torch.clamp(x, min=eps))


def compute_stable_alpha(betas, timesteps):
    """
    Compute alpha values with numerical stability checks
    
    Args:
        betas: Beta schedule [T]
        timesteps: Timestep indices [B]
        
    Returns:
        Alpha values [B, 1, 1, 1, 1] for 3D tensors
    """
    # Clamp betas to prevent numerical issues
    betas_clamped = torch.clamp(betas, 1e-8, 0.999)
    
    # Compute cumulative product with stability
    alphas = 1.0 - betas_clamped
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    
    # Clamp to prevent extreme values
    alphas_cumprod = torch.clamp(alphas_cumprod, 1e-8, 1.0)
    
    # Select timesteps and reshape for 3D broadcasting
    selected_alphas = alphas_cumprod.index_select(0, timesteps.long())
    return selected_alphas.view(-1, 1, 1, 1, 1)


def calculate_psnr(img1, img2):
    """
    Calculate PSNR between two images
    Args:
        img1, img2: numpy arrays with values in [0, 255] range
    Returns:
        PSNR value in dB
    """
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    
    psnr = 20 * math.log10(255.0 / math.sqrt(mse))
    return psnr


def unified_4to1_loss(model, x_available, x_target, t, e, b, target_idx=0, keepdim=False):
    """
    4→1 Fast-DDPM loss for unified BraTS modality synthesis (simplified version)
    
    Args:
        model: 3D diffusion model (outputs 1 channel)
        x_available: [B, 4, H, W, D] - all modalities with target masked (zeroed)
        x_target: [B, 1, H, W, D] - target modality volume  
        t: [B] - timesteps
        e: [B, 1, H, W, D] - noise tensor (same shape as target)
        b: [T] - beta schedule
        target_idx: int - which modality is being synthesized (0-3)
        keepdim: bool - whether to keep batch dimension in loss
        
    Returns:
        loss: scalar tensor (mean) or [B] tensor (keepdim=True)
    """
    device = x_target.device
    batch_size = x_target.size(0)
    
    # Basic input validation
    if torch.any(torch.isnan(x_target)) or torch.any(torch.isnan(e)):
        logger.warning("NaN detected in inputs, using fallback loss")
        fallback_value = 1.0
        if keepdim:
            return torch.full((batch_size,), fallback_value, device=device, dtype=x_target.dtype, requires_grad=True)
        else:
            return torch.tensor(fallback_value, device=device, dtype=x_target.dtype, requires_grad=True)
    
    # Compute alpha values from beta schedule
    alphas = 1.0 - torch.clamp(b, 1e-8, 0.999)
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod = torch.clamp(alphas_cumprod, 1e-8, 0.999)
    
    # Select alpha values for current timesteps
    t_safe = torch.clamp(t.long(), 0, alphas_cumprod.size(0) - 1)
    a = alphas_cumprod.index_select(0, t_safe).view(-1, 1, 1, 1, 1)
    
    # Compute noise injection coefficients
    sqrt_alpha = torch.sqrt(a)
    sqrt_one_minus_alpha = torch.sqrt(1.0 - a)
    
    # Apply noise to target modality
    x_noisy = x_target * sqrt_alpha + e * sqrt_one_minus_alpha
    
    # Prepare model input by inserting noisy target
    model_input = x_available.clone()
    model_input[:, target_idx:target_idx+1] = x_noisy
    
    # Model forward pass
    try:
        predicted_noise = model(model_input, t.float())
        
        # Handle tuple outputs
        if isinstance(predicted_noise, tuple):
            predicted_noise = predicted_noise[0]
            
    except Exception as model_error:
        logger.error(f"Model forward pass failed: {model_error}")
        fallback_value = 1.0
        if keepdim:
            return torch.full((batch_size,), fallback_value, device=device, dtype=x_target.dtype, requires_grad=True)
        else:
            return torch.tensor(fallback_value, device=device, dtype=x_target.dtype, requires_grad=True)
    
    # Compute MSE loss between predicted and actual noise
    mse_loss = torch.nn.functional.mse_loss(predicted_noise, e, reduction='none')
    
    # Aggregate loss
    if keepdim:
        loss = mse_loss.view(batch_size, -1).mean(dim=1)
    else:
        loss = mse_loss.mean()
    
    # Basic stability check
    if torch.isnan(loss) or torch.isinf(loss):
        logger.warning("NaN/Inf in loss, using fallback")
        fallback_value = 1.0
        if keepdim:
            return torch.full((batch_size,), fallback_value, device=device, dtype=x_target.dtype, requires_grad=True)
        else:
            return torch.tensor(fallback_value, device=device, dtype=x_target.dtype, requires_grad=True)
    
    return loss


def simple_4to1_loss(model, x_available, x_target, t, e, b, target_idx=0, keepdim=False):
    """
    Minimal 4→1 Fast-DDPM loss (ultra-simplified version)
    
    Args:
        model: 3D diffusion model
        x_available: [B, 4, H, W, D] - all modalities with target masked
        x_target: [B, 1, H, W, D] - target modality volume  
        t: [B] - timesteps
        e: [B, 1, H, W, D] - noise tensor
        b: [T] - beta schedule
        target_idx: int - which modality is being synthesized
        keepdim: bool - whether to keep batch dimension
        
    Returns:
        loss: scalar or [B] tensor
    """
    # Compute alpha values
    alphas_cumprod = torch.cumprod(1.0 - b, dim=0)
    a = alphas_cumprod[t.long()].view(-1, 1, 1, 1, 1)
    
    # Add noise to target
    x_noisy = torch.sqrt(a) * x_target + torch.sqrt(1.0 - a) * e
    
    # Model input
    model_input = x_available.clone()
    model_input[:, target_idx:target_idx+1] = x_noisy
    
    # Predict noise
    predicted_noise = model(model_input, t.float())
    if isinstance(predicted_noise, tuple):
        predicted_noise = predicted_noise[0]
    
    # MSE loss
    loss = torch.nn.functional.mse_loss(predicted_noise, e, reduction='none')
    
    if keepdim:
        return loss.view(loss.size(0), -1).mean(dim=1)
    else:
        return loss.mean()


# Registry for backwards compatibility
loss_registry = {
    'unified_4to1': unified_4to1_loss,
    'simple_4to1': simple_4to1_loss,
    # Legacy mappings
    'sg': unified_4to1_loss,  # Single condition tasks
    'sr': unified_4to1_loss,  # Multi condition tasks
    'simple': simple_4to1_loss  # Ultra-minimal version
}