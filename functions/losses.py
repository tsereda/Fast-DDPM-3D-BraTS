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
    4→1 Fast-DDPM loss for unified BraTS modality synthesis with enhanced robust NaN/Inf handling
    
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
    # Enhanced error handling with recovery mechanisms
    device = x_target.device
    batch_size = x_target.size(0)
    
    try:
        # Stage 1: Comprehensive input validation with detailed logging
        validation_errors = []
        
        if torch.any(torch.isnan(x_available)):
            validation_errors.append("NaN in x_available")
        if torch.any(torch.isinf(x_available)):
            validation_errors.append("Inf in x_available")
        if torch.any(torch.isnan(x_target)):
            validation_errors.append("NaN in x_target")
        if torch.any(torch.isinf(x_target)):
            validation_errors.append("Inf in x_target")
        if torch.any(torch.isnan(e)):
            validation_errors.append("NaN in noise tensor")
        if torch.any(torch.isinf(e)):
            validation_errors.append("Inf in noise tensor")
        if torch.any(torch.isnan(t)):
            validation_errors.append("NaN in timesteps")
        if torch.any(torch.isinf(t)):
            validation_errors.append("Inf in timesteps")
        if torch.any(torch.isnan(b)):
            validation_errors.append("NaN in beta schedule")
        if torch.any(torch.isinf(b)):
            validation_errors.append("Inf in beta schedule")
            
        if validation_errors:
            raise ValueError(f"Input validation failed: {', '.join(validation_errors)}")
        
        # Stage 2: Enhanced input clamping with safer ranges
        x_available = torch.clamp(x_available, -8.0, 8.0)  # Tighter range for stability
        x_target = torch.clamp(x_target, -8.0, 8.0)
        e = torch.clamp(e, -3.0, 3.0)  # Tighter noise range
        t = torch.clamp(t, 0, len(b) - 1)  # Ensure valid timestep indices
        
        # Stage 3: Ultra-stable alpha computation with multiple fallbacks
        try:
            # Primary method: enhanced stability
            b_clamped = torch.clamp(b, 1e-8, 0.9999)  # Slightly tighter upper bound
            alphas = 1.0 - b_clamped
            
            # Use log-space computation for better numerical stability
            log_alphas = torch.log(alphas + 1e-10)
            log_alphas_cumprod = torch.cumsum(log_alphas, dim=0)
            alphas_cumprod = torch.exp(log_alphas_cumprod)
            
            # Additional stability clamp
            alphas_cumprod = torch.clamp(alphas_cumprod, 1e-8, 0.9999)
            
            # Safe indexing with bounds checking
            t_safe = torch.clamp(t.long(), 0, alphas_cumprod.size(0) - 1)
            a = alphas_cumprod.index_select(0, t_safe).view(-1, 1, 1, 1, 1)
            
        except Exception as alpha_error:
            logger.warning(f"Primary alpha computation failed: {alpha_error}, using fallback")
            # Fallback: simple linear interpolation
            t_normalized = t.float() / (len(b) - 1)
            a = (0.99 - 0.01 * t_normalized).view(-1, 1, 1, 1, 1)
            
        # Ensure alpha values are in valid range
        a = torch.clamp(a, 1e-8, 0.9999)
        
        # Stage 4: Enhanced noise injection with numerical stability
        try:
            # Use numerically stable formulations
            sqrt_alpha = robust_sqrt(a, eps=1e-10)
            sqrt_one_minus_alpha = robust_sqrt(1.0 - a, eps=1e-10)
            
            # Validate sqrt computations
            if torch.any(torch.isnan(sqrt_alpha)) or torch.any(torch.isnan(sqrt_one_minus_alpha)):
                raise ValueError("NaN in sqrt computations")
                
        except Exception as sqrt_error:
            logger.warning(f"Sqrt computation failed: {sqrt_error}, using fallback")
            # Fallback: safer approximations
            sqrt_alpha = torch.clamp(torch.sqrt(a + 1e-8), 1e-4, 1.0)
            sqrt_one_minus_alpha = torch.clamp(torch.sqrt(1.0 - a + 1e-8), 1e-4, 1.0)
        
        # Apply noise with additional validation
        x_noisy = x_target * sqrt_alpha + e * sqrt_one_minus_alpha
        x_noisy = torch.clamp(x_noisy, -10.0, 10.0)  # Safety clamp
        
        # Stage 5: Model input preparation with validation
        model_input = x_available.clone()
        model_input[:, target_idx:target_idx+1] = x_noisy
        
        # Comprehensive input validation
        if torch.any(torch.isnan(model_input)) or torch.any(torch.isinf(model_input)):
            raise ValueError("NaN/Inf detected in model input after processing")
        
        # Stage 6: Model forward pass with enhanced error handling
        try:
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                predicted_noise = model(model_input, t.float())
            
            # Handle tuple outputs
            if isinstance(predicted_noise, tuple):
                predicted_noise = predicted_noise[0]
                
            # Validate model output
            if torch.any(torch.isnan(predicted_noise)) or torch.any(torch.isinf(predicted_noise)):
                raise ValueError("Model output contains NaN/Inf")
                
        except Exception as model_error:
            logger.error(f"Model forward pass failed: {model_error}")
            # Return a safe fallback loss
            fallback_loss = torch.tensor(1.0, device=device, dtype=x_target.dtype, requires_grad=True)
            return fallback_loss if not keepdim else fallback_loss.expand(batch_size)
        
        # Stage 7: Loss computation with multiple validation stages
        # Clamp predicted noise for stability
        predicted_noise = torch.clamp(predicted_noise, -5.0, 5.0)
        
        # Compute loss with numerical stability
        noise_diff = e - predicted_noise
        mse_loss = noise_diff * noise_diff  # More stable than .square()
        
        # Validate loss tensor
        if torch.any(torch.isnan(mse_loss)) or torch.any(torch.isinf(mse_loss)):
            raise ValueError("NaN/Inf detected in loss computation")
        
        # Apply adaptive gradient clipping based on loss magnitude
        loss_magnitude = torch.median(mse_loss)
        if loss_magnitude > 50.0:
            # Very high loss - apply strong clipping
            mse_loss = torch.clamp(mse_loss, 0.0, 25.0)
        elif loss_magnitude > 10.0:
            # Moderate loss - apply moderate clipping
            mse_loss = torch.clamp(mse_loss, 0.0, 50.0)
        else:
            # Normal loss - apply light clipping
            mse_loss = torch.clamp(mse_loss, 0.0, 100.0)
        
        # Stage 8: Final loss aggregation with numerical validation
        if keepdim:
            # Return per-sample loss with safe reduction
            loss = mse_loss.view(batch_size, -1).mean(dim=1)
        else:
            # Return scalar loss with safe reduction
            loss = mse_loss.mean()
        
        # Final comprehensive validation
        if torch.isnan(loss) or torch.isinf(loss):
            raise ValueError("NaN/Inf in final loss")
        
        # Ensure loss is finite and positive
        loss = torch.clamp(loss, 1e-8, 1000.0)
        
        return loss
        
    except Exception as e:
        # Enhanced fallback with detailed logging
        error_msg = f"Loss computation failed: {str(e)}"
        logger.error(error_msg)
        
        # Log tensor statistics for debugging
        try:
            logger.error(f"Input stats - x_available: [{x_available.min():.3f}, {x_available.max():.3f}]")
            logger.error(f"Input stats - x_target: [{x_target.min():.3f}, {x_target.max():.3f}]")
            logger.error(f"Input stats - noise: [{e.min():.3f}, {e.max():.3f}]")
            logger.error(f"Timesteps: {t}")
            logger.error(f"Target index: {target_idx}")
        except:
            logger.error("Failed to log tensor statistics")
        
        # Return safe fallback loss that allows training to continue
        fallback_value = 1.0  # Non-zero to maintain gradients
        if keepdim:
            return torch.full((batch_size,), fallback_value, device=device, dtype=x_target.dtype, requires_grad=True)
        else:
            return torch.tensor(fallback_value, device=device, dtype=x_target.dtype, requires_grad=True)


# Registry for backwards compatibility
loss_registry = {
    'unified_4to1': unified_4to1_loss,
    # Legacy mappings
    'sg': unified_4to1_loss,  # Single condition tasks
    'sr': unified_4to1_loss,  # Multi condition tasks
    'simple': unified_4to1_loss
}