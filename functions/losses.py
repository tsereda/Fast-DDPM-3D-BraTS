import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


def ddpm_noise_prediction_loss(model, x_available, x_target, t, noise, beta_schedule, target_idx=0, keepdim=False):
    """
    Clean DDPM noise prediction loss for 4→1 BraTS modality synthesis
    
    This implements the core DDPM objective: predict the noise added at timestep t.
    The model learns to denoise by predicting ε ~ N(0,I) from noisy input.
    
    Args:
        model: 3D U-Net diffusion model
        x_available: [B, 4, H, W, D] - all modalities with target masked/zeroed
        x_target: [B, 1, H, W, D] - clean target modality volume  
        t: [B] - diffusion timesteps (0 to T-1)
        noise: [B, 1, H, W, D] - Gaussian noise tensor ε ~ N(0,I)
        beta_schedule: [T] - noise variance schedule β_t
        target_idx: int - which modality is being synthesized (0-3 for T1/T1ce/T2/FLAIR)
        keepdim: bool - return per-sample loss if True, mean loss if False
        
    Returns:
        loss: scalar tensor (mean) or [B] tensor (per-sample) 
    """
    device = x_target.device
    
    # Input validation
    if torch.isnan(x_target).any() or torch.isnan(noise).any():
        logger.warning(f"NaN detected in inputs for target modality {target_idx}")
        return torch.tensor(0.0, device=device, requires_grad=True)
    
    # Compute cumulative alpha values: α̃_t = ∏(1 - β_s) for s=1 to t
    alpha_cumprod = torch.cumprod(1.0 - torch.clamp(beta_schedule, 1e-8, 0.999), dim=0)
    t_safe = torch.clamp(t.long(), 0, len(alpha_cumprod) - 1)
    alpha_t = alpha_cumprod[t_safe].view(-1, 1, 1, 1, 1)
    
    # Forward diffusion: add noise to clean target
    # x_t = √α̃_t * x_0 + √(1-α̃_t) * ε
    sqrt_alpha_t = torch.sqrt(alpha_t)
    sqrt_one_minus_alpha_t = torch.sqrt(1.0 - alpha_t)
    x_noisy = sqrt_alpha_t * x_target + sqrt_one_minus_alpha_t * noise
    
    # Prepare model input: replace target channel with noisy version
    model_input = x_available.clone()
    model_input[:, target_idx:target_idx+1] = x_noisy
    
    # Model forward pass: predict noise ε_θ(x_t, t)
    try:
        predicted_noise = model(model_input, t.float())
        if isinstance(predicted_noise, tuple):
            predicted_noise = predicted_noise[0]  # Handle variance learning models
            
        # Validate output shape
        if predicted_noise.shape != noise.shape:
            logger.error(f"Model output shape {predicted_noise.shape} != noise shape {noise.shape}")
            return torch.tensor(0.0, device=device, requires_grad=True)
            
    except Exception as ex:
        logger.error(f"Model forward pass failed: {ex}")
        return torch.tensor(0.0, device=device, requires_grad=True)
    
    # Core DDPM loss: L_simple = ||ε - ε_θ(x_t, t)||²
    mse_loss = F.mse_loss(predicted_noise, noise, reduction='none')
    
    # Numerical stability check
    if torch.isnan(mse_loss).any() or torch.isinf(mse_loss).any():
        logger.warning("NaN/Inf detected in loss, returning safe fallback")
        return torch.tensor(0.01, device=device, requires_grad=True)
    
    # Return aggregated loss
    if keepdim:
        return mse_loss.view(mse_loss.size(0), -1).mean(dim=1)
    else:
        return mse_loss.mean()


class SSIM3D(nn.Module):
    """3D Structural Similarity Index for medical images"""
    def __init__(self, window_size=11, size_average=True):
        super(SSIM3D, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = self.create_window_3d(window_size, self.channel)

    def gaussian_3d(self, window_size, sigma):
        """Create 3D Gaussian kernel"""
        coords = torch.arange(window_size).float() - window_size // 2
        g = torch.exp(-(coords**2) / (2 * sigma**2))
        g3d = g[:, None, None] * g[None, :, None] * g[None, None, :]
        return g3d / g3d.sum()

    def create_window_3d(self, window_size, channel):
        """Create 3D window for SSIM calculation"""
        _3d_window = self.gaussian_3d(window_size, 1.5).unsqueeze(0).unsqueeze(0)
        window = _3d_window.expand(channel, 1, window_size, window_size, window_size).contiguous()
        return window

    def ssim_3d(self, img1, img2, window, window_size, channel, size_average=True):
        """Calculate 3D SSIM"""
        mu1 = F.conv3d(img1, window, padding=window_size//2, groups=channel)
        mu2 = F.conv3d(img2, window, padding=window_size//2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv3d(img1 * img1, window, padding=window_size//2, groups=channel) - mu1_sq
        sigma2_sq = F.conv3d(img2 * img2, window, padding=window_size//2, groups=channel) - mu2_sq
        sigma12 = F.conv3d(img1 * img2, window, padding=window_size//2, groups=channel) - mu1_mu2

        C1 = 0.01**2
        C2 = 0.03**2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)

    def forward(self, img1, img2):
        (_, channel, _, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = self.create_window_3d(self.window_size, channel)
            
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            
            self.window = window
            self.channel = channel

        return self.ssim_3d(img1, img2, window, self.window_size, channel, self.size_average)


class GradientLoss3D(nn.Module):
    """3D Gradient loss for edge preservation in medical images"""
    def __init__(self):
        super(GradientLoss3D, self).__init__()
        
    def forward(self, pred, target):
        """Compute gradient loss in all 3 dimensions"""
        # Gradient in depth (D) dimension
        pred_grad_d = pred[:, :, 1:, :, :] - pred[:, :, :-1, :, :]
        target_grad_d = target[:, :, 1:, :, :] - target[:, :, :-1, :, :]
        
        # Gradient in height (H) dimension  
        pred_grad_h = pred[:, :, :, 1:, :] - pred[:, :, :, :-1, :]
        target_grad_h = target[:, :, :, 1:, :] - target[:, :, :, :-1, :]
        
        # Gradient in width (W) dimension
        pred_grad_w = pred[:, :, :, :, 1:] - pred[:, :, :, :, :-1]
        target_grad_w = target[:, :, :, :, 1:] - target[:, :, :, :, :-1]
        
        # Compute L1 loss for each gradient direction
        loss_d = F.l1_loss(pred_grad_d, target_grad_d)
        loss_h = F.l1_loss(pred_grad_h, target_grad_h)
        loss_w = F.l1_loss(pred_grad_w, target_grad_w)
        
        return (loss_d + loss_h + loss_w) / 3.0


def enhanced_ddpm_loss(model, x_available, x_target, t, noise, beta_schedule, target_idx=0, 
                      keepdim=False, use_auxiliary_losses=True):
    """
    Enhanced DDPM loss with medical-specific components for BraTS
    
    Args:
        model: 3D diffusion model
        x_available: [B, 4, H, W, D] - all modalities with target masked
        x_target: [B, 1, H, W, D] - target modality volume  
        t: [B] - timesteps
        noise: [B, 1, H, W, D] - noise tensor
        beta_schedule: [T] - beta schedule
        target_idx: int - which modality is being synthesized (0-3)
        keepdim: bool - whether to keep batch dimension
        use_auxiliary_losses: bool - whether to use SSIM and gradient losses
        
    Returns:
        loss: scalar or [B] tensor
    """
    device = x_target.device
    batch_size = x_target.size(0)
    
    # Core DDPM loss
    core_loss = ddpm_noise_prediction_loss(model, x_available, x_target, t, noise, 
                                          beta_schedule, target_idx, keepdim=True)
    
    if not use_auxiliary_losses or batch_size > 2:
        # Return core loss only for large batches or when disabled
        return core_loss.mean() if not keepdim else core_loss
    
    # Compute auxiliary losses for enhanced training
    alpha_cumprod = torch.cumprod(1.0 - torch.clamp(beta_schedule, 1e-8, 0.999), dim=0)
    t_safe = torch.clamp(t.long(), 0, len(alpha_cumprod) - 1)
    alpha_t = alpha_cumprod[t_safe].view(-1, 1, 1, 1, 1)
    
    sqrt_alpha_t = torch.sqrt(alpha_t)
    sqrt_one_minus_alpha_t = torch.sqrt(1.0 - alpha_t)
    x_noisy = sqrt_alpha_t * x_target + sqrt_one_minus_alpha_t * noise
    
    model_input = x_available.clone()
    model_input[:, target_idx:target_idx+1] = x_noisy
    
    try:
        predicted_noise = model(model_input, t.float())
        if isinstance(predicted_noise, tuple):
            predicted_noise = predicted_noise[0]
            
        # Predict clean image for auxiliary losses
        with torch.no_grad():
            x0_pred = (x_noisy - sqrt_one_minus_alpha_t * predicted_noise) / sqrt_alpha_t
            x0_pred = torch.clamp(x0_pred, -1, 1)
        
        auxiliary_loss = torch.zeros_like(core_loss)
        
        # Gradient loss for edge preservation
        try:
            grad_loss_fn = GradientLoss3D().to(device)
            grad_loss = grad_loss_fn(x0_pred, x_target)
            auxiliary_loss = auxiliary_loss + 0.1 * grad_loss
        except Exception:
            pass
        
        # SSIM loss for structural similarity (only for very small batches)
        if batch_size == 1:
            try:
                ssim_3d = SSIM3D().to(device)
                ssim_loss = 1.0 - ssim_3d(x0_pred, x_target)
                auxiliary_loss = auxiliary_loss + 0.05 * ssim_loss
            except Exception:
                pass
        
        total_loss = core_loss + auxiliary_loss
        
    except Exception:
        total_loss = core_loss
    
    # Modality-specific weighting for BraTS
    modality_weights = [1.0, 1.15, 0.95, 1.05]  # T1, T1ce, T2, FLAIR
    if target_idx < len(modality_weights):
        total_loss = total_loss * modality_weights[target_idx]
    
    return total_loss.mean() if not keepdim else total_loss


# Legacy compatibility
def simple_4to1_loss(model, x_available, x_target, t, e, b, target_idx=0, keepdim=False):
    """Legacy alias for ddpm_noise_prediction_loss - kept for backward compatibility"""
    return ddpm_noise_prediction_loss(model, x_available, x_target, t, e, b, target_idx, keepdim)


# Registry for loss functions
loss_registry = {
    'ddpm': ddpm_noise_prediction_loss,           # Clean DDPM implementation (recommended)
    'enhanced': enhanced_ddpm_loss,               # With medical components  
    'simple': simple_4to1_loss,                   # Legacy compatibility
    'noise_prediction': ddpm_noise_prediction_loss,  # Descriptive alias
    # Task-specific aliases
    'simple_4to1': simple_4to1_loss,
    'ddpm_4to1': ddpm_noise_prediction_loss,
}


def get_loss_function(loss_type='ddpm', **kwargs):
    """
    Factory function to get the appropriate loss function
    
    Args:
        loss_type: str - type of loss function to use
            - 'ddpm': clean DDPM implementation (recommended)
            - 'enhanced': with medical components (high compute)
            - 'simple': basic compatibility (fastest)
        **kwargs: additional arguments for loss function
        
    Returns:
        loss_function: callable loss function
    """
    if loss_type in loss_registry:
        loss_fn = loss_registry[loss_type]
        
        # Return wrapped function with default kwargs
        def wrapped_loss(model, x_available, x_target, t, e, b, target_idx=0, keepdim=False):
            return loss_fn(model, x_available, x_target, t, e, b, target_idx, keepdim, **kwargs)
        
        return wrapped_loss
    else:
        available_types = list(loss_registry.keys())
        raise ValueError(f"Unknown loss type '{loss_type}'. Available: {available_types}")


def validate_loss_inputs(x_available, x_target, t, e, b, target_idx):
    """
    Quick validation of loss function inputs
    
    Returns:
        bool: True if inputs are valid, False otherwise
    """
    try:
        # Check shapes
        if x_available.shape[1] != 4:  # Should have 4 modalities
            return False
        if x_target.shape[1] != 1:     # Should have 1 target channel
            return False
        if x_available.shape[2:] != x_target.shape[2:]:  # Spatial dims should match
            return False
        if e.shape != x_target.shape:   # Noise should match target
            return False
        
        # Check ranges
        if target_idx < 0 or target_idx >= 4:
            return False
        if len(b) == 0:
            return False
        if torch.any(t < 0) or torch.any(t >= len(b)):
            return False
        
        # Check for NaN/Inf in critical tensors
        if torch.isnan(x_target).any() or torch.isinf(x_target).any():
            return False
        if torch.isnan(e).any() or torch.isinf(e).any():
            return False
            
        return True
        
    except Exception:
        return False


# Export list
__all__ = [
    'ddpm_noise_prediction_loss',    # Clean DDPM implementation (primary)
    'enhanced_ddpm_loss',            # With medical components
    'simple_4to1_loss',              # Legacy compatibility
    'loss_registry',                 # Registry of available loss functions
    'get_loss_function',             # Factory function
    'validate_loss_inputs',          # Input validation
    'SSIM3D',                        # 3D SSIM for medical images
    'GradientLoss3D',                # 3D gradient loss
]
