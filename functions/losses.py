import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import logging

logger = logging.getLogger(__name__)


def validate_tensor_stability(tensor, name="tensor", max_val=10.0, debug=False):
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


def streamlined_4to1_loss(model, x_available, x_target, t, e, b, target_idx=0, keepdim=False):
    """
    Streamlined 4→1 Fast-DDPM loss with medical-specific components
    
    Combines:
    - Core diffusion MSE loss
    - 3D SSIM for structural similarity  
    - Gradient loss for edge preservation
    - FF-Parser for numerical stability
    
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
    
    # Quick NaN check (simplified - no expensive fallbacks)
    if torch.isnan(x_target).any() or torch.isnan(e).any():
        logger.warning(f"NaN detected in inputs for target_idx={target_idx}")
        return torch.tensor(0.0, device=device, requires_grad=True)
    
    # Efficient alpha computation (vectorized)
    alphas_cumprod = torch.cumprod(1.0 - torch.clamp(b, 1e-8, 0.999), dim=0)
    t_clamped = torch.clamp(t.long(), 0, len(alphas_cumprod) - 1)
    a = alphas_cumprod[t_clamped].view(-1, 1, 1, 1, 1)
    
    # Core diffusion: add noise to target
    sqrt_a = torch.sqrt(a)
    sqrt_1_minus_a = torch.sqrt(1.0 - a)
    x_noisy = x_target * sqrt_a + e * sqrt_1_minus_a
    
    # Prepare model input (memory efficient)
    model_input = x_available.clone()
    model_input[:, target_idx:target_idx+1] = x_noisy
    
    # Model prediction with enhanced error handling
    try:
        predicted_noise = model(model_input, t.float())
        if isinstance(predicted_noise, tuple):
            predicted_noise = predicted_noise[0]
            
        # Validate prediction shape
        if predicted_noise.shape != e.shape:
            logger.warning(f"Shape mismatch: predicted {predicted_noise.shape} vs expected {e.shape}")
            return torch.tensor(0.0, device=device, requires_grad=True)
            
    except RuntimeError as ex:
        logger.error(f"Model forward pass failed: {ex}")
        return torch.tensor(0.0, device=device, requires_grad=True)
    except Exception as ex:
        logger.error(f"Unexpected error in model forward: {ex}")
        return torch.tensor(0.0, device=device, requires_grad=True)
    
    # Core MSE loss (primary component)
    mse_loss = F.mse_loss(predicted_noise, e, reduction='none')
    
    # Apply FF-Parser for stability (if needed)
    if torch.isnan(mse_loss).any() or torch.isinf(mse_loss).any():
        logger.warning("NaN/Inf detected in MSE loss, applying FF-Parser")
        try:
            ff_parser = FFParser3D().to(device)
            predicted_noise = ff_parser(predicted_noise)
            mse_loss = F.mse_loss(predicted_noise, e, reduction='none')
        except Exception as ex:
            logger.warning(f"FF-Parser failed: {ex}, using fallback")
            return torch.tensor(0.1, device=device, requires_grad=True)  # Small fallback loss
    
    # Initialize total loss
    total_loss = mse_loss
    
    # Additional medical-specific components (computed on denoised prediction)
    # Only add auxiliary losses for small batches to manage memory
    if batch_size <= 2:
        with torch.no_grad():
            # Predict clean image for auxiliary losses
            x0_pred = (x_noisy - sqrt_1_minus_a * predicted_noise) / sqrt_a
            x0_pred = torch.clamp(x0_pred, -1, 1)  # Clamp to valid range
        
        # Add 3D SSIM loss (structural similarity) - memory intensive
        try:
            if "SSIM3D" not in _shared_instances:
                _shared_instances["SSIM3D"] = SSIM3D().to(device)
            ssim_loss = 1.0 - _shared_instances["SSIM3D"](x0_pred, x_target)
            total_loss = total_loss + 0.1 * ssim_loss  # Weight: 10% of MSE
        except RuntimeError as ex:
            if "out of memory" in str(ex).lower():
                logger.warning("SSIM computation skipped due to memory constraints")
            else:
                logger.warning(f"SSIM computation failed: {ex}")
        except Exception:
            pass  # Skip if SSIM computation fails
        
        # Add gradient loss (edge preservation) - lightweight
        try:
            grad_loss_fn = GradientLoss3D().to(device)
            grad_loss = grad_loss_fn(x0_pred, x_target)
            total_loss = total_loss + 0.05 * grad_loss  # Weight: 5% of MSE
        except Exception as ex:
            logger.warning(f"Gradient loss computation failed: {ex}")
    
    # Modality-specific weighting (based on BraTS importance)
    modality_weights = [1.0, 1.1, 0.95, 1.05]  # T1, T1ce, T2, FLAIR
    if target_idx < len(modality_weights):
        modality_weight = modality_weights[target_idx]
        total_loss = total_loss * modality_weight
    
    # Aggregate loss
    if keepdim:
        return total_loss.view(batch_size, -1).mean(dim=1)
    else:
        return total_loss.mean()


def enhanced_4to1_loss(model, x_available, x_target, t, e, b, target_idx=0, keepdim=False, 
                      loss_weights=None, use_auxiliary_losses=True):
    """
    Enhanced 4→1 Fast-DDPM loss with configurable medical components
    
    Args:
        model: 3D diffusion model
        x_available: [B, 4, H, W, D] - all modalities with target masked
        x_target: [B, 1, H, W, D] - target modality volume  
        t: [B] - timesteps
        e: [B, 1, H, W, D] - noise tensor
        b: [T] - beta schedule
        target_idx: int - which modality is being synthesized (0-3)
        keepdim: bool - whether to keep batch dimension
        loss_weights: dict - weights for different loss components
        use_auxiliary_losses: bool - whether to use SSIM and gradient losses
        
    Returns:
        loss: scalar or [B] tensor
    """
    if loss_weights is None:
        # Default weights optimized for medical imaging
        loss_weights = {
            'mse': 1.0,         # Base diffusion loss
            'ssim': 0.15,       # Structural similarity
            'gradient': 0.08,   # Edge preservation  
            'modality_specific': 0.05  # Modality-specific weighting
        }
    
    device = x_target.device
    batch_size = x_target.size(0)
    
    # Efficient alpha computation
    alphas_cumprod = torch.cumprod(1.0 - torch.clamp(b, 1e-8, 0.999), dim=0)
    t_safe = torch.clamp(t.long(), 0, len(alphas_cumprod) - 1)
    a = alphas_cumprod[t_safe].view(-1, 1, 1, 1, 1)
    
    # Add noise to target
    sqrt_a = torch.sqrt(a)
    sqrt_1_minus_a = torch.sqrt(1.0 - a)
    x_noisy = x_target * sqrt_a + e * sqrt_1_minus_a
    
    # Model input
    model_input = x_available.clone()
    model_input[:, target_idx:target_idx+1] = x_noisy
    
    # Model prediction
    predicted_noise = model(model_input, t.float())
    if isinstance(predicted_noise, tuple):
        predicted_noise = predicted_noise[0]
    
    # Core MSE loss
    mse_loss = F.mse_loss(predicted_noise, e, reduction='none')
    
    # Initialize total loss
    total_loss = loss_weights['mse'] * mse_loss
    
    if use_auxiliary_losses:
        # Predict clean image for auxiliary losses
        with torch.no_grad():
            x0_pred = (x_noisy - sqrt_1_minus_a * predicted_noise) / sqrt_a
            x0_pred = torch.clamp(x0_pred, -1, 1)
        
        # 3D SSIM loss (if computational budget allows)
        if batch_size <= 2:  # Only for small batches due to memory
            try:
                ssim_3d = SSIM3D().to(device)
                ssim_loss = 1.0 - ssim_3d(x0_pred, x_target)
                total_loss = total_loss + loss_weights['ssim'] * ssim_loss
            except:
                pass
        
        # Gradient loss (lightweight)
        try:
            grad_loss_fn = GradientLoss3D().to(device)
            grad_loss = grad_loss_fn(x0_pred, x_target)
            total_loss = total_loss + loss_weights['gradient'] * grad_loss
        except:
            pass
    
    # Modality-specific weighting
    modality_weights = [1.0, 1.2, 0.9, 1.1]  # T1, T1ce, T2, FLAIR
    modality_weight = modality_weights[target_idx] if target_idx < 4 else 1.0
    total_loss = total_loss * modality_weight * loss_weights['modality_specific']
    
    # Aggregate
    if keepdim:
        return total_loss.view(batch_size, -1).mean(dim=1)
    else:
        return total_loss.mean()


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


# Registry for loss functions
loss_registry = {
    'streamlined': streamlined_4to1_loss,  # Recommended default
    'enhanced': enhanced_4to1_loss,        # Full medical components
    'simple': simple_4to1_loss,            # Ultra-minimal version
    # Task-specific aliases
    'enhanced_4to1': enhanced_4to1_loss,
    'simple_4to1': simple_4to1_loss,
    'sg': streamlined_4to1_loss,           # Single condition tasks
    'sr': streamlined_4to1_loss,           # Multi condition tasks
}

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


class FFParser3D(nn.Module):
    """Frequency-domain stability module for 3D medical images"""
    def __init__(self, channels=1, low_freq_weight=0.8):
        super(FFParser3D, self).__init__()
        self.low_freq_weight = low_freq_weight
        self.high_freq_weight = 1.0 - low_freq_weight
        
    def forward(self, x):
        """Apply frequency domain filtering for numerical stability"""
        # 3D FFT
        x_freq = torch.fft.fftn(x, dim=(-3, -2, -1))
        
        # Create frequency mask (emphasize low frequencies)
        d, h, w = x.shape[-3:]
        mask = torch.ones_like(x_freq)
        
        # Apply low-pass filtering to reduce high-frequency noise
        center_d, center_h, center_w = d//2, h//2, w//2
        
        # Create distance matrix from center
        dd = torch.arange(d, device=x.device).float() - center_d
        hh = torch.arange(h, device=x.device).float() - center_h  
        ww = torch.arange(w, device=x.device).float() - center_w
        
        DD, HH, WW = torch.meshgrid(dd, hh, ww, indexing='ij')
        distance = torch.sqrt(DD**2 + HH**2 + WW**2)
        
        # Gaussian low-pass filter
        sigma = min(d, h, w) * 0.3  # Adjust cutoff frequency
        freq_weight = torch.exp(-distance**2 / (2 * sigma**2))
        freq_weight = self.low_freq_weight + self.high_freq_weight * freq_weight
        
        # Apply frequency weighting
        x_freq_filtered = x_freq * freq_weight.unsqueeze(0).unsqueeze(0)
        
        # Inverse FFT
        x_filtered = torch.fft.ifftn(x_freq_filtered, dim=(-3, -2, -1)).real
        
        return x_filtered


def get_loss_function(loss_type='streamlined', **kwargs):
    """
    Factory function to get the appropriate loss function
    
    Args:
        loss_type: str - type of loss function to use
            - 'streamlined': balanced medical components (recommended)
            - 'enhanced': full medical components (high compute)
            - 'simple': basic MSE only (fastest)
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


def monitor_loss_components(model, x_available, x_target, t, e, b, target_idx=0):
    """
    Monitor individual loss components for debugging and analysis
    
    Args:
        model: 3D diffusion model
        x_available: [B, 4, H, W, D] - all modalities with target masked
        x_target: [B, 1, H, W, D] - target modality volume  
        t: [B] - timesteps
        e: [B, 1, H, W, D] - noise tensor
        b: [T] - beta schedule
        target_idx: int - which modality is being synthesized
        
    Returns:
        dict: Individual loss components for logging
    """
    device = x_target.device
    batch_size = x_target.size(0)
    
    # Compute alpha values
    alphas_cumprod = torch.cumprod(1.0 - torch.clamp(b, 1e-8, 0.999), dim=0)
    t_clamped = torch.clamp(t.long(), 0, len(alphas_cumprod) - 1)
    a = alphas_cumprod[t_clamped].view(-1, 1, 1, 1, 1)
    
    # Add noise to target
    sqrt_a = torch.sqrt(a)
    sqrt_1_minus_a = torch.sqrt(1.0 - a)
    x_noisy = x_target * sqrt_a + e * sqrt_1_minus_a
    
    # Model input
    model_input = x_available.clone()
    model_input[:, target_idx:target_idx+1] = x_noisy
    
    # Model prediction
    with torch.no_grad():
        try:
            predicted_noise = model(model_input, t.float())
            if isinstance(predicted_noise, tuple):
                predicted_noise = predicted_noise[0]
        except Exception:
            return {"mse_loss": 0.0, "ssim_loss": 0.0, "grad_loss": 0.0, "error": True}
    
    # MSE loss
    mse_loss = F.mse_loss(predicted_noise, e).item()
    
    # Initialize results
    results = {
        "mse_loss": mse_loss,
        "ssim_loss": 0.0,
        "grad_loss": 0.0,
        "modality": target_idx,
        "error": False
    }
    
    # Auxiliary losses (if batch size allows)
    if batch_size <= 2:
        try:
            # Predict clean image
            x0_pred = (x_noisy - sqrt_1_minus_a * predicted_noise) / sqrt_a
            x0_pred = torch.clamp(x0_pred, -1, 1)
            
            # SSIM loss
            try:
                ssim_3d = SSIM3D().to(device)
                ssim_loss = (1.0 - ssim_3d(x0_pred, x_target)).item()
                results["ssim_loss"] = ssim_loss
            except Exception:
                pass
            
            # Gradient loss
            try:
                grad_loss_fn = GradientLoss3D().to(device)
                grad_loss = grad_loss_fn(x0_pred, x_target).item()
                results["grad_loss"] = grad_loss
            except Exception:
                pass
                
        except Exception:
            results["error"] = True
    
    return results


def adaptive_loss_weights(epoch, total_epochs=1000, warmup_epochs=50):
    """
    Adaptive loss weights that evolve during training
    
    Args:
        epoch: Current epoch (0-indexed)
        total_epochs: Total training epochs
        warmup_epochs: Number of warmup epochs
        
    Returns:
        dict: Loss weights for current epoch
    """
    # During warmup: focus on MSE loss
    if epoch < warmup_epochs:
        mse_weight = 1.0
        ssim_weight = 0.05 * (epoch / warmup_epochs)  # Gradually increase
        grad_weight = 0.02 * (epoch / warmup_epochs)  # Gradually increase
    
    # Early training: balanced weights
    elif epoch < total_epochs * 0.3:
        mse_weight = 1.0
        ssim_weight = 0.10
        grad_weight = 0.05
    
    # Mid training: emphasize structure
    elif epoch < total_epochs * 0.7:
        mse_weight = 1.0
        ssim_weight = 0.15  # Increase structural importance
        grad_weight = 0.08  # Increase edge preservation
    
    # Late training: fine-tuning balance
    else:
        mse_weight = 1.0
        ssim_weight = 0.12
        grad_weight = 0.06
    
    return {
        'mse': mse_weight,
        'ssim': ssim_weight,
        'gradient': grad_weight
    }


def optimized_4to1_loss(model, x_available, x_target, t, e, b, target_idx=0, keepdim=False, epoch=0):
    """
    Optimized 4→1 Fast-DDPM loss for BraTS training with adaptive components
    
    Features:
    - Memory-efficient computation
    - Adaptive auxiliary loss scheduling
    - Modality-specific handling
    - Training stage awareness
    
    Args:
        model: 3D diffusion model (outputs 1 channel)
        x_available: [B, 4, H, W, D] - all modalities with target masked (zeroed)
        x_target: [B, 1, H, W, D] - target modality volume  
        t: [B] - timesteps
        e: [B, 1, H, W, D] - noise tensor (same shape as target)
        b: [T] - beta schedule
        target_idx: int - which modality is being synthesized (0-3)
        keepdim: bool - whether to keep batch dimension in loss
        epoch: int - current training epoch for adaptive weighting
        
    Returns:
        loss: scalar tensor (mean) or [B] tensor (keepdim=True)
    """
    device = x_target.device
    batch_size = x_target.size(0)
    
    # Early validation
    if torch.isnan(x_target).any() or torch.isnan(e).any():
        return torch.tensor(0.0, device=device, requires_grad=True)
    
    # Efficient alpha computation
    alphas_cumprod = torch.cumprod(1.0 - torch.clamp(b, 1e-8, 0.999), dim=0)
    t_safe = torch.clamp(t.long(), 0, len(alphas_cumprod) - 1)
    a = alphas_cumprod[t_safe].view(-1, 1, 1, 1, 1)
    
    # Noise injection
    sqrt_a = torch.sqrt(a)
    sqrt_1_minus_a = torch.sqrt(1.0 - a)
    x_noisy = x_target * sqrt_a + e * sqrt_1_minus_a
    
    # Model input preparation
    model_input = x_available.clone()
    model_input[:, target_idx:target_idx+1] = x_noisy
    
    # Model prediction
    try:
        predicted_noise = model(model_input, t.float())
        if isinstance(predicted_noise, tuple):
            predicted_noise = predicted_noise[0]
    except Exception:
        return torch.tensor(0.0, device=device, requires_grad=True)
    
    # Core MSE loss
    mse_loss = F.mse_loss(predicted_noise, e, reduction='none')
    
    # Get adaptive weights based on training stage
    weights = adaptive_loss_weights(epoch)
    total_loss = weights['mse'] * mse_loss
    
    # Auxiliary losses (memory and compute aware)
    use_auxiliary = (
        batch_size <= 2 and  # Memory constraint
        epoch >= 10 and     # Skip early training
        epoch % 5 == 0       # Compute every 5th epoch only
    )
    
    if use_auxiliary:
        with torch.no_grad():
            x0_pred = (x_noisy - sqrt_1_minus_a * predicted_noise) / sqrt_a
            x0_pred = torch.clamp(x0_pred, -1, 1)
        
        # Lightweight gradient loss only (skip expensive SSIM)
        try:
            grad_loss_fn = GradientLoss3D().to(device)
            grad_loss = grad_loss_fn(x0_pred, x_target)
            total_loss = total_loss + weights['gradient'] * grad_loss
        except Exception:
            pass
    
    # Modality-specific weighting for BraTS
    modality_weights = {
        0: 1.0,   # T1 - baseline
        1: 1.15,  # T1ce - enhanced (important for tumor core)
        2: 0.95,  # T2 - slightly lower weight
        3: 1.05   # FLAIR - important for edema
    }
    
    if target_idx in modality_weights:
        total_loss = total_loss * modality_weights[target_idx]
    
    # Final aggregation
    if keepdim:
        return total_loss.view(batch_size, -1).mean(dim=1)
    else:
        return total_loss.mean()


# Add to registry
loss_registry['optimized'] = optimized_4to1_loss
loss_registry['optimized_4to1'] = optimized_4to1_loss


# Export list
__all__ = [
    'streamlined_4to1_loss',
    'enhanced_4to1_loss', 
    'simple_4to1_loss',
    'monitor_loss_components',
    'adaptive_loss_weights',
    'loss_registry',
    'get_loss_function',
    'validate_loss_inputs',
    'SSIM3D',
    'GradientLoss3D',
    'FFParser3D'
]