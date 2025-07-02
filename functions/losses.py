import torch
import math
import time
import numpy as np
np.bool = np.bool_


def calculate_psnr(img1, img2):
    """
    Calculate Peak Signal-to-Noise Ratio between two images
    
    Args:
        img1: predicted image (numpy array)
        img2: ground truth image (numpy array)
        Both images should have range [0, 255]
    
    Returns:
        psnr: Peak Signal-to-Noise Ratio value
    """
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    
    mse = np.mean((img1 - img2)**2)
    psnr = 20 * math.log10(255.0 / math.sqrt(mse))

    return psnr


def noise_estimation_loss(model,
                          x0: torch.Tensor,
                          t: torch.LongTensor,
                          e: torch.Tensor,
                          b: torch.Tensor, keepdim=False):
    """
    Basic DDPM noise estimation loss for single modality denoising
    
    Args:
        model: diffusion model
        x0: clean target volume [B, 1, H, W, D]
        t: timesteps [B]
        e: noise tensor [B, 1, H, W, D]
        b: beta schedule [T]
        keepdim: whether to keep batch dimension
    
    Returns:
        loss: noise prediction loss
    """
    # a: cumulative alpha values (1-b).cumprod()
    a = (1-b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1, 1)
    # X_t: noisy version of x0
    x = x0 * a.sqrt() + e * (1.0 - a).sqrt()
    output = model(x, t.float())
    if keepdim:
        return (e - output).square().sum(dim=(1, 2, 3, 4))
    else:
        return (e - output).square().sum(dim=(1, 2, 3, 4)).mean(dim=0)


def brats_4to1_loss(model,
                    x_available: torch.Tensor,
                    x_target: torch.Tensor,
                    t: torch.LongTensor,
                    e: torch.Tensor,
                    b: torch.Tensor, 
                    target_idx: int = 0,
                    keepdim=False):
    """
    BraTS 4→1 modality synthesis loss using available modalities as context
    
    This loss function synthesizes one missing modality given the other three.
    The model takes all 4 modalities as input (with target replaced by noise)
    and learns to predict the noise for the target modality.
    
    Args:
        model: 3D diffusion model  
        x_available: [B, 4, H, W, D] - all modalities with target masked/zeroed
        x_target: [B, 1, H, W, D] - clean target modality to synthesize
        t: [B] - diffusion timesteps
        e: [B, 1, H, W, D] - noise tensor
        b: [T] - beta schedule
        target_idx: which modality index to synthesize (0-3 for T1/T1ce/T2/FLAIR)
        keepdim: whether to keep batch dimension
    
    Returns:
        loss: noise prediction loss for target modality
    """
    # a: cumulative alpha values (1-b).cumprod()  
    a = (1-b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1, 1)
    # Add noise to target modality: X_t = sqrt(a) * x0 + sqrt(1-a) * noise
    x_noisy = x_target * a.sqrt() + e * (1.0 - a).sqrt()
    
    # Prepare model input: replace target channel with noisy version
    model_input = x_available.clone()
    model_input[:, target_idx:target_idx+1] = x_noisy
    
    # Model predicts noise
    output = model(model_input, t.float())
    
    # Use mean for proper loss scaling instead of sum
    if keepdim:
        return (e - output).square().mean(dim=(1, 2, 3, 4))
    else:
        return (e - output).square().mean()


def brats_multimodal_loss(model,
                          x_full: torch.Tensor,
                          x_target: torch.Tensor,
                          t: torch.LongTensor,
                          e: torch.Tensor,
                          b: torch.Tensor, keepdim=False):
    """
    Multi-modal BraTS loss for joint training on all modalities
    
    This loss can handle scenarios where we want to denoise/synthesize
    any combination of modalities simultaneously.
    
    Args:
        model: 3D diffusion model
        x_full: [B, 4, H, W, D] - all available modalities as context
        x_target: [B, C, H, W, D] - target modalities to synthesize (C can be 1-4)
        t: [B] - diffusion timesteps  
        e: [B, C, H, W, D] - noise tensor matching target shape
        b: [T] - beta schedule
        keepdim: whether to keep batch dimension
    
    Returns:
        loss: noise prediction loss
    """
    # a: cumulative alpha values (1-b).cumprod()
    a = (1-b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1, 1)
    # Add noise to target: X_t = sqrt(a) * x0 + sqrt(1-a) * noise  
    x_noisy = x_target * a.sqrt() + e * (1.0 - a).sqrt()
    
    # Concatenate full context with noisy target
    model_input = torch.cat([x_full, x_noisy], dim=1)
    output = model(model_input, t.float())

    if keepdim:
        return (e - output).square().sum(dim=(1, 2, 3, 4))
    else:
        return (e - output).square().sum(dim=(1, 2, 3, 4)).mean(dim=0)


# Registry for loss functions with clear naming
loss_registry = {
    'simple': noise_estimation_loss,           # Basic DDPM noise estimation
    'brats_4to1': brats_4to1_loss,            # BraTS 4→1 modality synthesis 
    'brats_multimodal': brats_multimodal_loss # Multi-modal BraTS synthesis
}


def get_loss_function(loss_type='brats_4to1'):
    """
    Factory function to get the appropriate loss function
    
    Args:
        loss_type: str - type of loss function to use
            - 'simple': basic noise estimation loss
            - 'brats_4to1': BraTS 4→1 modality synthesis (recommended)
            - 'brats_multimodal': multi-modal synthesis
        
    Returns:
        loss_function: callable loss function
    """
    if loss_type in loss_registry:
        return loss_registry[loss_type]
    else:
        available_types = list(loss_registry.keys())
        raise ValueError(f"Unknown loss type '{loss_type}'. Available: {available_types}")


# Export list
__all__ = [
    'noise_estimation_loss',         # Basic DDPM loss
    'brats_4to1_loss',              # Main BraTS 4→1 loss (primary)
    'brats_multimodal_loss',        # Multi-modal BraTS loss
    'calculate_psnr',               # PSNR calculation utility
    'loss_registry',                # Registry of available loss functions
    'get_loss_function',            # Factory function
]
