import torch
import math
import numpy as np


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
    4→1 Fast-DDPM loss for unified BraTS modality synthesis
    
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
    # Alpha computation for 3D diffusion: α_t = ∏(1-β_i) for i=1 to t
    a = (1-b).cumprod(dim=0).index_select(0, t.long()).view(-1, 1, 1, 1, 1)
    
    # Add noise to target modality: x_t = √α_t * x_0 + √(1-α_t) * ε
    x_noisy = x_target * a.sqrt() + e * (1.0 - a).sqrt()
    
    # Create model input: replace target channel with noisy version
    model_input = x_available.clone()
    model_input[:, target_idx:target_idx+1] = x_noisy
    
    # Model predicts noise for the target modality
    predicted_noise = model(model_input, t.float())
    
    # Handle models that output both mean and variance
    if isinstance(predicted_noise, tuple):
        predicted_noise = predicted_noise[0]
    
    # MSE loss between actual noise and predicted noise
    mse_loss = (e - predicted_noise).square()
    
    if keepdim:
        # Return per-sample loss
        return mse_loss.view(mse_loss.size(0), -1).mean(dim=1)
    else:
        # Return scalar loss
        return mse_loss.mean()


# Registry for backwards compatibility
loss_registry = {
    'unified_4to1': unified_4to1_loss,
    # Legacy mappings
    'sg': unified_4to1_loss,  # Single condition tasks
    'sr': unified_4to1_loss,  # Multi condition tasks
    'simple': unified_4to1_loss
}