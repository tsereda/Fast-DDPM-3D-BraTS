"""
Loss functions for 3D Fast-DDPM training
"""
import torch
import torch.nn.functional as F
import numpy as np


def compute_alpha(beta, t):
    """Compute alpha values for diffusion timesteps"""
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1, 1)
    return a


def sg_noise_estimation_loss(model, x_available, x_target, t, e, betas, unified_training=True):
    """
    Score-based Generative noise estimation loss for 3D Fast-DDPM
    
    Args:
        model: 3D diffusion model
        x_available: [B, 4, H, W, D] - available input modalities 
        x_target: [B, 1, H, W, D] - target modality to generate
        t: [B] - timesteps
        e: [B, 1, H, W, D] - noise
        betas: beta schedule
        unified_training: whether to use unified 4→4 training
    """
    a = compute_alpha(betas, t.long())
    
    # Forward diffusion: add noise to target
    x_noisy = x_target * a.sqrt() + e * (1.0 - a).sqrt()
    
    if unified_training:
        # Unified 4→4: replace one channel of available modalities with noisy target
        # For simplicity, replace the first channel - this can be made more sophisticated
        model_input = x_available.clone()
        model_input[:, 0:1] = x_noisy
    else:
        # Traditional approach: concatenate available + noisy target
        model_input = torch.cat([x_available, x_noisy], dim=1)
    
    # Predict noise
    output = model(model_input, t.float())
    
    # L2 loss between predicted and actual noise
    loss = F.mse_loss(e, output, reduction='none')
    loss = loss.mean(dim=(1, 2, 3, 4))  # Mean over spatial dimensions
    loss = loss.mean()  # Mean over batch
    
    return loss


def sg_noise_estimation_loss_simple(model, x, t, e, betas):
    """
    Simplified noise estimation loss for basic training
    
    Args:
        model: 3D diffusion model
        x: [B, C, H, W, D] - input (can be 4 or 1 channel)
        t: [B] - timesteps
        e: [B, 1, H, W, D] - noise
        betas: beta schedule
    """
    a = compute_alpha(betas, t.long())
    
    # If input has multiple channels, take only the first for noise addition
    if x.shape[1] > 1:
        x_target = x[:, 0:1]  # Take first channel as target
    else:
        x_target = x
    
    # Forward diffusion
    x_noisy = x_target * a.sqrt() + e * (1.0 - a).sqrt()
    
    # If multi-channel input, replace first channel with noisy version
    if x.shape[1] > 1:
        model_input = x.clone()
        model_input[:, 0:1] = x_noisy
    else:
        model_input = x_noisy
    
    # Predict noise
    output = model(model_input, t.float())
    
    # L2 loss
    loss = F.mse_loss(e, output)
    return loss


def perceptual_loss_3d(x_pred, x_target, weight=1.0):
    """
    Simple perceptual loss for 3D volumes
    Uses L1 + gradient loss
    """
    # L1 loss
    l1_loss = F.l1_loss(x_pred, x_target)
    
    # Gradient loss in all 3 spatial dimensions
    grad_loss = 0
    for dim in [2, 3, 4]:  # H, W, D dimensions
        grad_pred = torch.diff(x_pred, dim=dim)
        grad_target = torch.diff(x_target, dim=dim)
        grad_loss += F.l1_loss(grad_pred, grad_target)
    
    grad_loss /= 3  # Average over spatial dimensions
    
    return weight * (l1_loss + 0.1 * grad_loss)


def combined_loss(model, x_available, x_target, t, e, betas, perceptual_weight=0.1):
    """
    Combined noise estimation + perceptual loss
    """
    # Main diffusion loss
    noise_loss = sg_noise_estimation_loss(model, x_available, x_target, t, e, betas)
    
    # Generate prediction for perceptual loss
    with torch.no_grad():
        a = compute_alpha(betas, t.long())
        x_noisy = x_target * a.sqrt() + e * (1.0 - a).sqrt()
        model_input = x_available.clone()
        model_input[:, 0:1] = x_noisy
        pred_noise = model(model_input, t.float())
        
        # Estimate x0 from predicted noise
        x_pred = (x_noisy - pred_noise * (1.0 - a).sqrt()) / a.sqrt()
    
    # Perceptual loss
    perc_loss = perceptual_loss_3d(x_pred, x_target, weight=perceptual_weight)
    
    return noise_loss + perc_loss, noise_loss, perc_loss


# Alias for backward compatibility
noise_estimation_loss = sg_noise_estimation_loss