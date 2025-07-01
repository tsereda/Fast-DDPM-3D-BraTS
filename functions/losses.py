import torch
import torch.nn.functional as F
import math
import numpy as np


def calculate_psnr(img1, img2):
    """Calculate PSNR between two images"""
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    psnr = 20 * math.log10(255.0 / math.sqrt(mse))
    return psnr


def sg_noise_estimation_loss(model, x_img, x_gt, t, e, b, keepdim=False):
    """
    3D Fast-DDPM noise estimation loss for single-condition tasks
    (image translation, CT denoising)
    
    Args:
        model: 3D diffusion model
        x_img: [B, C, H, W, D] - condition image
        x_gt: [B, C, H, W, D] - ground truth target
        t: timesteps
        e: noise tensor
        b: beta schedule
        keepdim: whether to keep batch dimension in loss
    """
    # Original Fast-DDPM approach: simple alpha computation for 3D
    a = (1-b).cumprod(dim=0).index_select(0, t.long()).view(-1, 1, 1, 1, 1)
    
    # Add noise to ground truth
    x = x_gt * a.sqrt() + e * (1.0 - a).sqrt()
    
    # Model input: concatenate condition image with noisy target
    model_input = torch.cat([x_img, x], dim=1)
    
    # Predict noise
    output = model(model_input, t.float())
    
    # Handle models that output variance (take only mean prediction)
    if isinstance(output, tuple):
        output = output[0]
    
    # Compute MSE loss
    if keepdim:
        return (e - output).square().sum(dim=(1, 2, 3, 4))
    else:
        return (e - output).square().sum(dim=(1, 2, 3, 4)).mean(dim=0)


def sr_noise_estimation_loss(model, x_bw, x_md, x_fw, t, e, b, keepdim=False):
    """
    3D Fast-DDPM noise estimation loss for multi-condition tasks
    (super-resolution with backward/middle/forward frames)
    
    Args:
        model: 3D diffusion model
        x_bw: [B, C, H, W, D] - backward frame
        x_md: [B, C, H, W, D] - middle frame (target)
        x_fw: [B, C, H, W, D] - forward frame
        t: timesteps
        e: noise tensor
        b: beta schedule
        keepdim: whether to keep batch dimension in loss
    """
    # Original Fast-DDPM approach: simple alpha computation for 3D
    a = (1-b).cumprod(dim=0).index_select(0, t.long()).view(-1, 1, 1, 1, 1)
    
    # Add noise to middle frame (target)
    x = x_md * a.sqrt() + e * (1.0 - a).sqrt()
    
    # Model input: concatenate all condition images with noisy target
    model_input = torch.cat([x_bw, x_fw, x], dim=1)
    
    # Predict noise
    output = model(model_input, t.float())
    
    # Handle models that output variance (take only mean prediction)
    if isinstance(output, tuple):
        output = output[0]
    
    # Compute MSE loss
    if keepdim:
        return (e - output).square().sum(dim=(1, 2, 3, 4))
    else:
        return (e - output).square().sum(dim=(1, 2, 3, 4)).mean(dim=0)


def unified_4to4_loss(model, x_available, x_target, t, e, b, target_idx=0, keepdim=False):
    """
    3D Fast-DDPM loss for unified 4->4 BraTS modality synthesis
    
    Args:
        model: 3D diffusion model
        x_available: [B, 4, H, W, D] - all 4 modalities with target zeroed out
        x_target: [B, 1, H, W, D] - target modality volume
        t: timesteps
        e: noise tensor
        b: beta schedule
        target_idx: which modality is being synthesized (0-3)
        keepdim: whether to keep batch dimension in loss
    """
    # Original Fast-DDPM approach: simple alpha computation for 3D
    a = (1-b).cumprod(dim=0).index_select(0, t.long()).view(-1, 1, 1, 1, 1)
    
    # Add noise to target modality
    x_noisy = x_target * a.sqrt() + e * (1.0 - a).sqrt()
    
    # Replace the target channel with noisy version
    model_input = x_available.clone()
    model_input[:, target_idx:target_idx+1] = x_noisy
    
    # Predict noise
    output = model(model_input, t.float())
    
    # Handle models that output variance (take only mean prediction)
    if isinstance(output, tuple):
        output = output[0]
    
    # Compute MSE loss
    if keepdim:
        return (e - output).square().sum(dim=(1, 2, 3, 4))
    else:
        return (e - output).square().sum(dim=(1, 2, 3, 4)).mean(dim=0)


# Loss registry for different tasks
loss_registry = {
    'sg': sg_noise_estimation_loss,
    'sr': sr_noise_estimation_loss,
    'unified_4to4': unified_4to4_loss
}