import torch
import math
import time
from medpy import metric
import numpy as np
np.bool = np.bool_


def calculate_psnr(img1, img2):
    # img1: img
    # img2: gt
    # img1 and img2 have range [0, 255]
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
    # a: a_T in DDIM
    # 1-a: 1-a_T in DDIM 
    a = (1-b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
    # X_T
    x = x0 * a.sqrt() + e * (1.0 - a).sqrt()
    output = model(x, t.float())
    if keepdim:
        return (e - output).square().sum(dim=(1, 2, 3))
    else:
        return (e - output).square().sum(dim=(1, 2, 3)).mean(dim=0)


def sr_noise_estimation_loss(model,
                          x_bw: torch.Tensor,
                          x_md: torch.Tensor,
                          x_fw: torch.Tensor,
                          t: torch.LongTensor,
                          e: torch.Tensor,
                          b: torch.Tensor, keepdim=False):
    # a: a_T in DDIM
    # 1-a: 1-a_T in DDIM 
    a = (1-b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
    # X_T
    x = x_md * a.sqrt() + e * (1.0 - a).sqrt()

    output = model(torch.cat([x_bw, x_fw, x], dim=1), t.float())
    if keepdim:
        return (e - output).square().sum(dim=(1, 2, 3))
    else:
        return (e - output).square().sum(dim=(1, 2, 3)).mean(dim=0)


def sg_noise_estimation_loss(model,
                          x_img: torch.Tensor,
                          x_gt: torch.Tensor,
                          t: torch.LongTensor,
                          e: torch.Tensor,
                          b: torch.Tensor, keepdim=False):
    # a: a_T in DDIM
    # 1-a: 1-a_T in DDIM 
    a = (1-b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
    # X_T
    x = x_gt * a.sqrt() + e * (1.0 - a).sqrt()
    output = model(torch.cat([x_img, x], dim=1), t.float())

    if keepdim:
        return (e - output).square().sum(dim=(1, 2, 3))
    else:
        return (e - output).square().sum(dim=(1, 2, 3)).mean(dim=0)


def unified_4to1_loss(model, x_available, x_target, t, e, b, target_idx=0, keepdim=False):
    """
    4→1 Fast-DDPM loss for unified BraTS modality synthesis
    
    Args:
        model: 3D diffusion model (outputs 1 channel)
        x_available: [B, 4, H, W, D] - all modalities with target zeroed
        x_target: [B, 1, H, W, D] - target modality volume  
        t: timesteps
        e: [B, 1, H, W, D] - noise tensor (same shape as target)
        b: beta schedule
        target_idx: which modality is being synthesized (0-3)
        keepdim: whether to keep batch dimension in loss
    """
    # Alpha computation for 3D diffusion
    a = (1-b).cumprod(dim=0).index_select(0, t.long()).view(-1, 1, 1, 1, 1)
    
    # Add noise to target modality: x_t = sqrt(a) * x_0 + sqrt(1-a) * noise
    x_noisy = x_target * a.sqrt() + e * (1.0 - a).sqrt()
    
    # Create model input: replace target channel with noisy version
    model_input = x_available.clone()
    model_input[:, target_idx:target_idx+1] = x_noisy
    
    # Model predicts noise for the target modality [B, 1, H, W, D]
    # Note: This is actually 4→1, despite the function name for compatibility
    predicted_noise = model(model_input, t.float())
    
    # Handle variance learning models
    if isinstance(predicted_noise, tuple):
        predicted_noise = predicted_noise[0]
    
    # MSE loss between actual noise and predicted noise
    # Normalize by volume size to prevent scaling issues
    mse_loss = (e - predicted_noise).square()
    
    if keepdim:
        # Return per-sample loss, normalized by volume size
        return mse_loss.view(mse_loss.size(0), -1).mean(dim=1)
    else:
        # Return scalar loss, properly normalized
        return mse_loss.mean()


# 3D versions of existing loss functions
def sg_noise_estimation_loss_3d(model, x_img, x_gt, t, e, b, keepdim=False):
    """
    3D Fast-DDPM noise estimation loss for single-condition tasks
    (image translation, CT denoising)
    """
    a = (1-b).cumprod(dim=0).index_select(0, t.long()).view(-1, 1, 1, 1, 1)
    
    # Add noise to ground truth
    x = x_gt * a.sqrt() + e * (1.0 - a).sqrt()
    
    # Model input: concatenate condition image with noisy target
    model_input = torch.cat([x_img, x], dim=1)
    
    # Predict noise
    output = model(model_input, t.float())
    
    # Handle models that output variance
    if isinstance(output, tuple):
        output = output[0]
    
    # Compute MSE loss with proper normalization
    mse_loss = (e - output).square()
    
    if keepdim:
        # Return per-sample loss, normalized by volume size
        return mse_loss.view(mse_loss.size(0), -1).mean(dim=1)
    else:
        # Return scalar loss, properly normalized
        return mse_loss.mean()


def sr_noise_estimation_loss_3d(model, x_bw, x_md, x_fw, t, e, b, keepdim=False):
    """
    3D Fast-DDPM noise estimation loss for multi-condition tasks
    (super-resolution with backward/middle/forward frames)
    """
    a = (1-b).cumprod(dim=0).index_select(0, t.long()).view(-1, 1, 1, 1, 1)
    
    # Add noise to middle frame (target)
    x = x_md * a.sqrt() + e * (1.0 - a).sqrt()
    
    # Model input: concatenate all condition images with noisy target
    model_input = torch.cat([x_bw, x_fw, x], dim=1)
    
    # Predict noise
    output = model(model_input, t.float())
    
    # Handle models that output variance
    if isinstance(output, tuple):
        output = output[0]
    
    # Compute MSE loss with proper normalization
    mse_loss = (e - output).square()
    
    if keepdim:
        # Return per-sample loss, normalized by volume size
        return mse_loss.view(mse_loss.size(0), -1).mean(dim=1)
    else:
        # Return scalar loss, properly normalized
        return mse_loss.mean()


loss_registry = {
    'simple': noise_estimation_loss,
    'sr': sr_noise_estimation_loss,
    'sg': sg_noise_estimation_loss,
    'sr_3d': sr_noise_estimation_loss_3d,
    'sg_3d': sg_noise_estimation_loss_3d,
    'unified_4to1': unified_4to1_loss
}