"""
Fast-DDPM loss functions with variance learning
Implements the full Fast-DDPM approach including VLB loss
"""
import torch
import torch.nn.functional as F
import numpy as np


def compute_alpha(beta, t):
    """Compute alpha values for diffusion timesteps"""
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1, 1)
    return a


def extract(a, t, x_shape):
    """Extract coefficients at specified timesteps and reshape to broadcast with x_shape"""
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def q_posterior_mean_variance(x_start, x_t, alpha_t, alpha_prev):
    """Compute posterior mean and variance for diffusion process"""
    # Compute posterior mean
    posterior_mean = (
        alpha_prev.sqrt() * x_start +
        (1 - alpha_prev).sqrt() * x_t
    ) / (1 - alpha_t).sqrt()
    
    # Compute posterior variance  
    posterior_variance = (1 - alpha_prev) * (1 - alpha_t) / (1 - alpha_t)
    posterior_variance = posterior_variance.clamp(min=1e-20)
    posterior_log_variance = torch.log(posterior_variance)
    
    return posterior_mean, posterior_variance, posterior_log_variance


def normal_kl(mean1, logvar1, mean2, logvar2):
    """Compute KL divergence between two normal distributions"""
    return 0.5 * (
        -1.0 + logvar2 - logvar1 + torch.exp(logvar1 - logvar2)
        + ((mean1 - mean2) ** 2) * torch.exp(-logvar2)
    )


def discretized_gaussian_log_likelihood(x, means, log_scales):
    """Compute log likelihood for discretized Gaussian"""
    assert x.shape == means.shape == log_scales.shape
    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
    cdf_plus = torch.distributions.Normal(0, 1).cdf(plus_in)
    min_in = inv_stdv * (centered_x - 1.0 / 255.0)
    cdf_min = torch.distributions.Normal(0, 1).cdf(min_in)
    log_cdf_plus = torch.log(cdf_plus.clamp(min=1e-12))
    log_one_minus_cdf_min = torch.log((1.0 - cdf_min).clamp(min=1e-12))
    cdf_delta = cdf_plus - cdf_min
    log_probs = torch.where(
        x < -0.999,
        log_cdf_plus,
        torch.where(x > 0.999, log_one_minus_cdf_min, torch.log(cdf_delta.clamp(min=1e-12)))
    )
    return log_probs
    return log_probs


def fast_ddpm_loss(model, x_available, x_target, t, e, betas, var_type='learned'):
    """
    Fast-DDPM loss with variance learning
    
    Args:
        model: 3D diffusion model
        x_available: [B, 4, H, W, D] - available input modalities 
        x_target: [B, 1, H, W, D] - target modality to generate
        t: [B] - timesteps
        e: [B, 1, H, W, D] - noise
        betas: beta schedule
        var_type: 'fixed', 'learned', or 'learned_range'
    """
    batch_size = x_target.shape[0]
    device = x_target.device
    
    # Get alpha values
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
    
    # Get values for current timestep
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
    log_one_minus_alphas_cumprod = torch.log(1.0 - alphas_cumprod)
    sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod)
    sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod - 1)
    
    # Get values at timestep t
    a_t = extract(alphas_cumprod, t, x_target.shape)
    a_prev = extract(alphas_cumprod_prev, t, x_target.shape)
    sqrt_a_t = extract(sqrt_alphas_cumprod, t, x_target.shape)
    sqrt_one_minus_a_t = extract(sqrt_one_minus_alphas_cumprod, t, x_target.shape)
    
    # Forward diffusion: add noise to target
    x_noisy = x_target * sqrt_a_t + e * sqrt_one_minus_a_t
    
    # Unified 4→4: replace one channel with noisy target
    model_input = x_available.clone()
    model_input[:, 0:1] = x_noisy
    
    # Model prediction
    if var_type in ['learned', 'learned_range']:
        model_mean, model_log_variance = model(model_input, t.float())
    else:
        model_mean = model(model_input, t.float())
        model_log_variance = None
    
    # Get target values
    # Predict x0
    x0_pred = (x_noisy - sqrt_one_minus_a_t * model_mean) / sqrt_a_t
    x0_pred = torch.clamp(x0_pred, -1, 1)
    
    # Get distribution parameters
    posterior_mean, posterior_variance, posterior_log_variance = \
        q_posterior_mean_variance(x_target, x0_pred, a_t, a_prev)
    
    # Simple L2 loss for mean
    mean_loss = F.mse_loss(model_mean, e, reduction='none')
    mean_loss = mean_loss.mean(dim=(1, 2, 3, 4))
    
    if var_type == 'fixed':
        loss = mean_loss
    elif var_type == 'learned':
        # VLB loss for variance
        true_mean = posterior_mean
        true_log_variance = posterior_log_variance
        
        kl = normal_kl(true_mean, true_log_variance, x0_pred, model_log_variance)
        kl = kl.mean(dim=(1, 2, 3, 4)) / np.log(2.0)
        
        decoder_nll = -discretized_gaussian_log_likelihood(
            x_target, x0_pred, 0.5 * model_log_variance
        )
        decoder_nll = decoder_nll.mean(dim=(1, 2, 3, 4)) / np.log(2.0)
        
        # At t=0, use decoder NLL, otherwise use KL
        vb_loss = torch.where((t == 0), decoder_nll, kl)
        
        # Combine losses
        loss = mean_loss + 0.001 * vb_loss
    elif var_type == 'learned_range':
        # Learn interpolation between fixed small and large variance
        min_log = extract(posterior_log_variance, t, x_target.shape)
        max_log = extract(torch.log(betas), t, x_target.shape)
        # Model predicts interpolation fraction
        frac = (model_log_variance + 1) / 2
        model_log_variance = frac * max_log + (1 - frac) * min_log
        
        # Compute VLB loss
        true_mean = posterior_mean
        true_log_variance = posterior_log_variance
        
        kl = normal_kl(true_mean, true_log_variance, x0_pred, model_log_variance)
        kl = kl.mean(dim=(1, 2, 3, 4)) / np.log(2.0)
        
        loss = mean_loss + 0.001 * kl
    
    return loss.mean()


# Backward compatibility
noise_estimation_loss = fast_ddpm_loss

def sg_noise_estimation_loss(model, x_available, x_target, t, e, betas):
    """
    Simple noise estimation loss (alias for fast_ddpm_loss with fixed variance)
    For backward compatibility with existing training scripts
    """
    return fast_ddpm_loss(model, x_available, x_target, t, e, betas, var_type='fixed')


def combined_loss(model, x_available, x_target, t, e, betas, alpha=0.8):
    """
    Combined loss function with multiple components
    """
    # Main diffusion loss
    main_loss = fast_ddpm_loss(model, x_available, x_target, t, e, betas, var_type='fixed')
    
    # L1 loss for additional regularization
    with torch.no_grad():
        # Get model prediction
        sqrt_alphas_cumprod = torch.sqrt(1.0 - betas).cumprod(dim=0)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(betas.cumsum(dim=0))
        
        a_t = sqrt_alphas_cumprod[t].view(-1, 1, 1, 1, 1)
        sqrt_one_minus_a_t = sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1, 1)
        
        x_noisy = x_target * a_t + e * sqrt_one_minus_a_t
        
        model_input = x_available.clone()
        model_input[:, 0:1] = x_noisy
        
        pred = model(model_input, t.float())
        if isinstance(pred, tuple):
            pred = pred[0]
        
        # Predict x0
        x0_pred = (x_noisy - sqrt_one_minus_a_t * pred) / a_t
        l1_loss = torch.nn.functional.l1_loss(x0_pred, x_target)
    
    return alpha * main_loss + (1 - alpha) * l1_loss
