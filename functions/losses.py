"""
Fast-DDPM loss functions with variance learning
Implements the full Fast-DDPM approach including VLB loss

This script has been corrected for the following issues:
1.  Fixed tensor dimension mismatch by correcting the logic in `fast_ddpm_loss` where `extract` was called incorrectly.
2.  Fixed timestep indexing in `extract`, `sg_noise_estimation_loss`, and `combined_loss` by ensuring the timestep tensor is a LongTensor.
3.  Cleaned up a redundant calculation in `q_posterior_mean_variance`.
"""
import torch
import torch.nn.functional as F
import numpy as np


def compute_alpha(beta, t):
    """Compute alpha values for diffusion timesteps (utility function, not used in main loss)"""
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1, 1)
    return a


def extract(a, t, x_shape):
    """Extract coefficients at specified timesteps and reshape to broadcast with x_shape"""
    b, *_ = t.shape
    # Ensure the index tensor 't' is of type long for the gather operation.
    # The schedule 'a' is a 1D tensor, so we gather along dimension 0.
    out = a.gather(0, t.long().to(a.device))
    # Reshape to be broadcastable with the target tensor x_shape
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def q_posterior_mean_variance(x_start, x_t, alpha_t, alpha_prev):
    """
    Compute posterior mean and variance for diffusion process q(x_{t-1} | x_t, x_0).
    Note: The formulas used here appear to be simplifications of the ones in the original DDPM paper.
    """
    # Compute posterior mean
    posterior_mean = (
        alpha_prev.sqrt() * x_start +
        (1 - alpha_prev).sqrt() * x_t
    ) / (1 - alpha_t).sqrt()

    # Compute posterior variance
    # FIX: The original line `(1 - alpha_prev) * (1 - alpha_t) / (1 - alpha_t)` was redundant.
    # This is arithmetically equivalent and cleaner.
    posterior_variance = (1 - alpha_prev)
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

    # Get values at timestep t using the corrected extract function
    a_t = extract(alphas_cumprod, t, x_target.shape)
    a_prev = extract(alphas_cumprod_prev, t, x_target.shape)
    sqrt_a_t = extract(sqrt_alphas_cumprod, t, x_target.shape)
    sqrt_one_minus_a_t = extract(sqrt_one_minus_alphas_cumprod, t, x_target.shape)

    # Forward diffusion: add noise to target
    x_noisy = x_target * sqrt_a_t + e * sqrt_one_minus_a_t

    # Unified 4->4: replace one channel with noisy target
    model_input = x_available.clone()
    model_input[:, 0:1] = x_noisy

    # Model prediction
    model_output = model(model_input, t.float())
    if var_type in ['learned', 'learned_range']:
        model_mean, model_log_variance = model_output
    else:
        model_mean = model_output
        model_log_variance = None

    # Predict x0 from the model's predicted noise
    x0_pred = (x_noisy - sqrt_one_minus_a_t * model_mean) / sqrt_a_t
    x0_pred = torch.clamp(x0_pred, -1, 1)

    # Get true posterior distribution parameters q(x_{t-1} | x_t, x_0)
    posterior_mean, posterior_variance, posterior_log_variance = \
        q_posterior_mean_variance(x_target, x0_pred, a_t, a_prev)

    # Simple L2 loss for the mean (noise) prediction
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

        # At t=0, use decoder NLL, otherwise use KL divergence
        vb_loss = torch.where((t == 0), decoder_nll, kl)

        # Combine losses (from Improved DDPM paper)
        loss = mean_loss + 0.001 * vb_loss
    elif var_type == 'learned_range':
        # Learn interpolation between fixed small and large variance
        # FIX: `posterior_log_variance` is already computed for the batch timesteps `t`,
        # so it doesn't need to be re-indexed with `extract`. This was the source of the
        # dimension mismatch error.
        min_log = posterior_log_variance
        max_log = extract(torch.log(betas), t, x_target.shape)

        # Model predicts interpolation fraction `v` (renormalized from [-1, 1] to [0, 1])
        frac = (model_log_variance + 1) / 2
        model_log_variance = frac * max_log + (1 - frac) * min_log

        # Compute VLB loss with the interpolated variance
        true_mean = posterior_mean
        true_log_variance = posterior_log_variance

        kl = normal_kl(true_mean, true_log_variance, x0_pred, model_log_variance)
        kl = kl.mean(dim=(1, 2, 3, 4)) / np.log(2.0)

        loss = mean_loss + 0.001 * kl

    return loss.mean()


def sg_noise_estimation_loss(model, x_available, x_target, t, e, betas, keepdim=False):
    """
    Simple noise estimation loss for 3D (fixed variance)

    Args:
        model: 3D diffusion model
        x_available: [B, 4, H, W, D] - available input modalities
        x_target: [B, 1, H, W, D] - target modality
        t: [B] - timesteps
        e: [B, 1, H, W, D] - noise
        betas: beta schedule
        keepdim: whether to keep dimensions
    """
    # FIX: Ensure t is a LongTensor for indexing, as required by index_select.
    t_clamped = torch.clamp(t, 0, len(betas) - 1).long()

    # Get alpha values
    a = (1 - betas).cumprod(dim=0).index_select(0, t_clamped).view(-1, 1, 1, 1, 1)

    # Add noise to target
    x_noisy = x_target * a.sqrt() + e * (1.0 - a).sqrt()

    # Create model input by replacing first channel
    model_input = x_available.clone()
    model_input[:, 0:1] = x_noisy

    # Get model prediction
    output = model(model_input, t.float())

    # Handle tuple output from variance learning models
    if isinstance(output, tuple):
        output = output[0]

    # Compute loss
    if keepdim:
        return (e - output).square().mean(dim=(1, 2, 3, 4))
    else:
        return (e - output).square().mean()


def combined_loss(model, x_available, x_target, t, e, betas, alpha=0.8):
    """
    Combined loss function with multiple components (L2 + L1)
    """
    # Main diffusion loss (simple L2 on noise)
    main_loss = fast_ddpm_loss(model, x_available, x_target, t, e, betas, var_type='fixed')

    # L1 loss for additional regularization
    with torch.no_grad():
        # Get model prediction for x0
        sqrt_alphas_cumprod = torch.sqrt(1.0 - betas).cumprod(dim=0)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(betas.cumsum(dim=0))

        # FIX: Ensure t_clamped is a LongTensor for indexing.
        t_clamped = torch.clamp(t, 0, len(betas) - 1).long()

        a_t_sqrt = sqrt_alphas_cumprod[t_clamped].view(-1, 1, 1, 1, 1)
        sqrt_one_minus_a_t = sqrt_one_minus_alphas_cumprod[t_clamped].view(-1, 1, 1, 1, 1)

        x_noisy = x_target * a_t_sqrt + e * sqrt_one_minus_a_t

        model_input = x_available.clone()
        model_input[:, 0:1] = x_noisy

        pred = model(model_input, t.float())
        if isinstance(pred, tuple):
            pred = pred[0]

        # Predict x0 from the noise prediction
        x0_pred = (x_noisy - sqrt_one_minus_a_t * pred) / a_t_sqrt
        l1_loss = torch.nn.functional.l1_loss(x0_pred, x_target)

    return alpha * main_loss + (1 - alpha) * l1_loss


# Legacy loss functions for backward compatibility
def noise_estimation_loss(model, x0, t, e, b, keepdim=False):
    """Original 2D noise estimation loss for backward compatibility"""
    a = (1-b).cumprod(dim=0).index_select(0, t.long()).view(-1, 1, 1, 1)
    x = x0 * a.sqrt() + e * (1.0 - a).sqrt()
    output = model(x, t.float())
    if isinstance(output, tuple):
        output = output[0]
    if keepdim:
        return (e - output).square().sum(dim=(1, 2, 3))
    else:
        return (e - output).square().sum(dim=(1, 2, 3)).mean(dim=0)


def sr_noise_estimation_loss(model, x_bw, x_md, x_fw, t, e, b, keepdim=False):
    """SR noise estimation loss for backward compatibility"""
    a = (1-b).cumprod(dim=0).index_select(0, t.long()).view(-1, 1, 1, 1)
    x = x_md * a.sqrt() + e * (1.0 - a).sqrt()
    output = model(torch.cat([x_bw, x_fw, x], dim=1), t.float())
    if isinstance(output, tuple):
        output = output[0]
    if keepdim:
        return (e - output).square().sum(dim=(1, 2, 3))
    else:
        return (e - output).square().sum(dim=(1, 2, 3)).mean(dim=0)


# Loss registry for backward compatibility
loss_registry = {
    'simple': noise_estimation_loss,
    'sr': sr_noise_estimation_loss,
    'sg': sg_noise_estimation_loss
}
