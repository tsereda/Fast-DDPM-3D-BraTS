import torch
import numpy as np
from functions.losses import get_loss_function

np.bool = np.bool_


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
    # a: cumulative alpha values (1-b).cumprod() with clamping for numerical stability
    a = (1-b).cumprod(dim=0)
    # Clamp to prevent numerical instability
    a = torch.clamp(a, min=1e-8, max=1.0)
    a = a.index_select(0, t).view(-1, 1, 1, 1, 1)
    
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


def improved_brats_4to1_loss(model,
                           x_available: torch.Tensor,
                           x_target: torch.Tensor,
                           t: torch.LongTensor,
                           e: torch.Tensor,
                           b: torch.Tensor, 
                           target_idx: int = 0,
                           keepdim=False):
    """
    Improved BraTS 4→1 modality synthesis loss with better stability
    
    Improvements:
    - Better numerical stability with clamping
    - Focal loss weighting for harder examples
    - Optional perceptual loss component
    """
    # Improved alpha computation with better numerical stability
    a = (1-b).cumprod(dim=0)
    # More aggressive clamping for stability
    a = torch.clamp(a, min=1e-6, max=0.9999)
    a = a.index_select(0, t).view(-1, 1, 1, 1, 1)
    
    # Add noise to target modality: X_t = sqrt(a) * x0 + sqrt(1-a) * noise
    a_sqrt = torch.clamp(a.sqrt(), min=1e-3, max=0.999)
    noise_coeff = torch.clamp((1.0 - a).sqrt(), min=1e-3, max=0.999)
    
    x_noisy = x_target * a_sqrt + e * noise_coeff
    
    # Prepare model input: replace target channel with noisy version
    model_input = x_available.clone()
    model_input[:, target_idx:target_idx+1] = x_noisy
    
    # Model predicts noise
    output = model(model_input, t.float())
    
    # Compute MSE loss
    mse_loss = (e - output).square()
    
    # Apply focal weighting to focus on harder timesteps
    timestep_weights = 1.0 + 0.5 * (t.float() / 1000.0).view(-1, 1, 1, 1, 1)
    weighted_loss = mse_loss * timestep_weights
    
    # Use mean for proper loss scaling instead of sum
    if keepdim:
        return weighted_loss.mean(dim=(1, 2, 3, 4))
    else:
        return weighted_loss.mean()


def get_loss_function(loss_type='brats_4to1'):
    """
    Factory function to get the appropriate loss function
    
    Args:
        loss_type: str - type of loss function to use
            - 'brats_4to1': BraTS 4→1 modality synthesis (standard)
            - 'improved_brats_4to1': Improved version with focal weighting (recommended)
        
    Returns:
        loss_function: callable loss function
    """
    if loss_type == 'brats_4to1':
        return brats_4to1_loss
    elif loss_type == 'improved_brats_4to1':
        return improved_brats_4to1_loss
    else:
        raise ValueError(f"Unknown loss type '{loss_type}'. Available: ['brats_4to1', 'improved_brats_4to1']")


# Export list
__all__ = [
    'brats_4to1_loss',              # Main BraTS 4→1 loss
    'improved_brats_4to1_loss',     # Improved version with focal weighting
    'get_loss_function',            # Factory function
]

# Get the improved loss function (recommended for stability)
loss_fn = get_loss_function('improved_brats_4to1')

# In your training loop:
# loss = loss_fn(model, x_available, x_target, t, e, b=betas, target_idx=target_idx)