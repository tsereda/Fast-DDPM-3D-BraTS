import torch
import numpy as np

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
    """
    # a: cumulative alpha values (1-b).cumprod() with clamping for numerical stability
    a = (1-b).cumprod(dim=0)
    # Clamp to prevent numerical instability
    a = torch.clamp(a, min=1e-8, max=1.0)
    a = a.index_select(0, t).view(-1, 1, 1, 1, 1)
    
    # 🔥 MUCH MORE AGGRESSIVE FIX: Scale noise way down for [0,1] data
    e_scaled = e * 0.01  # Changed from 0.1 to 0.01 - much smaller noise
    
    # Add noise to target modality: X_t = sqrt(a) * x0 + sqrt(1-a) * noise
    x_noisy = x_target * a.sqrt() + e_scaled * (1.0 - a).sqrt()
    
    # 🔥 ADDITIONAL FIX: Clamp noisy input to prevent extreme values
    x_noisy = torch.clamp(x_noisy, -0.1, 1.1)
    
    # Prepare model input: replace target channel with noisy version
    model_input = x_available.clone()
    model_input[:, target_idx:target_idx+1] = x_noisy
    
    # Model predicts noise
    output = model(model_input, t.float())
    
    # 🔥 ADDITIONAL FIX: Clamp loss to prevent extreme values
    loss_raw = (e_scaled - output).square()
    loss_raw = torch.clamp(loss_raw, max=1.0)  # Cap loss at 1.0
    
    # Use mean for proper loss scaling instead of sum
    if keepdim:
        return loss_raw.mean(dim=(1, 2, 3, 4))
    else:
        return loss_raw.mean()


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
    """
    # Improved alpha computation with better numerical stability
    a = (1-b).cumprod(dim=0)
    # More aggressive clamping for stability
    a = torch.clamp(a, min=1e-6, max=0.9999)
    a = a.index_select(0, t).view(-1, 1, 1, 1, 1)
    
    # 🔥 MUCH MORE AGGRESSIVE FIX: Scale noise way down
    e_scaled = e * 0.01  # Changed from 0.1 to 0.01
    
    # Add noise to target modality: X_t = sqrt(a) * x0 + sqrt(1-a) * noise
    a_sqrt = torch.clamp(a.sqrt(), min=1e-3, max=0.999)
    noise_coeff = torch.clamp((1.0 - a).sqrt(), min=1e-3, max=0.999)
    
    x_noisy = x_target * a_sqrt + e_scaled * noise_coeff
    
    # 🔥 ADDITIONAL FIX: Clamp noisy input
    x_noisy = torch.clamp(x_noisy, -0.1, 1.1)
    
    # Prepare model input: replace target channel with noisy version
    model_input = x_available.clone()
    model_input[:, target_idx:target_idx+1] = x_noisy
    
    # Model predicts noise
    output = model(model_input, t.float())
    
    # Compute MSE loss with clamping
    mse_loss = (e_scaled - output).square()
    mse_loss = torch.clamp(mse_loss, max=1.0)  # Cap loss
    
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