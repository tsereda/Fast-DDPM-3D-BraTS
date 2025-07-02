"""
3D Denoising Functions for Fast-DDPM
Handles 5D tensors (B,C,H,W,D) instead of 4D tensors
"""
import torch


def compute_alpha(beta, t):
    """Compute alpha values for 3D diffusion"""
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    # Shape for 5D: [B, 1, 1, 1, 1] with clamping for numerical stability
    a = (1 - beta).cumprod(dim=0)
    # Clamp to prevent numerical instability
    a = torch.clamp(a, min=1e-8, max=1.0)
    a = a.index_select(0, t + 1).view(-1, 1, 1, 1, 1)
    return a


def generalized_steps_3d(x, seq, model, b, **kwargs):
    """3D version of generalized steps for Fast-DDPM"""
    with torch.no_grad():
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        x0_preds = []
        xs = [x]
        
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(b, t.long())
            at_next = compute_alpha(b, next_t.long())
            xt = xs[-1].to('cuda')
            et = model(xt, t)
            
            # Handle variance learning outputs
            if isinstance(et, tuple):
                et = et[0]
            
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
            x0_preds.append(x0_t.to('cpu'))
            
            # Equation (12) - same math as 2D but for 3D
            c1 = (
                kwargs.get("eta", 0) * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            )
            c2 = ((1 - at_next) - c1 ** 2).sqrt()

            xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et
            xs.append(xt_next.to('cpu'))

    return xs, x0_preds


def unified_4to1_generalized_steps_3d(x, x_available, target_idx, seq, model, b, **kwargs):
    """
    Unified 4→1 sampling for BraTS modality synthesis
    
    Args:
        x: [B, 1, H, W, D] - noisy target modality (random noise initially)
        x_available: [B, 4, H, W, D] - available modalities (zero-padded for target)
        target_idx: index of target modality (0-3)
        seq: timestep sequence
        model: diffusion model (outputs 1 channel, not 4)
        b: beta schedule
    
    Returns 4 input modalities → 1 target modality
    """
    with torch.no_grad():
        device = x.device
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        x0_preds = []
        xs = [x]
        
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * i).to(device)
            next_t = (torch.ones(n) * j).to(device)
            at = compute_alpha(b, t.long())
            at_next = compute_alpha(b, next_t.long())
            xt = xs[-1].to(device)
            
            # Create model input by replacing target channel with noisy version
            model_input = x_available.to(device)
            model_input[:, target_idx:target_idx+1] = xt
            
            # Model predicts noise for the target modality [B, 1, H, W, D]
            predicted_noise = model(model_input, t)
            
            if isinstance(predicted_noise, tuple):
                predicted_noise = predicted_noise[0]
            
            # Predict x0: x0 = (x_t - sqrt(1-at) * noise) / sqrt(at)
            x0_t = (xt - predicted_noise * (1 - at).sqrt()) / at.sqrt()
            x0_preds.append(x0_t.to('cpu'))
            
            if j < 0:  # Final step
                xt_next = x0_t
            else:
                # DDIM step
                c1 = (
                    kwargs.get("eta", 0) * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
                )
                c2 = ((1 - at_next) - c1 ** 2).sqrt()
                
                xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * predicted_noise
            
            xs.append(xt_next.to('cpu'))

    return xs, x0_preds