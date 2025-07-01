"""
3D Denoising Functions for Fast-DDPM
Handles 5D tensors (B,C,H,W,D) instead of 4D tensors
"""
import torch


def compute_alpha(beta, t):
    """Compute alpha values for 3D diffusion"""
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    # Shape for 5D: [B, 1, 1, 1, 1]
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1, 1)
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


def ddpm_steps_3d(x, seq, model, b, **kwargs):
    """3D version of DDPM steps"""
    with torch.no_grad():
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        xs = [x]
        x0_preds = []
        betas = b
        
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(betas, t.long())
            atm1 = compute_alpha(betas, next_t.long())
            beta_t = 1 - at / atm1
            x = xs[-1].to('cuda')

            output = model(x, t.float())
            # Handle variance learning outputs
            if isinstance(output, tuple):
                e = output[0]
            else:
                e = output

            x0_from_e = (1.0 / at).sqrt() * x - (1.0 / at - 1).sqrt() * e
            x0_from_e = torch.clamp(x0_from_e, -1, 1)
            x0_preds.append(x0_from_e.to('cpu'))
            
            mean_eps = (
                (atm1.sqrt() * beta_t) * x0_from_e + ((1 - beta_t).sqrt() * (1 - atm1)) * x
            ) / (1.0 - at)

            mean = mean_eps
            noise = torch.randn_like(x)
            mask = 1 - (t == 0).float()
            mask = mask.view(-1, 1, 1, 1, 1)  # 5D mask
            logvar = beta_t.log()
            sample = mean + mask * torch.exp(0.5 * logvar) * noise
            xs.append(sample.to('cpu'))
            
    return xs, x0_preds


def sg_generalized_steps_3d(x, x_img, seq, model, b, **kwargs):
    """3D single-condition generalized steps (image translation, denoising)"""
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
            
            # Model expects concatenated input
            et = model(torch.cat([x_img, xt], dim=1), t)
            
            if isinstance(et, tuple):
                et = et[0]
            
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
            x0_preds.append(x0_t.to('cpu'))
            
            c1 = (
                kwargs.get("eta", 0) * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            )
            c2 = ((1 - at_next) - c1 ** 2).sqrt()

            xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et
            xs.append(xt_next.to('cpu'))

    return xs, x0_preds


def sr_generalized_steps_3d(x, x_bw, x_fw, seq, model, b, **kwargs):
    """3D multi-condition generalized steps (super-resolution)"""
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
            
            # Model expects concatenated input
            et = model(torch.cat([x_bw, x_fw, xt], dim=1), t)
            
            if isinstance(et, tuple):
                et = et[0]
            
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
            x0_preds.append(x0_t.to('cpu'))
            
            c1 = (
                kwargs.get("eta", 0) * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            )
            c2 = ((1 - at_next) - c1 ** 2).sqrt()

            xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et
            xs.append(xt_next.to('cpu'))

    return xs, x0_preds


def unified_4to4_generalized_steps(x, x_available, target_idx, seq, model, b, **kwargs):
    """
    Fixed unified 4→1 sampling for BraTS modality synthesis
    
    Args:
        x: [B, 1, H, W, D] - noisy target modality (random noise initially)
        x_available: [B, 4, H, W, D] - available modalities (zero-padded for target)
        target_idx: index of target modality (0-3)
        seq: timestep sequence
        model: diffusion model (outputs 1 channel, not 4)
        b: beta schedule
    
    Note: Despite the name "4to4", this is actually 4→1 for compatibility with existing code
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