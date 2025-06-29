"""
3D Denoising Functions for Fast-DDPM
Handles 5D tensors (B,C,H,W,D) instead of 4D tensors
"""
import torch


def compute_alpha(beta, t):
    """Same logic as 2D but for 5D tensors"""
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    # [1, alphas_cumprod] -> shape for 5D: [B, 1, 1, 1, 1]
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
                et = et[0]  # Use mean prediction only
            
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
            x0_preds.append(x0_t.to('cpu'))
            
            # Equation (12) - same math as 2D
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
                e = output[0]  # Use mean prediction only
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


def unified_4to4_generalized_steps(x, x_available, modality_mask, seq, model, b, **kwargs):
    """
    Unified 4→4 sampling for any missing modality
    x: [B, 1, H, W, D] - noisy target modality
    x_available: [B, 4, H, W, D] - available modalities (zero-padded for missing ones)
    modality_mask: [B, 4] - which modalities are available
    """
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
            
            # Concatenate available modalities + noisy target for model input
            # Model expects [B, 4, H, W, D] input (unified 4→4)
            model_input = x_available.to('cuda')
            # Replace one channel with current noisy state
            # For now, use first channel (can be made more sophisticated)
            model_input[:, 0:1] = xt
            
            et = model(model_input, t)
            # Handle variance learning outputs
            if isinstance(et, tuple):
                et = et[0]  # Use mean prediction only
            
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
            x0_preds.append(x0_t.to('cpu'))
            
            # Equation (12)
            c1 = (
                kwargs.get("eta", 0) * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            )
            c2 = ((1 - at_next) - c1 ** 2).sqrt()

            xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et
            xs.append(xt_next.to('cpu'))

    return xs, x0_preds


def unified_4to4_ddpm_steps(x, x_available, modality_mask, seq, model, b, **kwargs):
    """
    Unified 4→4 DDPM sampling
    """
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

            # Unified 4→4 input
            model_input = x_available.to('cuda')
            model_input[:, 0:1] = x  # Replace first channel with noisy target
            
            output = model(model_input, t.float())
            # Handle variance learning outputs
            if isinstance(output, tuple):
                e = output[0]  # Use mean prediction only
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


# Backward compatibility aliases
generalized_steps = generalized_steps_3d
ddpm_steps = ddpm_steps_3d