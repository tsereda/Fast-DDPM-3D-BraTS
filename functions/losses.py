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
    a = (1-b).cumprod(dim=0)
    # Remove clamping
    a = a.index_select(0, t).view(-1, 1, 1, 1, 1)
    # Remove noise scaling
    # Add noise to target modality: X_t = sqrt(a) * x0 + sqrt(1-a) * noise
    x_noisy = x_target * a.sqrt() + e * (1.0 - a).sqrt()
    # Remove input clamping
    # Prepare model input: replace target channel with noisy version
    model_input = x_available.clone()
    model_input[:, target_idx:target_idx+1] = x_noisy
    # Model predicts noise
    output = model(model_input, t.float())
    # Use e directly in loss
    loss_raw = (e - output).square()
    if keepdim:
        return loss_raw.mean(dim=(1, 2, 3, 4))
    else:
        return loss_raw.mean()


def get_loss_function(loss_type='brats_4to1'):
    """
    Factory function to get the appropriate loss function
    """
    if loss_type == 'brats_4to1':
        return brats_4to1_loss
    else:
        raise ValueError(f"Unknown loss type '{loss_type}'. Available: ['brats_4to1']")


# Export list
__all__ = [
    'brats_4to1_loss',              # Main BraTS 4→1 loss
    'get_loss_function',            # Factory function
    'debug_loss_components',        # Debug function
    'test_minimal_loss',           # Quick test function
]


def debug_loss_components(x_available, x_target, t, e, b, target_idx=0):
    """
    Debug function to analyze loss components without model forward pass
    Returns statistics about each component to identify numerical issues
    
    Args:
        Same as brats_4to1_loss but without model
        
    Returns:
        dict: Statistics about each component
    """
    stats = {}
    
    # Alpha computation
    a = (1-b).cumprod(dim=0)
    stats['alpha_raw_range'] = (a.min().item(), a.max().item())
    
    a = torch.clamp(a, min=1e-8, max=1.0)
    a = a.index_select(0, t).view(-1, 1, 1, 1, 1)
    stats['alpha_selected'] = a.item() if a.numel() == 1 else (a.min().item(), a.max().item())
    
    # Test different noise scalings
    noise_scalings = [1.0, 0.1, 0.01, 0.001]
    stats['noise_scaling_tests'] = {}
    
    for scale in noise_scalings:
        e_scaled = e * scale
        x_noisy = x_target * a.sqrt() + e_scaled * (1.0 - a).sqrt()
        
        stats['noise_scaling_tests'][scale] = {
            'scaled_noise_range': (e_scaled.min().item(), e_scaled.max().item()),
            'noisy_input_range': (x_noisy.min().item(), x_noisy.max().item()),
            'extreme_values': x_noisy.abs().max().item() > 2.0
        }
    
    # Input data stats
    stats['input_data'] = {
        'x_available_range': (x_available.min().item(), x_available.max().item()),
        'x_target_range': (x_target.min().item(), x_target.max().item()),
        'original_noise_range': (e.min().item(), e.max().item())
    }
    
    return stats


def test_minimal_loss():
    """
    Quick test function to verify loss computation works
    Run this to quickly check if your loss function has obvious issues
    """
    print("🧪 Running minimal loss function test...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create minimal test data
    B, C, H, W, D = 1, 4, 8, 8, 8
    x_available = torch.rand(B, C, H, W, D).to(device) * 0.5
    x_target = torch.rand(B, 1, H, W, D).to(device) * 0.5
    e = torch.randn_like(x_target)
    t = torch.randint(0, 10, (B,)).to(device)
    b = torch.linspace(0.0001, 0.02, 10).to(device)
    
    print(f"Test data shapes: x_available={x_available.shape}, x_target={x_target.shape}")
    
    # Debug components
    stats = debug_loss_components(x_available, x_target, t, e, b)
    
    print("\n📊 Component Analysis:")
    print(f"Alpha selected: {stats['alpha_selected']}")
    print(f"Input data ranges:")
    print(f"  x_available: {stats['input_data']['x_available_range']}")
    print(f"  x_target: {stats['input_data']['x_target_range']}")
    print(f"  original_noise: {stats['input_data']['original_noise_range']}")
    
    print(f"\n🎛️  Noise Scaling Tests:")
    for scale, results in stats['noise_scaling_tests'].items():
        status = "🚨 EXTREME" if results['extreme_values'] else "✅ OK"
        print(f"  Scale {scale}: noisy_range={results['noisy_input_range']} {status}")
    
    # Test with simple model
    class SimpleTestModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv3d(4, 1, 3, padding=1)
        
        def forward(self, x, t):
            return self.conv(x)
    
    model = SimpleTestModel().to(device)
    
    # Test loss computation
    try:
        loss = brats_4to1_loss(model, x_available, x_target, t, e, b, target_idx=0)
        print(f"\n📈 Loss Value: {loss.item():.6f}")
        
        if torch.isnan(loss) or torch.isinf(loss):
            print("🚨 LOSS IS NaN OR INFINITY!")
        elif loss.item() > 10:
            print("⚠️  Loss is quite large")
        else:
            print("✅ Loss value seems reasonable")
        
        # Test gradients
        loss.backward()
        total_norm = sum(p.grad.norm(2).item()**2 for p in model.parameters() if p.grad is not None)**0.5
        print(f"📐 Gradient norm: {total_norm:.6f}")
        
        if total_norm > 100:
            print("🚨 GRADIENT EXPLOSION DETECTED!")
        else:
            print("✅ Gradients are reasonable")
            
    except Exception as e:
        print(f"❌ Error in loss computation: {e}")
    
    return stats


# Quick test when file is run directly
if __name__ == '__main__':
    test_minimal_loss()