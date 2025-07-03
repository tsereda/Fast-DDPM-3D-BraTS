import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

np.bool = np.bool_


def compute_3d_gradient(tensor):
    """Compute 3D gradients for edge preservation"""
    # Sobel-like 3D gradient computation
    grad_x = F.conv3d(tensor, torch.tensor([[[[-1, 0, 1]]]]).float().to(tensor.device), padding=1)
    grad_y = F.conv3d(tensor, torch.tensor([[[[-1], [0], [1]]]]).float().to(tensor.device), padding=1)
    grad_z = F.conv3d(tensor, torch.tensor([[[[-1]], [[0]], [[1]]]]).float().to(tensor.device), padding=1)
    
    return torch.sqrt(grad_x**2 + grad_y**2 + grad_z**2 + 1e-8)


def compute_3d_ssim(x, y, window_size=11, window_sigma=1.5):
    """Simplified 3D SSIM for structural similarity"""
    # Create 3D Gaussian window
    coords = torch.arange(window_size, dtype=torch.float32, device=x.device)
    coords -= window_size // 2
    
    g = torch.exp(-(coords**2) / (2 * window_sigma**2))
    g = g / g.sum()
    
    # Create 3D kernel
    kernel = g.view(1, 1, window_size, 1, 1) * g.view(1, 1, 1, window_size, 1) * g.view(1, 1, 1, 1, window_size)
    kernel = kernel.expand(x.size(1), 1, window_size, window_size, window_size)
    
    # Compute means
    mu_x = F.conv3d(x, kernel, padding=window_size//2, groups=x.size(1))
    mu_y = F.conv3d(y, kernel, padding=window_size//2, groups=y.size(1))
    
    # Compute variances and covariance
    mu_x_sq = mu_x**2
    mu_y_sq = mu_y**2
    mu_xy = mu_x * mu_y
    
    sigma_x_sq = F.conv3d(x**2, kernel, padding=window_size//2, groups=x.size(1)) - mu_x_sq
    sigma_y_sq = F.conv3d(y**2, kernel, padding=window_size//2, groups=y.size(1)) - mu_y_sq
    sigma_xy = F.conv3d(x*y, kernel, padding=window_size//2, groups=x.size(1)) - mu_xy
    
    # SSIM computation
    c1 = 0.01**2
    c2 = 0.03**2
    
    ssim_map = ((2*mu_xy + c1) * (2*sigma_xy + c2)) / ((mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2))
    
    return ssim_map.mean()


class Simple3DPerceptualNet(nn.Module):
    """Lightweight 3D feature extractor for perceptual loss"""
    
    def __init__(self, input_channels=1):
        super().__init__()
        
        # Simple 3D feature extractor - lightweight for memory efficiency
        self.features = nn.Sequential(
            # Layer 1: Basic edge detection
            nn.Conv3d(input_channels, 16, 3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(16, 16, 3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.AvgPool3d(2),
            
            # Layer 2: Pattern detection
            nn.Conv3d(16, 32, 3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 32, 3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.AvgPool3d(2),
            
            # Layer 3: Structure detection
            nn.Conv3d(32, 64, 3, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )
        
        # Initialize as identity-like transformation
        self._init_weights()
    
    def _init_weights(self):
        """Initialize to be close to identity transformation"""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                # Scale down to prevent dominating the main loss
                m.weight.data *= 0.1
    
    def forward(self, x):
        features = []
        x = self.features[:4](x)  # First layer
        features.append(x)
        
        x = self.features[4:8](x)  # Second layer
        features.append(x)
        
        x = self.features[8:](x)   # Third layer
        features.append(x)
        
        return features


def brats_4to1_enhanced_loss(model,
                           x_available: torch.Tensor,
                           x_target: torch.Tensor,
                           t: torch.LongTensor,
                           e: torch.Tensor,
                           b: torch.Tensor,
                           target_idx: int = 0,
                           perceptual_net: nn.Module = None,
                           loss_weights: dict = None,
                           keepdim=False):
    """
    Enhanced BraTS 4→1 loss with perceptual components
    
    Args:
        model: The diffusion model
        x_available: Available modalities [B, 4, H, W, D]
        x_target: Target modality [B, 1, H, W, D]
        t: Timesteps
        e: Noise
        b: Beta schedule
        target_idx: Index of target modality
        perceptual_net: Pre-trained perceptual network
        loss_weights: Dictionary of loss component weights
        keepdim: Whether to keep dimensions for batch-wise loss
    
    Returns:
        Enhanced loss combining MSE, gradient, SSIM, and perceptual losses
    """
    
    # Default loss weights - conservative to not break existing training
    if loss_weights is None:
        loss_weights = {
            'mse': 1.0,        # Main loss
            'gradient': 0.1,   # Edge preservation
            'ssim': 0.1,       # Structural similarity
            'perceptual': 0.05 # Feature matching (small weight)
        }
    
    # Standard diffusion forward process
    a = (1-b).cumprod(dim=0)
    a = a.index_select(0, t).view(-1, 1, 1, 1, 1)
    
    # Add noise to target
    x_noisy = x_target * a.sqrt() + e * (1.0 - a).sqrt()
    
    # Prepare model input
    model_input = x_available.clone()
    model_input[:, target_idx:target_idx+1] = x_noisy
    
    # Model prediction
    predicted_noise = model(model_input, t.float())
    
    # === LOSS COMPONENTS ===
    
    # 1. Main MSE loss (noise prediction)
    mse_loss = (e - predicted_noise).square().mean(dim=(1, 2, 3, 4) if keepdim else None)
    
    # 2. Gradient loss for edge preservation
    # Predict the clean image to compute perceptual losses
    x_pred = (x_noisy - predicted_noise * (1.0 - a).sqrt()) / a.sqrt()
    
    grad_target = compute_3d_gradient(x_target)
    grad_pred = compute_3d_gradient(x_pred)
    gradient_loss = F.mse_loss(grad_pred, grad_target, reduction='none')
    gradient_loss = gradient_loss.mean(dim=(1, 2, 3, 4) if keepdim else None)
    
    # 3. SSIM loss for structural similarity
    ssim_value = compute_3d_ssim(x_pred, x_target)
    ssim_loss = 1 - ssim_value
    
    # 4. Perceptual loss (if network provided)
    perceptual_loss = 0
    if perceptual_net is not None:
        with torch.no_grad():
            # Don't update perceptual net weights
            target_features = perceptual_net(x_target)
        
        pred_features = perceptual_net(x_pred)
        
        # Multi-scale feature matching
        perceptual_loss = 0
        for pred_feat, target_feat in zip(pred_features, target_features):
            perceptual_loss += F.mse_loss(pred_feat, target_feat.detach())
        
        perceptual_loss = perceptual_loss / len(pred_features)
    
    # === COMBINE LOSSES ===
    total_loss = (loss_weights['mse'] * mse_loss + 
                  loss_weights['gradient'] * gradient_loss + 
                  loss_weights['ssim'] * ssim_loss + 
                  loss_weights['perceptual'] * perceptual_loss)
    
    # Return detailed loss info for monitoring
    if keepdim:
        return {
            'total_loss': total_loss,
            'mse_loss': mse_loss,
            'gradient_loss': gradient_loss,
            'ssim_loss': ssim_loss,
            'perceptual_loss': perceptual_loss
        }
    else:
        return total_loss


def get_loss_function(loss_type='brats_4to1_enhanced'):
    """
    Factory function to get the appropriate loss function
    """
    if loss_type == 'brats_4to1':
        return brats_4to1_loss  # Original loss
    elif loss_type == 'brats_4to1_enhanced':
        return brats_4to1_enhanced_loss  # Enhanced loss
    else:
        raise ValueError(f"Unknown loss type '{loss_type}'. Available: ['brats_4to1', 'brats_4to1_enhanced']")


# Keep original loss for backwards compatibility
def brats_4to1_loss(model, x_available, x_target, t, e, b, target_idx=0, keepdim=False):
    """Original BraTS 4→1 loss for backwards compatibility"""
    a = (1-b).cumprod(dim=0)
    a = a.index_select(0, t).view(-1, 1, 1, 1, 1)
    x_noisy = x_target * a.sqrt() + e * (1.0 - a).sqrt()
    model_input = x_available.clone()
    model_input[:, target_idx:target_idx+1] = x_noisy
    output = model(model_input, t.float())
    loss_raw = (e - output).square()
    if keepdim:
        return loss_raw.mean(dim=(1, 2, 3, 4))
    else:
        return loss_raw.mean()


def test_enhanced_loss():
    """Test the enhanced loss function"""
    print("🧪 Testing Enhanced Loss Function...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create test data
    B, C, H, W, D = 2, 4, 16, 16, 16
    x_available = torch.randn(B, C, H, W, D).to(device) * 0.5
    x_target = torch.randn(B, 1, H, W, D).to(device) * 0.5
    e = torch.randn_like(x_target)
    t = torch.randint(0, 1000, (B,)).to(device)
    b = torch.linspace(0.0001, 0.02, 1000).to(device)
    
    # Create perceptual network
    perceptual_net = Simple3DPerceptualNet(input_channels=1).to(device)
    
    # Simple model for testing
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv3d(4, 1, 3, padding=1)
        
        def forward(self, x, t):
            return self.conv(x)
    
    model = TestModel().to(device)
    
    # Test original loss
    original_loss = brats_4to1_loss(model, x_available, x_target, t, e, b, target_idx=0)
    print(f"Original Loss: {original_loss.item():.6f}")
    
    # Test enhanced loss
    enhanced_loss = brats_4to1_enhanced_loss(
        model, x_available, x_target, t, e, b, 
        target_idx=0, perceptual_net=perceptual_net
    )
    print(f"Enhanced Loss: {enhanced_loss.item():.6f}")
    
    # Test with detailed output
    loss_details = brats_4to1_enhanced_loss(
        model, x_available, x_target, t, e, b,
        target_idx=0, perceptual_net=perceptual_net, keepdim=True
    )
    
    print("\n📊 Loss Component Breakdown:")
    for key, value in loss_details.items():
        if torch.is_tensor(value):
            print(f"  {key}: {value.mean().item():.6f}")
        else:
            print(f"  {key}: {value:.6f}")
    
    print("✅ Enhanced loss function test completed!")
    return loss_details


# Export list
__all__ = [
    'brats_4to1_loss',              # Original loss
    'brats_4to1_enhanced_loss',     # Enhanced loss
    'Simple3DPerceptualNet',        # Perceptual network
    'get_loss_function',            # Factory function
    'test_enhanced_loss',           # Test function
]


if __name__ == '__main__':
    test_enhanced_loss()