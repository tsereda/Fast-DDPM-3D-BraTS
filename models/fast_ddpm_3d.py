import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_timestep_embedding(timesteps, embedding_dim):
    """Sinusoidal timestep embeddings"""
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:
        emb = F.pad(emb, (0, 1, 0, 0))
    return emb


def nonlinearity(x):
    # 🔥 FIXED: Use ReLU instead of Swish to prevent gradient explosion
    return F.relu(x)


def Normalize(in_channels, num_groups=32):
    """
    Improved GroupNorm with better group selection for 3D medical imaging
    
    Uses a more sophisticated approach to select the number of groups:
    - Ensures sufficient channels per group (minimum 4)
    - Falls back to fewer groups if needed
    - Uses LayerNorm for very small channel counts
    """
    if in_channels == 0:
        return nn.Identity()
    
    # For very small channel counts, use LayerNorm instead
    if in_channels <= 8:
        # LayerNorm over spatial dimensions for medical images
        return nn.GroupNorm(num_groups=1, num_channels=in_channels, eps=1e-5, affine=True)
    
    # Ensure minimum 4 channels per group for stable statistics
    min_channels_per_group = 4
    max_groups = in_channels // min_channels_per_group
    
    # Use standard approach but with better fallback
    num_groups = min(num_groups, max_groups, in_channels)
    
    # Find largest divisor that gives reasonable group size
    while num_groups > 1 and in_channels % num_groups != 0:
        num_groups -= 1
    
    # Ensure we have at least 2 groups for the benefits of GroupNorm
    if num_groups == 1 and in_channels >= 8:
        # Find the best divisor close to desired groups
        possible_groups = [i for i in range(2, min(33, in_channels + 1)) if in_channels % i == 0]
        if possible_groups:
            # Choose group count closest to 32 or in_channels//4, whichever is smaller
            target = min(32, in_channels // 4)
            num_groups = min(possible_groups, key=lambda x: abs(x - target))
    
    return nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-5, affine=True)


class Upsample3D(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = nn.Conv3d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample3D(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = nn.Conv3d(in_channels, in_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        if self.with_conv:
            x = self.conv(x)
        else:
            x = F.avg_pool3d(x, kernel_size=2, stride=2)
        return x


class ResnetBlock3D(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout, temb_channels=512):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if temb_channels > 0:
            self.temb_proj = nn.Linear(temb_channels, out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            else:
                self.nin_shortcut = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None, None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + h


class AttnBlock3D(nn.Module):
    """3D self-attention block for medical image diffusion"""
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = nn.Conv3d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = nn.Conv3d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = nn.Conv3d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = nn.Conv3d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # Compute attention
        b, c, h, w, d = q.shape
        q = q.reshape(b, c, h*w*d)
        q = q.permute(0, 2, 1)   # b,hw,c
        k = k.reshape(b, c, h*w*d) # b,c,hw
        w_ = torch.bmm(q, k)     # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c)**(-0.5))
        w_ = F.softmax(w_, dim=2)

        # Apply attention
        v = v.reshape(b, c, h*w*d)
        w_ = w_.permute(0, 2, 1)   # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v, w_)     # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b, c, h, w, d)

        h_ = self.proj_out(h_)

        return x + h_

class FastDDPM3D(nn.Module):
    """3D Fast-DDPM for unified 4->1 BraTS modality synthesis"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        ch = config.model.ch
        out_ch = config.model.out_ch
        ch_mult = tuple(config.model.ch_mult)
        num_res_blocks = config.model.num_res_blocks
        dropout = config.model.dropout
        in_channels = config.model.in_channels
        resolution = config.data.crop_size[0]  # Updated to use crop_size
        resamp_with_conv = config.model.resamp_with_conv
        
        # Always use fixed variance for 3D
        self.out_channels = out_ch
        
        # Model parameters
        self.ch = ch
        self.temb_ch = self.ch * 4
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        # Time embedding
        self.temb = nn.Module()
        self.temb.dense = nn.ModuleList([
            nn.Linear(self.ch, self.temb_ch),
            nn.Linear(self.temb_ch, self.temb_ch),
        ])

        # Get attention resolutions
        attn_resolutions = getattr(config.model, 'attn_resolutions', [])

        # Downsampling
        self.conv_in = nn.Conv3d(in_channels, self.ch, kernel_size=3, stride=1, padding=1)

        curr_res = resolution
        in_ch_mult = (1,) + ch_mult
        self.down = nn.ModuleList()
        
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock3D(
                    in_channels=block_in,
                    out_channels=block_out,
                    temb_channels=self.temb_ch,
                    dropout=dropout
                ))
                block_in = block_out
                # Add attention if resolution matches
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock3D(block_in))
                else:
                    attn.append(nn.Identity())
                    
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample3D(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        # Middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock3D(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout
        )
        self.mid.attn_1 = AttnBlock3D(block_in)
        self.mid.block_2 = ResnetBlock3D(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout
        )

        # Upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            skip_in = ch * ch_mult[i_level]
            
            for i_block in range(self.num_res_blocks + 1):
                if i_block == self.num_res_blocks:
                    skip_in = ch * in_ch_mult[i_level]
                
                block.append(ResnetBlock3D(
                    in_channels=block_in + skip_in,
                    out_channels=block_out,
                    temb_channels=self.temb_ch,
                    dropout=dropout
                ))
                block_in = block_out
                # Add attention if resolution matches
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock3D(block_in))
                else:
                    attn.append(nn.Identity())
                    
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample3D(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up)

        # Output
        self.norm_out = Normalize(block_in)
        self.conv_out = nn.Conv3d(block_in, self.out_channels, kernel_size=3, stride=1, padding=1)
        
        # 🔥 CRITICAL: Apply proper weight initialization AFTER building the model
        self._init_weights()

    def forward(self, x, t):
        # x shape: [B, 4, H, W, D] for unified 4->1 input
        assert x.shape[1] == self.in_channels
        
        # Timestep embedding
        temb = get_timestep_embedding(t, self.ch)
        temb = self.temb.dense[0](temb)
        temb = nonlinearity(temb)
        temb = self.temb.dense[1](temb)

        # Downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # Middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # Upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](
                    torch.cat([h, hs.pop()], dim=1), temb)
                h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # Output - 🔥 CRITICAL FIX: NO sigmoid for diffusion models
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        
        # 🔥 REMOVED sigmoid activation - diffusion models predict noise, not images
        return h
    
    def _init_weights(self):
        """
        🔥 CRITICAL FIX: Proper weight initialization for 3D medical imaging
        This prevents gradient explosion in large 3D models
        """
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv3d):
                # He initialization for Conv3d
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
                    
            elif isinstance(module, nn.Linear):
                # Xavier initialization for Linear layers
                nn.init.xavier_normal_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
                    
            elif isinstance(module, nn.GroupNorm):
                # Proper GroupNorm initialization
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
        
        # Special initialization for output layer - start very small
        if hasattr(self, 'conv_out'):
            nn.init.normal_(self.conv_out.weight, mean=0.0, std=0.001)
            if self.conv_out.bias is not None:
                nn.init.constant_(self.conv_out.bias, 0)