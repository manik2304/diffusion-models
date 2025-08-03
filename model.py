import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange

class Time_Embedding(nn.Module):
    def __init__(self, dim: int = 128):
        super().__init__()
        # base_dim should always be even
        assert dim % 2 == 0, "base_dim should be even"

        self.dim = dim
        self.neural_net = nn.Sequential(nn.Linear(dim, dim*4),
                                        nn.SiLU(),
                                        nn.Linear(dim*4, dim))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Input t: picked time steps (batch)
        Output: time embeddings (batch, output_dim)"""

        half_dim = self.dim//2
        device = t.device # Get the device of the input tensor to ensure the output is on the same device
        freq = math.log(10000) / (half_dim-1)
        inv_freq = torch.exp(torch.arange(half_dim, device = device) * -freq)
        angles = t[:, None] * inv_freq[None, :]
        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim = -1) # (batch, dim)

        emb = self.neural_net(emb)
        return emb

class Residual_Block(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_groups: int = 32, time_emb_dim: int = 512):
        super().__init__()

        self.group_norm1 = nn.GroupNorm(num_groups, num_channels = in_channels)
        self.activation1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding='same')
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        self.group_norm2 = nn.GroupNorm(num_groups, num_channels = out_channels)
        self.activation2 = nn.SiLU()
        self.dropout = nn.Dropout(p=0.1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding='same')

        if in_channels != out_channels:
            self.skip_connection = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding='same')
        else:
            self.skip_connection = nn.Identity()

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        """x: input tensor (batch, in_channels, height, width)
        time_emb: time embeddings (batch, time_emb_dim)"""
        h = self.group_norm1(x)
        h = self.activation1(h)
        h = self.conv1(h) # (batch, out_channels, height, width)

        time_emb = self.time_mlp(time_emb)[:, :, None, None] # Reshape to (batch, out_channels, 1, 1)
        h = h + time_emb  # Add time embeddings to the output of the first convolution

        h = self.group_norm2(h)
        h = self.activation2(h)
        h = self.conv2(h)
        h = self.dropout(h)

        return h + self.skip_connection(x) # Skip connection to add input to output
    

class Attention_Block(nn.Module):
    def __init__(self, channels: int, num_heads: int = 1, num_groups: int = 32):
        super().__init__()
        self.num_heads = num_heads
        self.channels = channels
        assert channels % num_heads == 0, "channels must be divisible by num_heads"
        self.head_dim = channels // num_heads
        self.hidden_dim = channels * 4 # Hidden dimension for feedforward network

        self.group_norm1 = nn.GroupNorm(num_groups = num_groups, num_channels = channels)
        self.qkv = nn.Conv1d(channels, channels * 3, kernel_size=1) # Linear is equivalent to Conv1d with kernel_size=1
        self.out_projection = nn.Conv1d(channels, channels, kernel_size=1)

        self.group_norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=channels)
        self.mlp1 = nn.Linear(channels, self.hidden_dim * 2, bias = False)
        self.activation = nn.SiLU()
        self.mlp2 = nn.Linear(self.hidden_dim, channels, bias = False)



    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: input tensor (batch, channels, height, width)"""
        batch, channels, height, width = x.shape
        x_norm = self.group_norm1(x)  # Apply group normalization

        x_norm = rearrange(x_norm, 'b c h w -> b c (h w)')  # Reshape to (B, C, N), N = H*W
        qkv = self.qkv(x_norm) # (B, C*3, N) , query, key, value together
        q, k, v = qkv.chunk(3, dim = 1)  # Split into query, key, value each of shape (B, C, N)

        # reshape to (B, num_heads, head_dim, N)
        q = rearrange(q, 'b (h d) n -> b h n d', h = self.num_heads, d = self.head_dim)
        k = rearrange(k, 'b (h d) n -> b h n d', h = self.num_heads, d = self.head_dim)
        v = rearrange(v, 'b (h d) n -> b h n d', h = self.num_heads, d = self.head_dim)

        # Compute attention scores
        attn_score = torch.einsum('bhnd, bhmd -> bhnm', q, k)/math.sqrt(self.head_dim)  # (B, num_heads, N, N)')
        attn_score_stable = attn_score - attn_score.max(dim=-1, keepdim=True).values  # for numerically stable softmax
        attn_score_stable = torch.softmax(attn_score_stable, dim = -1)  # Softmax over the last dimension
        #attn_score = torch.softmax(attn_score, dim = -1)  # Softmax over the last dimension
        x_attn = torch.einsum('bhnm, bhmd -> bhnd', attn_score_stable, v)  # (B, num_heads, N, head_dim)

        x_attn = rearrange(x_attn, 'b h n d -> b (h d) n')  # Reshape back to (B, C, N)
        x_attn = self.out_projection(x_attn) # (B, C, N)
        #x_attn = rearrange(x_attn, 'b c (h w) -> b c h w', h = height, w = width)  # Reshape back to (B, C, H, W)
        x_attn = self.group_norm2(x_attn)  # Apply group normalization
        x_attn = rearrange(x_attn, 'b c n -> b n c') # (B, N, C)
        x_attn, gate = self.mlp1(x_attn).chunk(2, dim = -1)  # Split into two parts for gating
        x_attn = self.activation(x_attn) * gate # SwiGLU activation
        x_attn = self.mlp2(x_attn) # (B, N, C)

        x_attn = rearrange(x_attn, 'b (h w) c -> b c h w', h = height, w = width)  # Reshape back to (B, C, H, W)

        return x + x_attn # Skip connection to add input to output
    
class Flash_Attention_Block(nn.Module):
    def __init__(self, channels: int, num_heads: int = 1, num_groups: int = 32):
        super().__init__()
        self.num_heads = num_heads
        self.channels = channels
        assert channels % num_heads == 0, "channels must be divisible by num_heads"
        self.head_dim = channels // num_heads
        self.hidden_dim = channels * 4 # Hidden dimension for feedforward network

        self.group_norm1 = nn.GroupNorm(num_groups = num_groups, num_channels = channels)
        self.qkv = nn.Conv1d(channels, channels * 3, kernel_size=1) # Linear is equivalent to Conv1d with kernel_size=1
        self.out_projection = nn.Conv1d(channels, channels, kernel_size=1)

        self.group_norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=channels)
        self.mlp1 = nn.Linear(channels, self.hidden_dim * 2, bias = False)
        self.activation = nn.SiLU()
        self.mlp2 = nn.Linear(self.hidden_dim, channels, bias = False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: input tensor (batch, channels, height, width)"""
        batch, channels, height, width = x.shape
        x_norm = self.group_norm1(x)  # Apply group normalization

        x_norm = rearrange(x_norm, 'b c h w -> b c (h w)')  # Reshape to (B, C, N), N = H*W
        qkv = self.qkv(x_norm) # (B, C*3, N) , query, key, value together
        q, k, v = qkv.chunk(3, dim = 1)  # Split into query, key, value each of shape (B, C, N)

        # reshape to (B, num_heads, head_dim, N)
        q = rearrange(q, 'b (h d) n -> b h n d', h = self.num_heads, d = self.head_dim)
        k = rearrange(k, 'b (h d) n -> b h n d', h = self.num_heads, d = self.head_dim)
        v = rearrange(v, 'b (h d) n -> b h n d', h = self.num_heads, d = self.head_dim)

        
        # Compute attention scores
        x_attn = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False) # Flash attention
        # x_attn shape is (B, num_heads, N, head_dim)

        x_attn = rearrange(x_attn, 'b h n d -> b (h d) n')  # Reshape back to (B, C, N)
        x_attn = self.out_projection(x_attn) # (B, C, N)
        #x_attn = rearrange(x_attn, 'b c (h w) -> b c h w', h = height, w = width)  # Reshape back to (B, C, H, W)
        x_attn = self.group_norm2(x_attn)  # Apply group normalization
        x_attn = rearrange(x_attn, 'b c n -> b n c') # (B, N, C)
        x_attn, gate = self.mlp1(x_attn).chunk(2, dim = -1)  # Split into two parts for gating
        x_attn = self.activation(x_attn) * gate # SwiGLU activation
        x_attn = self.mlp2(x_attn) # (B, N, C)

        x_attn = rearrange(x_attn, 'b (h w) c -> b c h w', h = height, w = width)  # Reshape back to (B, C, H, W)

        return x + x_attn # Skip connection to add input to output

class Residual_Attention_Block(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, 
                 num_groups: int = 32, num_heads: int = 1, time_emb_dim: int = 512):
        super().__init__()
        self.residual_block = Residual_Block(in_channels, out_channels,
                                                num_groups, time_emb_dim)
        
        #self.attention_block = Attention_Block(out_channels, num_heads, num_groups)
        self.attention_block = Flash_Attention_Block(out_channels, num_heads, num_groups)

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        """x: input tensor (batch, in_channels, height, width)
        time_emb: time embeddings (batch, time_emb_dim)"""
        h = self.residual_block(x, time_emb) # Apply residual block (batch, out_channels, height, width)
        h = self.attention_block(h) # Apply attention block (batch, out_channels, height, width)
        return h
    
class Downsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size = 3, stride = 2, padding = 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: input tensor (batch, channels, height, width)"""
        x = self.conv(x) # (batch, channels, height/2, width/2)
        return x
    
class Upsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size = 3, stride = 1, padding = 'same')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: input tensor (batch, channels, height, width)"""
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv(x) # (batch, channels, height*2, width*2)
        return x
    
class Unet(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 3, time_emb_dim: int = 512,
                 base_channels: int = 128, num_groups: int = 32, num_heads: int = 4):
        super().__init__()

        self.time_embedding = Time_Embedding(time_emb_dim)

        self.in_conv = nn.Conv2d(in_channels, base_channels, kernel_size = 3, stride = 1, padding = 'same') # Initial convolution layer

        self.down_path = nn.ModuleList([
            Residual_Block(base_channels, base_channels, num_groups, time_emb_dim), 
            Residual_Block(base_channels, base_channels, num_groups, time_emb_dim),
            Downsample(base_channels), 

            Residual_Attention_Block(base_channels, base_channels * 2, num_groups, num_heads, time_emb_dim), 
            Residual_Attention_Block(base_channels * 2, base_channels * 2, num_groups, num_heads, time_emb_dim),
            Downsample(base_channels * 2),

            Residual_Block(base_channels * 2, base_channels * 2, num_groups, time_emb_dim),
            Residual_Block(base_channels * 2, base_channels * 2, num_groups, time_emb_dim),
            Downsample(base_channels * 2),

            Residual_Block(base_channels * 2, base_channels * 2, num_groups, time_emb_dim),
            Residual_Block(base_channels * 2, base_channels * 2, num_groups, time_emb_dim)
        ])

        self.bottleneck = nn.ModuleList([
            Residual_Attention_Block(base_channels * 2, base_channels * 2, num_groups, num_heads, time_emb_dim),
            Residual_Block(base_channels * 2, base_channels * 2, num_groups, time_emb_dim)
        ])

        self.up_path = nn.ModuleList([
            Residual_Block(base_channels * 4, base_channels * 2, num_groups, time_emb_dim),
            Residual_Block(base_channels * 4, base_channels * 2, num_groups, time_emb_dim),
            Upsample(base_channels * 2),

            Residual_Block(base_channels * 4, base_channels * 2, num_groups, time_emb_dim),
            Residual_Block(base_channels * 4, base_channels * 2, num_groups, time_emb_dim),
            Upsample(base_channels * 2),

            Residual_Attention_Block(base_channels * 4, base_channels * 2, num_groups, num_heads, time_emb_dim),
            Residual_Attention_Block(base_channels * 4, base_channels * 2, num_groups, num_heads, time_emb_dim),
            Upsample(base_channels * 2),
            
            Residual_Block(base_channels * 3, base_channels, num_groups, time_emb_dim),
            Residual_Block(base_channels * 2, base_channels, num_groups, time_emb_dim),
            ])
        self.out_conv = nn.Sequential(
            nn.GroupNorm(num_groups, num_channels = base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, out_channels, kernel_size = 3, stride = 1, padding = 'same')
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """x: input tensor (batch, in_channels, height, width), in_channels = 3, height = width = 32
        t: time steps (batch)"""
        batch_size = x.shape[0]
        time_emb = self.time_embedding(t)

        skips = []
        out = self.in_conv(x)  # Initial convolution (batch, base_channels, height, width)
        skips.append(out)

        for block in self.down_path:
            if isinstance(block, (Residual_Block, Residual_Attention_Block)):
                out = block(out, time_emb)
                skips.append(out)
            elif isinstance(block, Downsample):
                out = block(out)
            else:
                raise ValueError(f"Unknown block type: {type(block)}")
            
        for block in self.bottleneck:
            if isinstance(block, (Residual_Block, Residual_Attention_Block)):
                out = block(out, time_emb)
            else:
                raise ValueError(f"Unknown block type: {type(block)}")
            
        for block in self.up_path:
            if isinstance(block, (Residual_Block, Residual_Attention_Block)):
                skip = skips.pop() 
                out = torch.cat([out, skip], dim = 1)
                out = block(out, time_emb)
            elif isinstance(block, Upsample):
                out = block(out)
            else:
                raise ValueError(f"Unknown block type: {type(block)}")
            
        out = self.out_conv(out)  # Final output convolution (batch, out_channels, height, width)
        return out

class NoiseScheduler:
    def __init__(self, beta_start: float = 1e-4, beta_end: float = 0.02, 
                 num_diffusion_timesteps: int = 1000,
                 device: torch.device = torch.device("cpu")):
        self.T = num_diffusion_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end

        # Precompute schedules (linear schedule )
        self.betas = torch.linspace(self.beta_start, self.beta_end, self.T, device=device)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def get_alpha_bar(self, t: torch.Tensor) -> torch.Tensor:
        """ t: tensor of timesteps (B,) """
        return self.alpha_bars[t]  # used in forward noising
    
    def get_alpha(self, t: torch.Tensor) -> torch.Tensor:
        """ t: tensor of timesteps (B,) """
        return self.alphas[t]
    
    def get_beta(self, t: torch.Tensor) -> torch.Tensor:
        """ t: tensor of timesteps (B,) """
        return self.betas[t]

class AddGaussianNoise(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, t: torch.Tensor, noise_scheduler: NoiseScheduler):
        alpha_bar_t = rearrange(noise_scheduler.get_alpha_bar(t), 'b -> b 1 1 1') # Reshape to match x dimensions (B, 1, 1, 1)
        noise = torch.randn_like(x) # Normal noise with same shape as x
        noisy_x = torch.sqrt(alpha_bar_t) * x + torch.sqrt(1 - alpha_bar_t) * noise
        return noisy_x, noise
    






        
