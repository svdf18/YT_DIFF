import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_model import BaseModel
from src.configs.model_config import Config
import math

class ResnetBlock(nn.Module):
    """
    Residual block with optional time embedding conditioning.
    
    Architecture:
    Input -> Conv1 -> Norm -> Act -> (Time Proj) -> Conv2 -> Norm -> + Input -> Output
    
    Args:
        dim (int): Input dimension
        dim_out (int): Output dimension
        time_emb_dim (int, optional): Time embedding dimension for conditioning
    """
    def __init__(self, dim, dim_out, time_emb_dim=None, *, groups=8, dropout=0.1, time_scale=1.0):
        super().__init__()
        
        # Adjust groups to be compatible with channel dimensions
        groups = min(groups, dim)
        while dim % groups != 0:
            groups -= 1
        
        dim_out_groups = min(groups, dim_out)
        while dim_out % dim_out_groups != 0:
            dim_out_groups -= 1
        
        # Pixel normalization with adjusted groups
        self.pixel_norm = nn.GroupNorm(groups, dim)
        
        # Enhanced time embedding MLP
        if time_emb_dim is not None:
            self.mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_emb_dim, dim_out * 2),
                nn.Dropout(p=dropout)
            )
            self.time_scale = time_scale
        else:
            self.mlp = None
            
        # Enhanced convolution blocks with groups
        self.block1 = nn.Sequential(
            nn.Conv2d(dim, dim_out, 3, padding=1, groups=1),
            nn.GroupNorm(dim_out_groups, dim_out),
            nn.SiLU()
        )
        
        self.block2 = nn.Sequential(
            nn.Conv2d(dim_out, dim_out, 3, padding=1, groups=1),
            nn.GroupNorm(dim_out_groups, dim_out),
            nn.Dropout(p=dropout)
        )
        
        # Residual connection with group norm
        self.res_conv = nn.Sequential(
            nn.Conv2d(dim, dim_out, 1),
            nn.GroupNorm(dim_out_groups, dim_out)
        ) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        """
        Forward pass of ResNet block.
        
        Args:
            x (Tensor): Input tensor [B, C, H, W]
            time_emb (Tensor, optional): Time embeddings for conditioning [B, time_emb_dim]
            
        Returns:
            Tensor: Output tensor with residual connection
        """
        h = self.pixel_norm(x)
        h = self.block1(h)
        
        if self.mlp and time_emb is not None:
            time_emb = self.mlp(time_emb)
            # Split time embeddings into scale and shift
            scale, shift = time_emb.chunk(2, dim=1)
            # Add proper broadcasting dimensions
            scale = scale.view(scale.shape[0], -1, 1, 1)
            shift = shift.view(shift.shape[0], -1, 1, 1)
            # Apply scale and shift
            h = h * (1 + scale * self.time_scale) + shift * self.time_scale
            
        h = self.block2(h)
        return h + self.res_conv(x)  # Residual connection

class SelfAttention(nn.Module):
    """
    Self-attention layer for capturing long-range dependencies.
    
    Architecture:
    Input -> LayerNorm -> MultiHeadAttention -> + Input -> FFN -> + Input -> Output
    
    Args:
        channels (int): Number of input channels
    """
    def __init__(self, channels):
        super().__init__()
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        """
        Forward pass of self-attention layer.
        
        Args:
            x (Tensor): Input tensor [B, C, H, W]
            
        Returns:
            Tensor: Self-attended output tensor
        """
        size = x.shape[-2:]  # Store spatial dimensions
        
        # Reshape to sequence for attention
        x = x.view(x.shape[0], x.shape[1], -1).permute(0, 2, 1)  # [B, HW, C]
        
        # Self-attention with skip connection
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        
        # Feedforward network with skip connection
        attention_value = self.ff_self(attention_value) + attention_value
        
        # Reshape back to spatial
        return attention_value.permute(0, 2, 1).view(x.shape[0], x.shape[2], *size)

class UNetEDM(BaseModel):
    """
    U-Net architecture for EDM (Elucidated Diffusion Model).
    
    Features:
    - Residual blocks with time conditioning
    - Self-attention in bottleneck
    - Skip connections between encoder and decoder
    - Time embeddings for diffusion process
    
    Args:
        config (Config): Model configuration object
    """
    def __init__(self, config: Config):
        super().__init__()
        
        # Validate config
        if not config.edm2.enabled:
            raise ValueError("EDM2 must be enabled in configuration")
        
        # Extract config parameters
        self.hidden_dims = config.edm2.hidden_dims
        self.time_embedding_dim = config.edm2.time_embedding_dim
        self.dropout_rate = config.edm2.dropout_rate
        
        # Extract ResNet specific parameters
        self.resnet_groups = config.edm2.resnet_groups
        self.time_scale = config.edm2.resnet_time_scale
        self.use_pixel_norm = config.edm2.use_pixel_norm
        self.mlp_dim_mult = config.edm2.mlp_dim_mult
        self.resnet_dropout = config.edm2.resnet_dropout
        
        # Time embedding network
        self.time_mlp = nn.Sequential(
            nn.Linear(self.time_embedding_dim, self.time_embedding_dim * 4),
            nn.SiLU(),
            nn.Linear(self.time_embedding_dim * 4, self.time_embedding_dim)
        )
        
        # Initialize U-Net components
        self.downs = nn.ModuleList([])  # Downsampling path
        self.ups = nn.ModuleList([])    # Upsampling path
        
        # Build downsampling path
        in_channels = config.audio.in_channels
        for dim in self.hidden_dims:
            self.downs.append(nn.ModuleList([
                ResnetBlock(
                    dim=in_channels, 
                    dim_out=dim, 
                    time_emb_dim=self.time_embedding_dim,
                    groups=self.resnet_groups,
                    dropout=self.dropout_rate,
                    time_scale=self.time_scale
                ),
                ResnetBlock(
                    dim=dim, 
                    dim_out=dim, 
                    time_emb_dim=self.time_embedding_dim,
                    groups=self.resnet_groups,
                    dropout=self.dropout_rate,
                    time_scale=self.time_scale
                ),
                nn.Conv2d(dim, dim, kernel_size=4, stride=2, padding=1)
            ]))
            in_channels = dim
        
        # Middle blocks (at 5x5 resolution)
        mid_dim = self.hidden_dims[-1]
        self.mid_block1 = ResnetBlock(
            dim=mid_dim, 
            dim_out=mid_dim, 
            time_emb_dim=self.time_embedding_dim,
            groups=self.resnet_groups,
            dropout=self.dropout_rate,
            time_scale=self.time_scale
        )
        self.mid_attn = SelfAttention(mid_dim)
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim, self.time_embedding_dim)
        
        # Build upsampling path
        for i, dim in enumerate(reversed(self.hidden_dims[:-1])):
            prev_dim = self.hidden_dims[-(i+1)]
            next_dim = dim
            
            self.ups.append(nn.ModuleList([
                nn.ConvTranspose2d(prev_dim, next_dim, 4, 2, 1),
                nn.Conv2d(prev_dim, next_dim, 1),
                ResnetBlock(next_dim * 2, next_dim, self.time_embedding_dim),
                ResnetBlock(next_dim, next_dim, self.time_embedding_dim)
            ]))
        
        # Add final upsampling to match input resolution
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(self.hidden_dims[0], self.hidden_dims[0], 4, 2, 1),
            nn.Conv2d(self.hidden_dims[0], config.audio.in_channels, 1)
        )
        
        self.logger.info(f"Initialized UNetEDM with {self.count_parameters():,} parameters")

    def forward(self, x, time):
        """
        Forward pass of U-Net.
        
        Args:
            x (Tensor): Input tensor [B, C, H, W]
            time (Tensor): Time embeddings [B]
        """
        # Generate time embeddings using sinusoidal encoding
        half_dim = self.time_embedding_dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=time.device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        
        # Process time embeddings through MLP
        t = self.time_mlp(embeddings)
        
        # Store intermediate outputs for skip connections
        h = x
        residuals = []
        
        # Downsampling
        for block1, block2, downsample in self.downs:
            h = block1(h, t)
            h = block2(h, t)
            residuals.append(h)
            h = downsample(h)
        
        # Middle
        h = self.mid_block1(h, t)
        h = self.mid_attn(h)
        h = self.mid_block2(h, t)
        
        # Upsampling with skip connections
        for upsample, reduce_residual, block1, block2 in self.ups:
            h = upsample(h)
            residual = residuals.pop()
            residual = reduce_residual(residual)
            
            # Add interpolation to match spatial dimensions
            if h.shape[-2:] != residual.shape[-2:]:
                h = F.interpolate(h, size=residual.shape[-2:], mode='bilinear', align_corners=False)
            
            h = torch.cat((h, residual), dim=1)
            h = block1(h, t)
            h = block2(h, t)
        
        # Final upsampling to match input resolution
        print(f"\n--- Final Upsampling ---")
        print(f"Before final up - h shape: {h.shape}")
        h = self.final_up(h)
        print(f"After final up - h shape: {h.shape}")
        
        return h

    def get_device(self):
        """Get the device the model is on"""
        # Get the device of the first parameter of the model
        device = next(self.parameters()).device
        # Return just the base device type without index
        return torch.device(device.type)