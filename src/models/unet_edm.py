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
    def __init__(self, dim, dim_out, time_emb_dim=None):
        super().__init__()
        
        # Time embedding projection if provided
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out)
        ) if time_emb_dim else None

        # Main convolution blocks
        self.block1 = nn.Sequential(
            nn.Conv2d(dim, dim_out, 3, padding=1),
            nn.BatchNorm2d(dim_out),
            nn.GELU()
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(dim_out, dim_out, 3, padding=1),
            nn.BatchNorm2d(dim_out)
        )
        
        # Residual connection: if dimensions don't match, project input
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        """
        Forward pass of ResNet block.
        
        Args:
            x (Tensor): Input tensor [B, C, H, W]
            time_emb (Tensor, optional): Time embeddings for conditioning [B, time_emb_dim]
            
        Returns:
            Tensor: Output tensor with residual connection
        """
        h = self.block1(x)
        
        # Add time embedding if provided
        if self.mlp and time_emb is not None:
            h += self.mlp(time_emb)[:, :, None, None]  # Project and broadcast time embedding
            
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
        
        # Time embedding network (sinusoidal embeddings -> MLP)
        self.time_mlp = nn.Sequential(
            nn.Linear(self.time_embedding_dim, self.time_embedding_dim * 4),
            nn.GELU(),
            nn.Linear(self.time_embedding_dim * 4, self.time_embedding_dim)
        )
        
        # Initialize U-Net components
        self.downs = nn.ModuleList([])  # Downsampling path
        self.ups = nn.ModuleList([])    # Upsampling path
        
        # Build downsampling path
        in_channels = config.audio.in_channels
        for dim in self.hidden_dims:
            self.downs.append(nn.ModuleList([
                ResnetBlock(in_channels, dim, self.time_embedding_dim),
                ResnetBlock(dim, dim, self.time_embedding_dim),
                nn.Conv2d(dim, dim, 4, 2, 1)
            ]))
            in_channels = dim
        
        # Middle blocks (at 5x5 resolution)
        mid_dim = self.hidden_dims[-1]
        self.mid_block1 = ResnetBlock(mid_dim, mid_dim, self.time_embedding_dim)
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
            print(f"\n--- Upsampling Step ---")
            print(f"Before upsample - h shape: {h.shape}")
            h = upsample(h)
            print(f"After upsample - h shape: {h.shape}")
            residual = residuals.pop()
            print(f"Residual shape: {residual.shape}")
            residual = reduce_residual(residual)
            print(f"Reduced residual shape: {residual.shape}")
            h = torch.cat((h, residual), dim=1)
            print(f"After concatenation - h shape: {h.shape}")
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