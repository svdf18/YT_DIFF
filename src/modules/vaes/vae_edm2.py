from typing import Optional, Union, Literal
from dataclasses import dataclass

import torch
import numpy as np

from src.modules.formats.format import DualDiffusionFormat
from src.modules.vaes.vae import DualDiffusionVAEConfig, DualDiffusionVAE, IsotropicGaussianDistribution
from src.modules.mp_tools import MPConv, normalize, resample, mp_silu, mp_sum

@dataclass
class DualDiffusionVAE_EDM2Config(DualDiffusionVAEConfig):
    """Configuration for EDM2 VAE"""
    model_channels: int = 96           # Changed from 256 to match training
    channel_mult: list[int] = (1,2,3,5)  # Changed to match training
    channel_mult_emb: Optional[int] = None
    channels_per_head: int = 64
    num_layers_per_block: int = 3      # Changed from 2 to match training
    res_balance: float = 0.3
    attn_balance: float = 0.3
    mlp_multiplier: int = 1
    mlp_groups: int = 1
    add_mid_block_attention: bool = False

class Block(torch.nn.Module):
    """Basic building block for VAE with advanced features"""
    
    def __init__(self,
        level: int,                          # Resolution level
        in_channels: int,                    # Input channels
        out_channels: int,                   # Output channels
        emb_channels: int,                   # Embedding channels
        flavor: Literal["enc", "dec"] = "enc",
        resample_mode: Literal["keep", "up", "down"] = "keep",
        dropout: float = 0.,                 # Dropout probability
        res_balance: float = 0.3,            # Main/residual balance
        attn_balance: float = 0.3,           # Main/attention balance
        clip_act: float = 256,               # Activation clipping
        mlp_multiplier: int = 1,             # MLP channel multiplier
        mlp_groups: int = 1,                 # MLP groups
        channels_per_head: int = 64,         # Attention head size
        use_attention: bool = False,         # Use self-attention
    ) -> None:
        super().__init__()

        self.level = level
        self.use_attention = use_attention
        self.num_heads = max(1, out_channels // channels_per_head)  # Ensure at least 1 head
        self.out_channels = out_channels
        self.flavor = flavor
        self.resample_mode = resample_mode
        self.dropout = dropout
        self.res_balance = res_balance
        self.attn_balance = attn_balance
        self.clip_act = clip_act
        
        # Main convolution path - ensure input/output channels are properly aligned
        conv_in_channels = out_channels if flavor == "enc" else in_channels
        self.conv_res0 = MPConv(conv_in_channels, out_channels * mlp_multiplier, kernel=(3,3), groups=1)
        self.conv_res1 = MPConv(out_channels * mlp_multiplier, out_channels, kernel=(3,3), groups=1)
        
        # Skip connection - only create if channels differ
        self.conv_skip = None if in_channels == out_channels else MPConv(in_channels, out_channels, kernel=(1,1), groups=1)

        # Embedding projections
        self.emb_gain = torch.nn.Parameter(torch.zeros([]))
        self.emb_linear = None if emb_channels == 0 else MPConv(emb_channels, out_channels * mlp_multiplier, kernel=(1,1), groups=1)

        # Attention components
        if self.use_attention:
            self.emb_gain_qk = torch.nn.Parameter(torch.zeros([]))
            self.emb_gain_v = torch.nn.Parameter(torch.zeros([]))
            self.emb_linear_qk = MPConv(emb_channels, out_channels, kernel=(1,1), groups=1) if emb_channels != 0 else None
            self.emb_linear_v = MPConv(emb_channels, out_channels, kernel=(1,1), groups=1) if emb_channels != 0 else None

            self.attn_qk = MPConv(out_channels, out_channels * 2, kernel=(1,1))
            self.attn_v = MPConv(out_channels, out_channels, kernel=(1,1))
            self.attn_proj = MPConv(out_channels, out_channels, kernel=(1,1))

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        # First, ensure emb has same spatial dimensions as x, but with memory optimization
        if emb.shape[-2:] != x.shape[-2:]:
            # Use more memory-efficient interpolation
            target_size = x.shape[-2:]  # Get target size
            if target_size[0] * target_size[1] > emb.shape[-2] * emb.shape[-1]:
                # If upsampling, do it in chunks to save memory
                chunks = []
                chunk_size = 4  # Adjust this value based on your memory constraints
                for i in range(0, emb.shape[1], chunk_size):
                    chunk = emb[:, i:i+chunk_size]
                    chunk = torch.nn.functional.interpolate(
                        chunk,
                        size=target_size,
                        mode='nearest'
                    )
                    chunks.append(chunk)
                emb = torch.cat(chunks, dim=1)
            else:
                # If downsampling, we can do it all at once
                emb = torch.nn.functional.interpolate(
                    emb,
                    size=target_size,
                    mode='nearest'
                )
        
        # Debug prints for input dimensions
        print(f"Block input x shape: {x.shape}")
        print(f"Block input emb shape: {emb.shape}")
        
        # 1. Input Resampling
        x = resample(x, mode=self.resample_mode)
        if self.resample_mode != "keep":
            emb = resample(emb, mode=self.resample_mode)
        print(f"After resample shape: {x.shape}")

        # 2. Encoder-specific Processing
        if self.flavor == "enc":
            if self.conv_skip is not None:
                x = self.conv_skip(x)
            x = normalize(x, dim=1)
        print(f"After encoder processing shape: {x.shape}")

        # 3. Main Convolution Path
        y = self.conv_res0(mp_silu(x))
        print(f"After conv_res0 shape: {y.shape}")

        # 4. Embedding Conditioning
        if self.emb_linear is not None:
            print(f"Before emb_linear - emb shape: {emb.shape}")
            c = self.emb_linear(emb, gain=self.emb_gain) + 1.
            print(f"After emb_linear - c shape: {c.shape}")
            print(f"y shape before multiplication: {y.shape}")
            y = mp_silu(y * c)  # multiplicative conditioning

        # 5. Dropout (with magnitude preservation)
        if self.dropout != 0 and self.training:
            y = torch.nn.functional.dropout(y, p=self.dropout) * (1. - self.dropout)**0.5

        # 6. Second Convolution
        y = self.conv_res1(y)

        # 7. Skip Connection for Decoder
        if self.flavor == "dec" and self.conv_skip is not None:
            x = self.conv_skip(x)
        
        # 8. Residual Connection
        x = mp_sum(x, y, t=self.res_balance)  # balance between main and residual paths
        
        # 9. Self-Attention Block (if enabled)
        if self.use_attention:
            # Embedding conditioning for attention
            if self.emb_linear_qk is not None:
                c = self.emb_linear_qk(emb, gain=self.emb_gain_qk) + 1.

            # Generate Query, Key, Value
            qk = self.attn_qk(x * c)
            qk = qk.reshape(qk.shape[0], self.num_heads, -1, 2, y.shape[2] * y.shape[3])
            q, k = normalize(qk, dim=2).unbind(3)  # normalize for stable attention

            v = self.attn_v(x)
            v = v.reshape(v.shape[0], self.num_heads, -1, y.shape[2] * y.shape[3])
            v = normalize(v, dim=2)

            # Compute scaled dot-product attention
            y = torch.nn.functional.scaled_dot_product_attention(
                q.transpose(-1, -2),  # [B, H, N, C]
                k.transpose(-1, -2),  # [B, H, N, C]
                v.transpose(-1, -2)   # [B, H, N, C]
            ).transpose(-1, -2)
            y = y.reshape(*x.shape)

            # Value conditioning and projection
            if self.emb_linear_v is not None:
                c = self.emb_linear_v(emb, gain=self.emb_gain_v) + 1.
                y = mp_silu(y * c)

            y = self.attn_proj(y)
            x = mp_sum(x, y, t=self.attn_balance)  # balance between main and attention paths

        # 10. Optional Activation Clipping
        if self.clip_act is not None:
            x = x.clip_(-self.clip_act, self.clip_act)
            
        return x

class DualDiffusionVAE_EDM2(DualDiffusionVAE):
    """EDM2-style VAE implementation"""

    def __init__(self, config: DualDiffusionVAE_EDM2Config) -> None:
        super().__init__()
        self.config = config
        
        # Enable gradient checkpointing
        self.gradient_checkpointing = True

        # Add output gain parameter
        self.out_gain = torch.nn.Parameter(torch.ones([]))  # Initialize to 1.0
        
        # Calculate channel dimensions
        cblock = [config.model_channels * mult for mult in config.channel_mult]
        self.num_levels = len(cblock)

        # Embedding dimensions
        cemb = config.model_channels * (config.channel_mult_emb or config.channel_mult[-1])

        # 2. Common block parameters
        block_kwargs = dict(
            dropout=config.dropout,
            res_balance=config.res_balance,
            attn_balance=config.attn_balance,
            mlp_multiplier=config.mlp_multiplier,
            mlp_groups=config.mlp_groups,
            channels_per_head=config.channels_per_head
        )

        # 3. Encoder architecture
        self.enc = torch.nn.ModuleDict()
        # Initial input has 4 channels: 2 for audio + 1 const + 1 pos
        cout = config.in_channels + 2  
        
        # First convolution to match model dimensions
        self.conv_in = MPConv(cout, config.model_channels, kernel=(3,3))
        cout = config.model_channels

        for level, channels in enumerate(cblock):
            # Downsampling block (except first level)
            if level > 0:
                self.enc[f"block{level}_down"] = Block(level, cout, cout, cemb,
                    use_attention=False, flavor="enc", resample_mode="down", **block_kwargs)
            
            # Resolution blocks
            for idx in range(config.num_layers_per_block):
                cin = cout
                cout = channels
                self.enc[f"block{level}_layer{idx}"] = Block(level, cin, cout, cemb,
                    use_attention=False, flavor="enc", **block_kwargs)

        # 6. Latent processing
        self.conv_latents_out = MPConv(cout, config.latent_channels * 2, kernel=(3,3))  # *2 for mu and logvar
        
        # Fix: Add the number of extra channels (const + pos_h) to latent channels
        latent_in_channels = config.latent_channels + 2  # latent channels + const + pos_h
        self.conv_latents_in = MPConv(latent_in_channels, cout, kernel=(3,3))

        # 7. Decoder architecture
        self.dec = torch.nn.ModuleDict()
        for level, channels in reversed(list(enumerate(cblock))):
            # Mid-block with optional attention
            if level == len(cblock) - 1:
                self.dec[f"block{level}_in0"] = Block(level, cout, cout, cemb,
                    flavor="dec", use_attention=config.add_mid_block_attention, **block_kwargs)
                self.dec[f"block{level}_in1"] = Block(level, cout, cout, cemb,
                    flavor="dec", use_attention=config.add_mid_block_attention, **block_kwargs)
            else:
                # Upsampling block
                self.dec[f"block{level}_up"] = Block(level, cout, cout, cemb,
                    flavor="dec", resample_mode="up", **block_kwargs)
            
            # Resolution blocks
            for idx in range(config.num_layers_per_block + 1):
                cin = cout
                cout = channels
                self.dec[f"block{level}_layer{idx}"] = Block(level, cin, cout, cemb,
                    flavor="dec", use_attention=False, **block_kwargs)

        # 8. Output convolution
        self.conv_out = MPConv(cout, config.out_channels, kernel=(3,3))

    def encode(self, x: torch.Tensor) -> IsotropicGaussianDistribution:
        """Encode input to latent distribution"""
        # Add positional and constant channels
        B, C, H, W = x.shape
        # Create pos encoding for both height and width dimensions
        pos_h = torch.linspace(-1, 1, H, device=x.device).view(1, 1, -1, 1).expand(B, 1, H, W)
        pos_w = torch.linspace(-1, 1, W, device=x.device).view(1, 1, 1, -1).expand(B, 1, H, W)
        const = torch.ones(B, 1, H, W, device=x.device)
        
        # Concatenate input with positional encodings and constant
        x = torch.cat([x, const, pos_h], dim=1)
        
        # Initial convolution to get to model dimensions
        x = self.conv_in(x)
        
        # Create embeddings with correct shape - add spatial dimensions
        emb = torch.zeros(B, self.config.model_channels * (self.config.channel_mult_emb or self.config.channel_mult[-1]), 
                          1, 1, device=x.device)  # Added 1,1 for spatial dimensions
        emb = emb.expand(-1, -1, H, W)  # Expand to match spatial dimensions of x
        
        # Run through encoder blocks
        for name, block in self.enc.items():
            x = block(x, emb)
        
        # Get latent parameters
        params = self.conv_latents_out(x)
        mu, logvar = params.chunk(2, dim=1)
        
        return IsotropicGaussianDistribution(mu, logvar)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent to output"""
        # Add positional and constant channels
        B, C, H, W = z.shape
        pos_h = torch.linspace(-1, 1, H, device=z.device).view(1, 1, -1, 1).expand(B, 1, H, W)
        const = torch.ones(B, 1, H, W, device=z.device)
        
        # Only add const and one positional encoding, like in encode
        x = torch.cat([z, const, pos_h], dim=1)
        
        # Create embeddings with correct shape - add spatial dimensions
        emb = torch.zeros(B, self.config.model_channels * (self.config.channel_mult_emb or self.config.channel_mult[-1]), 
                          1, 1, device=z.device)
        emb = emb.expand(-1, -1, H, W)  # Expand to match spatial dimensions of x
        
        # Initial convolution
        x = self.conv_latents_in(x)
        
        # Run through decoder blocks
        for name, block in self.dec.items():
            x = block(x, emb)
        
        # Final convolution
        x = self.conv_out(x)
        return x * self.out_gain

    def get_latent_shape(self, batch_size: int) -> tuple:
        # Return shape of latent space
        pass

    def get_sample_shape(self, batch_size: int) -> tuple:
        # Return shape of output samples
        pass

    def get_class_embeddings(self, labels: torch.Tensor) -> torch.Tensor:
        # Return embeddings for class labels
        pass

    def get_recon_loss_logvar(self) -> torch.Tensor:
        # Return reconstruction loss log variance
        pass

    def get_target_snr(self) -> float:
        # Return target signal-to-noise ratio
        pass
