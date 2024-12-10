from typing import Optional, Union, Literal
from dataclasses import dataclass

import torch
import numpy as np

from modules.formats.format import DualDiffusionFormat
from modules.vaes.vae import DualDiffusionVAEConfig, DualDiffusionVAE, IsotropicGaussianDistribution
from modules.mp_tools import MPConv, normalize, resample, mp_silu, mp_sum

@dataclass
class DualDiffusionVAE_EDM2Config(DualDiffusionVAEConfig):
    """Configuration for EDM2 VAE"""
    model_channels: int = 256          # Base multiplier for channels
    channel_mult: list[int] = (1,2,3,4)  # Channel multipliers per resolution
    channel_mult_emb: Optional[int] = None  # Multiplier for embedding dimensionality
    channels_per_head: int = 64        # Channels per attention head
    num_layers_per_block: int = 2      # ResNet blocks per resolution
    res_balance: float = 0.3           # Balance between main and residual branches
    attn_balance: float = 0.3          # Balance between main and attention paths
    mlp_multiplier: int = 1            # MLP channel multiplier
    mlp_groups: int = 1                # MLP group count
    add_mid_block_attention: bool = False  # Add attention in decoder mid-block

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
        self.num_heads = out_channels // channels_per_head
        self.out_channels = out_channels
        self.flavor = flavor
        self.resample_mode = resample_mode
        self.dropout = dropout
        self.res_balance = res_balance
        self.attn_balance = attn_balance
        self.clip_act = clip_act
        
        # Main convolution path
        self.conv_res0 = MPConv(out_channels if flavor == "enc" else in_channels,
                               out_channels * mlp_multiplier, kernel=(3,3), groups=mlp_groups)
        self.conv_res1 = MPConv(out_channels * mlp_multiplier, out_channels, kernel=(3,3), groups=mlp_groups)
        self.conv_skip = MPConv(in_channels, out_channels, kernel=(1,1), groups=1) if in_channels != out_channels else None

        # Embedding projections
        self.emb_gain = torch.nn.Parameter(torch.zeros([]))
        self.emb_linear = MPConv(emb_channels, out_channels * mlp_multiplier,
                                kernel=(1,1), groups=mlp_groups) if emb_channels != 0 else None

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
        # 1. Input Resampling
        # Apply up/downsampling based on resample_mode
        x = resample(x, mode=self.resample_mode)

        # 2. Encoder-specific Processing
        # For encoder blocks, apply skip connection and pixel normalization
        if self.flavor == "enc":
            if self.conv_skip is not None:
                x = self.conv_skip(x)
            x = normalize(x, dim=1)  # pixel norm for stability

        # 3. Main Convolution Path
        # First conv with SiLU activation
        y = self.conv_res0(mp_silu(x))

        # 4. Embedding Conditioning
        # Apply class/noise embedding conditioning
        if self.emb_linear is not None:
            c = self.emb_linear(emb, gain=self.emb_gain) + 1.
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

        # 1. Calculate channel dimensions
        # Base channel count for each resolution level
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

        # 3. Class embedding
        if config.label_dim != 0:
            self.emb_label = MPConv(config.label_dim, cemb, kernel=(1,1))

        # 4. Reconstruction loss parameters
        self.register_buffer('recon_loss_logvar', torch.zeros([]))
        self.register_buffer('latents_out_gain', torch.ones([]))
        self.register_buffer('out_gain', torch.ones([]))

        # 5. Encoder architecture
        self.enc = torch.nn.ModuleDict()
        cout = config.in_channels + 2  # Extra channels: 1 const, 1 pos embedding
        for level, channels in enumerate(cblock):
            # Input convolution at first level
            if level == 0:
                cin = cout
                cout = channels
                self.enc[f"conv_in"] = MPConv(cin, cout, kernel=(3,3))
            else:
                # Downsampling block
                self.enc[f"block{level}_down"] = Block(level, cout, cout, cemb,
                    use_attention=False, flavor="enc", resample_mode="down", **block_kwargs)
            
            # Resolution blocks
            for idx in range(config.num_layers_per_block):
                cin = cout
                cout = channels
                self.enc[f"block{level}_layer{idx}"] = Block(level, cin, cout, cemb,
                    use_attention=False, flavor="enc", **block_kwargs)

        # 6. Latent processing
        self.conv_latents_out = MPConv(cout, config.latent_channels, kernel=(3,3))
        self.conv_latents_in = MPConv(config.latent_channels + 2, cout, kernel=(3,3))  # Extra channels as in encoder

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
