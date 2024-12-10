from dataclasses import dataclass
from typing import Union, Optional, Literal

import torch

from modules.unets.unet import DualDiffusionUNet, DualDiffusionUNetConfig
from modules.mp_tools import MPConv, MPFourier, mp_cat, mp_silu, mp_sum, normalize, resample

@dataclass
class UNetConfig(DualDiffusionUNetConfig):
    """Configuration for EDM2 UNet"""
    model_channels: int  = 256               # Base multiplier for the number of channels
    logvar_channels: int = 128               # Number of channels for training uncertainty estimation
    channel_mult: list[int] = (1,2,3,4,5)   # Per-resolution multipliers for the number of channels
    channel_mult_noise: Optional[int] = None # Multiplier for noise embedding dimensionality
    channel_mult_emb: Optional[int] = None   # Multiplier for final embedding dimensionality
    channels_per_head: int = 64              # Number of channels per attention head
    num_layers_per_block: int = 2            # Number of resnet blocks per resolution
    label_balance: float = 0.5               # Balance between noise embedding (0) and class embedding (1)
    concat_balance: float = 0.5              # Balance between skip connections (0) and main path (1)
    res_balance: float = 0.3                 # Balance between main branch (0) and residual branch (1)
    attn_balance: float = 0.3                # Balance between main branch (0) and self-attention (1)
    attn_levels: list[int] = (3,4)          # List of resolution levels to use self-attention
    mlp_multiplier: int = 2                  # Multiplier for the number of channels in the MLP
    mlp_groups: int = 8                      # Number of groups for the MLPs

class Block(torch.nn.Module):
    def __init__(self,
        level: int,                          # Resolution level
        in_channels: int,                    # Number of input channels
        out_channels: int,                   # Number of output channels
        emb_channels: int,                   # Number of embedding channels
        flavor: Literal["enc", "dec"] = "enc",
        resample_mode: Literal["keep", "up", "down"] = "keep",
        dropout: float = 0.,                 # Dropout probability
        res_balance: float = 0.3,            # Balance between main branch (0) and residual branch (1)
        attn_balance: float = 0.3,           # Balance between main branch (0) and self-attention (1)
        clip_act: float = 256,               # Clip output activations. None = do not clip
        mlp_multiplier: int = 2,             # Multiplier for the number of channels in the MLP
        mlp_groups: int = 8,                 # Number of groups for the MLPs
        channels_per_head: int = 64,         # Number of channels per attention head
        use_attention: bool = False,         # Use self-attention in this block
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
        self.conv_skip = MPConv(in_channels, out_channels, kernel=(1,1), groups=1)

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
        # Resample input if needed
        x = resample(x, mode=self.resample_mode)

        # Encoder-specific normalization
        if self.flavor == "enc":
            if self.conv_skip is not None:
                x = self.conv_skip(x)
            x = normalize(x, dim=1)  # pixel norm

        # Main convolution path
        y = self.conv_res0(mp_silu(x))

        # Apply embedding conditioning
        c = self.emb_linear(emb, gain=self.emb_gain) + 1.
        y = mp_silu(y * c)

        # Magnitude preserving dropout
        if self.dropout != 0 and self.training == True:
            y = torch.nn.functional.dropout(y, p=self.dropout) * (1. - self.dropout)**0.5

        y = self.conv_res1(y)

        # Decoder-specific skip connection
        if self.flavor == "dec" and self.conv_skip is not None:
            x = self.conv_skip(x)
        x = mp_sum(x, y, t=self.res_balance)
        
        # Self-attention block
        if self.use_attention:
            # Embedding conditioning for attention
            c = self.emb_linear_qk(emb, gain=self.emb_gain_qk) + 1.

            # Generate Q, K, V
            qk = self.attn_qk(x * c)
            qk = qk.reshape(qk.shape[0], self.num_heads, -1, 2, y.shape[2] * y.shape[3])
            q, k = normalize(qk, dim=2).unbind(3)

            v = self.attn_v(x)
            v = v.reshape(v.shape[0], self.num_heads, -1, y.shape[2] * y.shape[3])
            v = normalize(v, dim=2)

            # Compute attention
            y = torch.nn.functional.scaled_dot_product_attention(
                q.transpose(-1, -2),
                k.transpose(-1, -2),
                v.transpose(-1, -2)
            ).transpose(-1, -2)
            y = y.reshape(*x.shape)

            # Apply embedding conditioning to values
            c = self.emb_linear_v(emb, gain=self.emb_gain_v) + 1.
            y = mp_silu(y * c)

            # Project and combine with main path
            y = self.attn_proj(y)
            x = mp_sum(x, y, t=self.attn_balance)

        # Optional activation clipping
        if self.clip_act is not None:
            x = x.clip_(-self.clip_act, self.clip_act)
        return x
