from typing import Optional, Union, Literal
from dataclasses import dataclass

import torch
import numpy as np

from src.modules.formats.format import DualDiffusionFormat
from src.modules.vaes.vae import DualDiffusionVAEConfig, DualDiffusionVAE, IsotropicGaussianDistribution
from src.modules.mp_tools import MPConv, normalize, resample, mp_silu, mp_sum

@dataclass
class DualDiffusionVAE_EDM2Config:
    # EDM2-specific
    target_snr: float = 20.0  # Reduced from 31.98
    res_balance: float = 0.4  # Increased from 0.3
    attn_balance: float = 0.4  # Increased from 0.3
    
    # Model architecture
    in_channels: int = 2
    out_channels: int = 2
    latent_channels: int = 4
    model_channels: int = 128  # Reduced from 256
    channel_mult: tuple[int, ...] = (1, 2, 4, 5)  # Changed from (1, 2, 4, 8)
    num_layers_per_block: int = 4
    
    # New EDM2 features
    mlp_multiplier: int = 2  # Increased from 1
    mlp_groups: int = 1
    channel_mult_emb: Optional[int] = None   # Embedding dimensionality multiplier
    add_mid_block_attention: bool = True  # Enabled
    
    # Modern improvements
    use_sdp_attention: bool = True
    channels_per_head: int = 32  # Reduced from 64
    dropout: float = 0.0
    clip_act: Optional[float] = 256.0
    
    # Training settings
    enable_compilation: bool = True
    compile_mode: str = "reduce-overhead"
    
    num_frequencies: int = 256  # Match spectrogram frequency bins

class Block(torch.nn.Module):
    def __init__(self,
        level: int,                          # Resolution level
        in_channels: int,                    # Number of input channels
        out_channels: int,                   # Number of output channels
        emb_channels: int,                   # Number of embedding channels
        flavor: Literal["enc", "dec"] = "enc",
        resample_mode: Literal["keep", "up", "down"] = "keep",
        dropout: float = 0.,                 # Dropout probability
        res_balance: float = 0.3,            # Balance between main and residual
        attn_balance: float = 0.3,           # Balance between main and attention
        clip_act: float = 256,               # Clip output activations
        mlp_multiplier: int = 1,             # MLP channel multiplier
        mlp_groups: int = 1,                 # MLP group count
        channels_per_head: int = 64,         # Channels per attention head
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
        
        # MLP path with groups
        self.conv_res0 = MPConv(
            out_channels if flavor == "enc" else in_channels,
            out_channels * mlp_multiplier, 
            kernel=(3,3), 
            groups=mlp_groups
        )
        self.conv_res1 = MPConv(
            out_channels * mlp_multiplier,
            out_channels,
            kernel=(3,3),
            groups=mlp_groups
        )
        
        # Skip connection
        self.conv_skip = None if in_channels == out_channels else \
                        MPConv(in_channels, out_channels, kernel=(1,1))
        
        # Embedding gains and projections
        self.emb_gain = torch.nn.Parameter(torch.zeros([]))
        self.emb_linear = MPConv(
            emb_channels,
            out_channels * mlp_multiplier,
            kernel=(),
            groups=mlp_groups
        ) if emb_channels != 0 else None
        
        # Attention components
        if self.use_attention:
            # Attention embedding gains
            self.emb_gain_qk = torch.nn.Parameter(torch.zeros([]))
            self.emb_gain_v = torch.nn.Parameter(torch.zeros([]))
            
            # Attention embedding projections
            self.emb_linear_qk = MPConv(
                emb_channels,
                out_channels,
                kernel=(1,1),
                groups=1
            ) if emb_channels != 0 else None
            
            self.emb_linear_v = MPConv(
                emb_channels,
                out_channels,
                kernel=(1,1),
                groups=1
            ) if emb_channels != 0 else None
            
            # Attention layers
            self.attn_qk = MPConv(out_channels, out_channels * 2, kernel=(1,1))
            self.attn_v = MPConv(out_channels, out_channels, kernel=(1,1))
            self.attn_proj = MPConv(out_channels, out_channels, kernel=(1,1))

    def forward(self, x: torch.Tensor, emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Handle resampling
        x = resample(x, mode=self.resample_mode)
        
        # Encoder-specific normalization
        if self.flavor == "enc":
            if self.conv_skip is not None:
                x = self.conv_skip(x)
            x = normalize(x, dim=1)  # pixel norm
        
        # MLP path
        y = self.conv_res0(mp_silu(x))
        
        # Apply embedding modulation
        if emb is not None and self.emb_linear is not None:
            c = self.emb_linear(emb, gain=self.emb_gain) + 1.
            y = mp_silu(y * c.unsqueeze(-1).unsqueeze(-1))
        
        # Magnitude-preserving dropout
        if self.dropout > 0 and self.training:
            y = torch.nn.functional.dropout(y, p=self.dropout) * (1. - self.dropout)**0.5
        
        y = self.conv_res1(y)
        
        # Decoder-specific skip connection
        if self.flavor == "dec" and self.conv_skip is not None:
            x = self.conv_skip(x)
        
        # Residual connection with balance
        x = mp_sum(x, y, t=self.res_balance)
        
        # Self-attention block
        if self.use_attention and emb is not None:
            # Query-Key embedding modulation
            c = self.emb_linear_qk(emb, gain=self.emb_gain_qk) + 1. if self.emb_linear_qk is not None else 1.
            
            # Generate Q, K, V
            qk = self.attn_qk(x * c)
            qk = qk.reshape(qk.shape[0], self.num_heads, -1, 2, x.shape[2] * x.shape[3])
            q, k = normalize(qk, dim=2).unbind(3)
            
            v = self.attn_v(x)
            v = v.reshape(v.shape[0], self.num_heads, -1, x.shape[2] * x.shape[3])
            v = normalize(v, dim=2)
            
            # Scaled dot-product attention
            y = torch.nn.functional.scaled_dot_product_attention(
                q.transpose(-1, -2),
                k.transpose(-1, -2),
                v.transpose(-1, -2)
            ).transpose(-1, -2)
            
            y = y.reshape(*x.shape)
            
            # Value embedding modulation
            c = self.emb_linear_v(emb, gain=self.emb_gain_v) + 1. if self.emb_linear_v is not None else 1.
            y = mp_silu(y * c)
            
            # Project and apply attention balance
            y = self.attn_proj(y)
            x = mp_sum(x, y, t=self.attn_balance)
        
        # Optional activation clipping
        if self.clip_act is not None:
            x = x.clip(-self.clip_act, self.clip_act)
            
        return x

class DualDiffusionVAE_EDM2(DualDiffusionVAE):
    """EDM2-style VAE implementation"""

    def __init__(self, config: DualDiffusionVAE_EDM2Config) -> None:
        super().__init__()
        self.config = config
        
        # EDM2 SNR-based scaling
        target_noise_std = (1 / (config.target_snr**2 + 1))**0.5
        target_sample_std = (1 - target_noise_std**2)**0.5
        self.latents_out_gain = torch.nn.Parameter(torch.tensor(target_sample_std))
        self.out_gain = torch.nn.Parameter(torch.ones([]))
        
        # Calculate channels for each level
        cblock = [config.model_channels * mult for mult in config.channel_mult]
        
        # Common block parameters
        block_kwargs = {
            "dropout": config.dropout,
            "res_balance": config.res_balance,
            "attn_balance": config.attn_balance,
            "clip_act": config.clip_act,
            "mlp_multiplier": config.mlp_multiplier,
            "mlp_groups": config.mlp_groups,
            "channels_per_head": config.channels_per_head,
            "use_attention": config.add_mid_block_attention
        }
        
        # Build encoder and decoder
        self.encoder = self._build_encoder(cblock, block_kwargs)
        self.decoder = self._build_decoder(cblock, block_kwargs)
        
        # Enable compilation if requested
        if config.enable_compilation:
            self.encoder = torch.compile(self.encoder, mode=config.compile_mode, fullgraph=True)
            self.decoder = torch.compile(self.decoder, mode=config.compile_mode, fullgraph=True)

    def _build_encoder(self, cblock: list[int], block_kwargs: dict) -> torch.nn.Sequential:
        """Build encoder network"""
        layers = []
        in_ch = self.config.in_channels
        
        # Add encoder blocks with progressive downsampling
        for level, out_ch in enumerate(cblock):
            layers.append(Block(
                level=level,
                in_channels=in_ch,
                out_channels=out_ch,
                emb_channels=0,  # No embedding for encoder
                flavor="enc",
                resample_mode="down" if level > 0 else "keep",
                **block_kwargs
            ))
            in_ch = out_ch
        
        # Final projection to latent channels
        layers.append(torch.nn.Conv2d(in_ch, self.config.latent_channels, kernel_size=1))
        
        return torch.nn.Sequential(*layers)

    def _build_decoder(self, cblock: list[int], block_kwargs: dict) -> torch.nn.Sequential:
        """Build decoder network"""
        layers = []
        in_ch = self.config.latent_channels
        
        # Initial projection from latent space
        layers.append(torch.nn.Conv2d(in_ch, cblock[-1], kernel_size=1))
        
        # Add decoder blocks with progressive upsampling
        for level, out_ch in enumerate(reversed(cblock[:-1])):
            layers.append(Block(
                level=len(cblock) - level - 1,
                in_channels=cblock[-level-1],
                out_channels=out_ch,
                emb_channels=0,  # No embedding for decoder
                flavor="dec",
                resample_mode="up",
                **block_kwargs
            ))
        
        # Final projection to output channels
        layers.append(torch.nn.Conv2d(cblock[0], self.config.out_channels, kernel_size=1))
        
        return torch.nn.Sequential(*layers)

    def encode(self, x: torch.Tensor) -> IsotropicGaussianDistribution:
        """Encode input to latent distribution"""
        # Add EDM2 noise scaling
        h = self.encoder(x)
        mu = h * self.latents_out_gain
        
        # Calculate logvar based on target SNR
        logvar = torch.full_like(mu, np.log(1 / (self.config.target_snr**2 + 1)))
        
        return IsotropicGaussianDistribution(mu, logvar)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent to output"""
        return self.decoder(z) * self.out_gain

    def get_latent_shape(self, batch_size: int) -> tuple:
        """Return shape of latent space"""
        # Calculate latent dimensions based on actual spectrogram size
        h = self.config.num_frequencies // (2 ** (len(self.config.channel_mult) - 1))
        w = 46 // (2 ** (len(self.config.channel_mult) - 1))  # Match time dimension
        return (batch_size, self.config.latent_channels, h, w)

    def get_sample_shape(self, batch_size: int) -> tuple:
        """Return shape of output samples"""
        # Match the spectrogram shape from format processor
        return (batch_size, self.config.out_channels,  # [B, 2, 256, 46]
                self.config.num_frequencies, 46)  # Time steps from spectrogram

    def get_class_embeddings(self, labels: torch.Tensor) -> torch.Tensor:
        # Return embeddings for class labels
        pass

    def get_recon_loss_logvar(self) -> torch.Tensor:
        # Return reconstruction loss log variance
        pass

    def get_target_snr(self) -> float:
        """Return target signal-to-noise ratio"""
        return self.config.target_snr
