import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm
import math


# =============================================================================
# TIME EMBEDDING MODULES (Diffusion-GAN)
# =============================================================================

class SinusoidalPositionalEmbedding(nn.Module):
    """
    Sinusoidal positional embedding for timestep t.
    Standard in DDPM/Diffusion models.
    
    Transforms scalar timestep into high-dimensional embedding:
    PE(pos, 2i) = sin(pos / 10000^(2i/d))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        half_dim = dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
        self.register_buffer('emb', emb, persistent=False)
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: Timestep tensor, shape (batch_size,) - values in [0, T_max]
        Returns:
            Embedding tensor, shape (batch_size, dim)
        """
        t = t.float()
        emb = t[:, None] * self.emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


class TimeEmbeddingMLP(nn.Module):
    """
    MLP to project sinusoidal embedding to feature space.
    Architecture: Linear -> SiLU -> Linear (as per paper recommendation)
    """
    def __init__(self, embedding_dim: int, output_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, output_dim * 4),
            nn.SiLU(),
            nn.Linear(output_dim * 4, output_dim)
        )
    
    def forward(self, t_emb: torch.Tensor) -> torch.Tensor:
        return self.mlp(t_emb)


class ResBlockDiscriminator(nn.Module):
    """
    Bloc pour le Discriminateur : Downsampling (/2) + Convolutions
    IMPORTANT : On utilise spectral_norm pour la stabilité (SNGAN)
    """
    def __init__(self, in_channels, out_channels, downsample=True):
        super().__init__()
        self.downsample = downsample
        
        # Spectral Norm sur toutes les convs du discriminateur !
        self.conv1 = spectral_norm(nn.Conv2d(in_channels, out_channels, 3, 1, 1))
        self.conv2 = spectral_norm(nn.Conv2d(out_channels, out_channels, 3, 1, 1))
        
        self.shortcut = spectral_norm(nn.Conv2d(in_channels, out_channels, 1, 1, 0))

    def forward(self, x):
        # Branche principale
        h = F.relu(x)
        h = self.conv1(h)
        h = F.relu(h)
        h = self.conv2(h)
        if self.downsample:
            h = F.avg_pool2d(h, 2) # On divise la taille par 2

        # Branche raccourci
        x_s = self.shortcut(x)
        if self.downsample:
            x_s = F.avg_pool2d(x_s, 2)

        return h + x_s

class ResNetDiscriminatorBaseline(nn.Module):
    """
    Discriminateur SNGAN avec support Diffusion-GAN.
    
    Modes:
    - use_diffusion=False: D(x) -> score (GAN classique)
    - use_diffusion=True:  D(x, t) -> score (Diffusion-GAN)
    
    Le conditionnement sur t utilise:
    1. Sinusoidal Position Embedding (comme DDPM)
    2. MLP projection (Linear -> SiLU -> Linear)
    3. Addition aux feature maps après extraction CNN
    
    Ref: "At each diffusion timestep... the timestep-dependent discriminator 
         learns to distinguish the diffused real data from the diffused generated data."
    """
    def __init__(
            self,
            channels=3,
            conv_channels: list[int] = None,
            use_diffusion: bool = False,
            time_embedding_dim: int = 128,
            image_size=32
            ):
        super().__init__()
        print(f"[Discriminator] conv_channels={conv_channels}, use_diffusion={use_diffusion}")
        if conv_channels is None:
            conv_channels = [64, 128, 256]
            
        self.use_diffusion = use_diffusion
        self.time_embedding_dim = time_embedding_dim
        self.image_size = image_size
        
        # Calculate number of downsampling blocks needed
        # 32x32 -> 16x16 -> 8x8 -> 4x4 (3 blocks for 32x32)
        # 64x64 -> 32x32 -> 16x16 -> 8x8 -> 4x4 (4 blocks for 64x64)
        num_downsample_blocks = int(torch.log2(torch.tensor(image_size / 4)).item())
        
        # Premier bloc un peu spécial (optimisé)
        self.block1 = nn.Sequential(
            spectral_norm(nn.Conv2d(channels, conv_channels[0], 3, 1, 1)),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Conv2d(conv_channels[0], conv_channels[0], 3, 1, 1)),
            nn.AvgPool2d(2) # First downsample
        )
        
        # Blocs dynamiques basés sur conv_channels
        self.blocks = nn.ModuleList()
        for i in range(len(conv_channels) - 1):
            in_ch = conv_channels[i]
            out_ch = conv_channels[i + 1]
            self.blocks.append(ResBlockDiscriminator(in_ch, out_ch))

        # Add remaining downsampling blocks if needed for larger image sizes
        remaining_blocks = num_downsample_blocks - len(conv_channels)
        last_ch = conv_channels[-1]
        for _ in range(remaining_blocks):
            self.blocks.append(ResBlockDiscriminator(last_ch, last_ch, downsample=True))
        
        # Dernier bloc sans downsample
        self.blocks.append(ResBlockDiscriminator(last_ch, last_ch, downsample=False))

        # =================================================================
        # TIME EMBEDDING (Diffusion-GAN specific)
        # =================================================================
        if self.use_diffusion:
            # Sinusoidal embedding: t -> embedding of dim time_embedding_dim
            self.time_embed = SinusoidalPositionalEmbedding(time_embedding_dim)
            # MLP projection: project to same dim as final feature vector
            self.time_mlp = TimeEmbeddingMLP(time_embedding_dim, last_ch)

        # Classifier final
        # Si diffusion: on concatène features image (last_ch) + time embedding (last_ch)
        if self.use_diffusion:
            self.linear = spectral_norm(nn.Linear(last_ch * 2, 1))
        else:
            self.linear = spectral_norm(nn.Linear(last_ch, 1))

    def forward(self, x, t=None):
        """
        Forward pass du discriminateur.
        
        Args:
            x: Image tensor, shape (B, C, H, W)
            t: Timestep tensor, shape (B,) - REQUIRED if use_diffusion=True
               Values should be in [0, T_max]
        
        Returns:
            Score tensor, shape (B, 1)
        """
        # Extraction des features image
        x = self.block1(x)
        
        for block in self.blocks:
            x = block(x)

        # Global Sum Pooling (Standard dans SNGAN)
        x = torch.sum(x, dim=(2, 3))  # (B, last_ch)
        
        # Conditionnement sur t (Diffusion-GAN)
        if self.use_diffusion:
            if t is None:
                raise ValueError("Timestep t is required when use_diffusion=True")
            
            # Compute time embedding
            t_emb = self.time_embed(t)  # (B, time_embedding_dim)
            t_emb = self.time_mlp(t_emb)  # (B, last_ch)
            
            # Concatenate image features and time embedding
            x = torch.cat([x, t_emb], dim=-1)  # (B, last_ch * 2)
        
        return self.linear(x)