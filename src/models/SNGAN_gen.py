import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

# =============================================================================
# BLOCS DE BASE (ResBlocks)
# =============================================================================

class ResBlockGenerator(nn.Module):
    """
    Bloc pour le Générateur : Upsampling (x2) + Convolutions
    Structure: BN -> ReLU -> Upsample -> Conv -> BN -> ReLU -> Conv
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.shortcut = nn.Conv2d(in_channels, out_channels, 1, 1, 0)

        # BatchNorm est standard dans le Générateur
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        # Branche principale
        h = self.bn1(x)
        h = F.relu(h)
        h = F.interpolate(h, scale_factor=2) # On double la taille (Upsample)
        h = self.conv1(h)
        h = self.bn2(h)
        h = F.relu(h)
        h = self.conv2(h)

        # Branche raccourci (Skip Connection)
        x_s = F.interpolate(x, scale_factor=2)
        x_s = self.shortcut(x_s)

        return h + x_s # Addition !


# =============================================================================
# ARCHITECTURES COMPLÈTES (BASELINE V1)
# =============================================================================

class ResNetGenerator(nn.Module):
    def __init__(
            self,
            z_dim=128,
            channels=3,
            conv_channels: list[int] = None,
            image_size=32
            ):
        super().__init__()
        if conv_channels is None:
            conv_channels = [256, 128, 64]
        
        self.image_size = image_size
        
        # Calculate number of upsampling blocks needed
        # 4x4 -> 8x8 -> 16x16 -> 32x32 (3 blocks for 32x32)
        # 4x4 -> 8x8 -> 16x16 -> 32x32 -> 64x64 (4 blocks for 64x64)
        num_upsample_blocks = int(torch.log2(torch.tensor(image_size / 4)).item())
        
        # On part d'un vecteur z -> 4x4 x conv_channels[0]
        self.linear = nn.Linear(z_dim, 4 * 4 * conv_channels[0])
        self.start_channels = conv_channels[0]
        self.blocks = nn.ModuleList()

        # Add upsampling blocks based on image size
        for i in range(len(conv_channels) - 1):
            in_ch = conv_channels[i]
            out_ch = conv_channels[i + 1]
            self.blocks.append(ResBlockGenerator(in_ch, out_ch))
        
        # Add remaining blocks if needed for larger image sizes
        remaining_blocks = num_upsample_blocks - len(conv_channels) + 1
        for _ in range(remaining_blocks):
            self.blocks.append(ResBlockGenerator(conv_channels[-1], conv_channels[-1]))

        self.bn = nn.BatchNorm2d(conv_channels[-1])
        self.conv_out = nn.Conv2d(conv_channels[-1], channels, 3, 1, 1)
        self.tanh = nn.Tanh()

    def forward(self, z):
        # Projection et Reshape 4x4
        x = self.linear(z)
        x = x.view(x.size(0), -1, 4, 4)

        for block in self.blocks:
            x = block(x)
        
        x = self.bn(x)
        x = F.relu(x)
        x = self.conv_out(x)
        return self.tanh(x)
        return self.tanh(x)

