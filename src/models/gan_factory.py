"""
Helper pour instancier soit un Diffusion Projection GAN (discriminateur projection
StyleGAN2), soit un SNGAN classique.

Usage rapide
------------
from models.gan_factory import create_gan_models
bundle = create_gan_models(
    variant="projection",  # ou "sngan"
    image_channels=1,
    z_dim=128,
)
gen, disc = bundle.generator, bundle.discriminator
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import NamedTuple, Literal

import torch.nn as nn

from src.models.SNGAN_disc import ResNetDiscriminatorBaseline
from src.models.SNGAN_gen import ResNetGenerator

class ModelBundle(NamedTuple):
    generator: nn.Module
    discriminator: nn.Module


def create_gan_models(
    variant: Literal["sngan"] = "sngan",
    use_diffusion: bool = True,
    image_channels: int = 1,
    z_dim: int = 128,
    gen_channels: list[int] | None = None,
    disc_channels: list[int] | None = None,
    time_embedding_dim: int = 128,
    image_size: int = 32,
) -> ModelBundle:
    """
    Fabrique un couple (generator, discriminator) en fonction du variant demandé.

    Args:
        variant: "sngan" (par défaut).
        use_diffusion: Active le conditionnement temporel (True) ou le désactive pour ablation (False).
        image_channels: Nb de canaux d'entrée/sortie (1 ou 3).
        z_dim: Dimension du latent z pour le générateur.
        gen_channels: Liste de canaux conv pour le générateur (ex: [256, 128, 64]).
                      Par défaut: [256, 128, 64].
        disc_channels: Liste de canaux conv pour le discriminateur (ex: [64, 128, 256]).
                       Par défaut: [64, 128, 256].
        time_embedding_dim: Dimension de l'embedding temporel (utilisé si use_diffusion=True).
        feature_dim: Dimension de projection (non utilisé en SNGAN basique).

    Returns:
        ModelBundle(generator, discriminator)
    """
    if gen_channels is None:
        gen_channels = [256, 128, 64]
    if disc_channels is None:
        disc_channels = [32, 64, 128]
    print("disc_channels:", disc_channels)

    if variant == "sngan":
        generator = ResNetGenerator(
            z_dim=z_dim,
            channels=image_channels,
            conv_channels=gen_channels,
            image_size=image_size
        )
        discriminator = ResNetDiscriminatorBaseline(
            channels=image_channels,
            conv_channels=disc_channels,
            use_diffusion=use_diffusion,
            time_embedding_dim=time_embedding_dim,
            image_size=image_size
        )
        return ModelBundle(generator=generator, discriminator=discriminator)

    raise ValueError("variant doit être 'sngan'")
