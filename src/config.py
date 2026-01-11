"""
Configuration system for Diffusion-GAN experiments.
Centralizes all hyperparameters and training settings.

Hyperparameters based on the paper: "Diffusion-GAN: Training GANs with Diffusion"
See Appendix G for exact values.
"""

from dataclasses import dataclass, field
from typing import List, Literal, Optional


@dataclass
class Config:
    """Configuration class for GAN training experiments."""
    
    # Experiment identification
    name: str = "default"
    description: str = ""
    
    # Dataset parameters
    dataset_name: Literal['mnist', 'fashion_mnist', 'cifar10', 'celeba'] = 'mnist'
    data_dir: str = './data'
    max_samples: Optional[int] = None  # None = use all data
    batch_size: int = 64
    
    # Model architecture
    variant: str = 'sngan'
    z_dim: int = 128
    image_size: int = 32  # Target image size (32x32 or 64x64)
    gen_channels: List[int] = field(default_factory=lambda: [256, 128, 64])
    disc_channels: List[int] = field(default_factory=lambda: [64, 128, 256])
    
    # =================================================================
    # DIFFUSION PARAMETERS (Ref: Appendix G of Diffusion-GAN paper)
    # =================================================================
    use_diffusion: bool = False
    
    # Diffusion noise schedule (Linear scheduler)
    beta_start: float = 0.0001  # Paper value
    beta_end: float = 0.02      # Paper value
    diffusion_sigma: float = 0.05  # σ for pixels normalized in [-1, 1] (paper value: fixed at 0.05)
    scheduler_type: Literal['linear', 'cosine'] = 'linear'
    
    # Adaptive T schedule parameters
    T_max: int = 1000           # Maximum diffusion timesteps (paper default)
    T_min: int = 5              # Minimum T to avoid trivial diffusion
    T_current: int = 1000       # Current adaptive T (starts at T_max)
    
    # Time embedding
    time_embedding_dim: int = 128
    
    # Adaptive schedule control
    target_d: float = 0.75       # Target discriminator accuracy (overfitting threshold)
    adjust_C: int = 1           # T adjustment step size
    adjust_frequency: int = 4   # Adjust T every N iterations (batches)
    
    # Timestep sampling strategy
    t_sampling: Literal['uniform', 'priority'] = 'uniform'  # 'priority' is slightly better
    
    # Training parameters
    num_epochs: int = 10
    lr_g: float = 0.0002  # Generator learning rate
    lr_d: float = 0.0002  # Discriminator learning rate
    beta1: float = 0.0  # Adam beta1
    beta2: float = 0.9  # Adam beta2
    n_critic: int = 1  # Number of discriminator updates per generator update
    loss_type: Literal['hinge', 'bce', 'wgan'] = 'hinge'  # Loss function type
    
    # FID evaluation
    compute_fid: bool = True  # Enable FID computation during training
    fid_frequency: int = 10  # Compute FID every N epochs
    fid_num_samples: int = 5000  # Number of samples to generate for FID
    
    # Logging
    log_dir: str = './runs'
    save_dir: str = './models'
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.use_diffusion and self.T_max == 0:
            raise ValueError("T_max must be > 0 when use_diffusion=True")
        if self.n_critic < 1:
            raise ValueError("n_critic must be >= 1")
        # Only validate T_min < T_max when diffusion is enabled
        if self.use_diffusion and self.T_min >= self.T_max:
            raise ValueError("T_min must be < T_max when use_diffusion=True")
        if not 0 < self.target_d < 1:
            raise ValueError("target_d must be in (0, 1)")


# Predefined configurations
CONFIGS = {
    'sngan_baseline_cifar10': Config(
        name='sngan_baseline',
        description='Baseline SNGAN without diffusion on CIFAR10',
        dataset_name='cifar10',
        max_samples=10000,
        batch_size=256,
        variant='sngan',
        z_dim=128,
        gen_channels=[64, 32, 16],
        disc_channels=[16, 32, 64],
        use_diffusion=False,
        T_max=0,
        num_epochs=50,
        lr_g=0.0002,
        lr_d=0.0002,
        beta1=0.0,
        beta2=0.9,
        n_critic=5,  # Train discriminator 2 times per generator update
        loss_type='hinge',  # Hinge loss is critical for SNGAN stability
    ),
    
    'sngan_celeba': Config(
        name='sngan_celeba',
        description='SNGAN for CelebA face generation (64x64 images)',
        dataset_name='celeba',
        max_samples=10000,
        batch_size=64,
        variant='sngan',
        z_dim=128,
        image_size=64,
        gen_channels=[128, 64, 32, 16],
        disc_channels=[16, 32, 64, 128],
        use_diffusion=False,
        T_max=0,
        num_epochs=200,
        lr_g=0.0002,
        lr_d=0.0002,
        beta1=0.0,
        beta2=0.9,
        n_critic=2,
        loss_type='hinge',
    ),

    'sngan_celeba_diffusion': Config(
        name='sngan_celeba_diffusion',
        description='SNGAN + Diffusion-GAN for CelebA (64x64) - Paper settings',
        dataset_name='celeba',
        max_samples=10000,
        batch_size=64,
        variant='sngan',
        z_dim=128,
        image_size=64,
        gen_channels=[128, 64, 32, 16],
        disc_channels=[16, 32, 64, 128],
        # Diffusion-GAN parameters
        use_diffusion=True,
        T_max=1000,
        T_min=5,
        T_current=5,
        diffusion_sigma=0.05,  # Paper value (fixed for all experiments)
        scheduler_type='linear',
        target_d=0.75,
        adjust_C=1,  # Ajustement plus agressif de T (était 1, trop lent pour T_max=1000)
        adjust_frequency=4,  # Ajuste tous les 10 appels (5 batches avec n_critic=2)
        t_sampling='uniform',  # 'priority' for better results

        
        beta_start=0.0001,
        beta_end=0.02,
        time_embedding_dim=128,
        # Training
        num_epochs=200,
        lr_g=0.0002,
        lr_d=0.0002,
        beta1=0.0,
        beta2=0.9,
        n_critic=2,
        loss_type='hinge',
    ),
    
}


def get_config(name: str) -> Config:
    """
    Retrieve a predefined configuration by name.
    
    Args:
        name: Configuration name (e.g., 'sngan_baseline', 'diffusion_gan_full')
        
    Returns:
        Config object
        
    Raises:
        KeyError: If configuration name not found
    """
    if name not in CONFIGS:
        available = ', '.join(CONFIGS.keys())
        raise KeyError(f"Configuration '{name}' not found. Available: {available}")
    return CONFIGS[name]


def list_configs():
    """Print all available configurations with descriptions."""
    print("📋 Available configurations:\n")
    for name, config in CONFIGS.items():
        print(f"  • {name}")
        print(f"    {config.description}")
        print(f"    Dataset: {config.dataset_name} ({config.max_samples or 'all'} samples)")
        print(f"    Diffusion: {'Yes (T_max=' + str(config.T_max) + ')' if config.use_diffusion else 'No'}")
        print(f"    Epochs: {config.num_epochs}, n_critic: {config.n_critic}, loss: {config.loss_type}")
        print()
