import torch
import numpy as np


def get_beta_schedule(schedule_type: str, beta_start: float, beta_end: float, T: int) -> torch.Tensor:
    """
    Compute beta schedule for diffusion process.
    
    Args:
        schedule_type: 'linear' or 'cosine'
        beta_start: Starting beta value (paper: 0.0001)
        beta_end: Ending beta value (paper: 0.02)
        T: Number of timesteps
    
    Returns:
        betas: Tensor of shape (T,)
    """
    if schedule_type == 'linear':
        return torch.linspace(beta_start, beta_end, T)
    elif schedule_type == 'cosine':
        # Cosine schedule from "Improved DDPM"
        steps = T + 1
        s = 0.008
        x = torch.linspace(0, T, steps)
        alphas_cumprod = torch.cos(((x / T) + s) / (1 + s) * np.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clamp(betas, 0.0001, 0.9999)
    else:
        raise ValueError(f"Unknown schedule type: {schedule_type}")


def compute_alpha_bar(betas: torch.Tensor) -> torch.Tensor:
    """
    Compute cumulative product of alphas: ᾱ_t = ∏_{s=1}^{t} α_s
    
    Args:
        betas: Beta schedule tensor of shape (T,)
    
    Returns:
        alpha_bar: Cumulative alpha tensor of shape (T,)
    """
    alphas = 1.0 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)
    return alpha_bar


def apply_diffusion(x: torch.Tensor, t: torch.Tensor, alpha_bar: torch.Tensor, sigma: float = 0.05) -> torch.Tensor:
    """
    Applique le processus de diffusion forward.
    
    Ref: Equation (2) du papier Diffusion-GAN:
    q(y|x, t) = N(y; sqrt(ᾱ_t)x, (1 - ᾱ_t)σ² I)
    
    Formule: y = √(ᾱ_t) * x + √(1-ᾱ_t) * σ * ε
    
    Args:
        x: Input images, shape (B, C, H, W), normalized to [-1, 1]
        t: Timestep indices, shape (B,), values in [0, T-1]
        alpha_bar: Precomputed ᾱ values, shape (T,)
        sigma: Noise scale (paper: 0.05 fixed for all experiments)
    
    Returns:
        y: Diffused images, shape (B, C, H, W)
    """
    # Get alpha_bar values for each sample in batch
    # t is (B,), alpha_bar is (T,) -> alpha_bar_t is (B, 1, 1, 1)
    alpha_bar_t = alpha_bar[t].view(-1, 1, 1, 1)
    
    sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t)
    sqrt_one_minus_alpha_bar_t = torch.sqrt(1.0 - alpha_bar_t)
    
    # Générer du bruit gaussien
    noise = torch.randn_like(x)
    
    # Appliquer la diffusion: y = √(ᾱ_t) * x + √(1-ᾱ_t) * σ * ε
    y = sqrt_alpha_bar_t * x + sqrt_one_minus_alpha_bar_t * sigma * noise
    
    return y


def sample_timesteps_uniform(batch_size: int, T: int, device: torch.device) -> torch.Tensor:
    """
    Sample timesteps uniformly from [0, T-1].
    
    Args:
        batch_size: Number of samples
        T: Maximum timestep (exclusive)
        device: Target device
    
    Returns:
        t: Timestep tensor of shape (batch_size,)
    """
    return torch.randint(0, T, (batch_size,), device=device)


def sample_timesteps_priority(batch_size: int, T: int, d_scores: torch.Tensor = None, 
                               device: torch.device = None) -> torch.Tensor:
    """
    Priority sampling for timesteps (Equation 6 in paper).
    Favors higher timesteps when discriminator is too strong.
    
    For simplicity, we use a weighted distribution that gives higher
    probability to larger t values. The full priority algorithm 
    maintains an exploration list updated dynamically.
    
    Args:
        batch_size: Number of samples
        T: Maximum timestep
        d_scores: Optional discriminator scores for adaptive weighting
        device: Target device
    
    Returns:
        t: Timestep tensor of shape (batch_size,)
    """
    # Simple priority: linear weights favoring higher t
    # p(t) ∝ t + 1 (so t=0 has weight 1, t=T-1 has weight T)
    weights = torch.arange(1, T + 1, dtype=torch.float32, device=device)
    weights = weights / weights.sum()
    
    # Sample from categorical distribution
    indices = torch.multinomial(weights, batch_size, replacement=True)
    return indices


class DiffusionHelper:
    """
    Helper class for Diffusion-GAN training.
    Precomputes diffusion schedule and provides utility methods.
    """
    def __init__(self, config, device: torch.device):
        self.config = config
        self.device = device
        
        # Compute beta schedule
        self.betas = get_beta_schedule(
            schedule_type=config.scheduler_type,
            beta_start=config.beta_start,
            beta_end=config.beta_end,
            T=config.T_max
        ).to(device)
        
        # Compute alpha_bar
        self.alpha_bar = compute_alpha_bar(self.betas).to(device)
        
        # Current adaptive T
        self.T_current = config.T_max
        
        # Sigma for diffusion
        self.sigma = config.diffusion_sigma
        
        # Tracking for adaptive T
        self.d_accuracy_history = []
        self.iteration_count = 0
    
    def diffuse(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Apply forward diffusion to images."""
        return apply_diffusion(x, t, self.alpha_bar, self.sigma)
    
    def sample_timesteps(self, batch_size: int) -> torch.Tensor:
        """Sample timesteps according to configured strategy."""
        if self.config.t_sampling == 'priority':
            return sample_timesteps_priority(batch_size, self.T_current, device=self.device)
        else:
            return sample_timesteps_uniform(batch_size, self.T_current, self.device)
    
    def update_adaptive_T(self, d_accuracy: float):
        """
        Update T based on discriminator accuracy (Adaptive Diffusion Schedule).
        
        If D is too strong (accuracy > target_d), increase T to add more noise.
        If D is too weak, decrease T.
        
        Ref: Algorithm described in paper for preventing discriminator overfitting.
        """
        self.iteration_count += 1
        self.d_accuracy_history.append(d_accuracy)
        
        # Only adjust every adjust_frequency iterations
        if self.iteration_count % self.config.adjust_frequency != 0:
            return
        
        # Compute average accuracy over recent history
        recent_acc = np.mean(self.d_accuracy_history[-self.config.adjust_frequency:])
        
        # Adjust T
        if recent_acc > self.config.target_d:
            # D is overfitting, increase noise (increase T)
            self.T_current = min(self.T_current + self.config.adjust_C, self.config.T_max)
        else:
            # D is underfitting, decrease noise (decrease T)
            self.T_current = max(self.T_current - self.config.adjust_C, self.config.T_min)
    
    def get_T_current(self) -> int:
        """Get current adaptive T value."""
        return self.T_current