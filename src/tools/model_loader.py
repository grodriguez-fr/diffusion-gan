"""
Model loader for checkpoint management.
Handles saving/loading models with epoch tracking for training resumption.
"""

import torch
from pathlib import Path
from typing import Optional, Tuple, Dict, Any


def get_checkpoint_path(config, save_dir: str = './models') -> Path:
    """
    Get the checkpoint path for a specific config.
    One checkpoint per config name.
    
    Args:
        config: Configuration object with 'name' attribute
        save_dir: Directory to save/load checkpoints
        
    Returns:
        Path to the checkpoint file
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    return save_dir / f'checkpoint_{config.name}.pt'


def save_checkpoint(
    generator: torch.nn.Module,
    discriminator: torch.nn.Module,
    optimizer_g: torch.optim.Optimizer,
    optimizer_d: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    config,
    history: Optional[Dict] = None,
    current_T: Optional[int] = None,
    save_dir: str = './models'
) -> Path:
    """
    Save a training checkpoint with all necessary state for resumption.
    
    Args:
        generator: Generator model
        discriminator: Discriminator model
        optimizer_g: Generator optimizer
        optimizer_d: Discriminator optimizer
        epoch: Current epoch (0-indexed)
        global_step: Current global step for TensorBoard
        config: Configuration object
        history: Training history dict
        current_T: Current diffusion timestep T (for Diffusion-GAN)
        save_dir: Directory to save checkpoint
        
    Returns:
        Path to saved checkpoint
    """
    checkpoint_path = get_checkpoint_path(config, save_dir)
    
    checkpoint = {
        'epoch': epoch,
        'global_step': global_step,
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'optimizer_g_state_dict': optimizer_g.state_dict(),
        'optimizer_d_state_dict': optimizer_d.state_dict(),
        'config_name': config.name,
        'history': history or {},
        'current_T': current_T
    }
    
    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path


def load_checkpoint(
    config,
    generator: torch.nn.Module,
    discriminator: torch.nn.Module,
    optimizer_g: Optional[torch.optim.Optimizer] = None,
    optimizer_d: Optional[torch.optim.Optimizer] = None,
    device: torch.device = None,
    save_dir: str = './models'
) -> Tuple[int, int, Dict[str, Any]]:
    """
    Load a training checkpoint and restore model/optimizer states.
    
    Args:
        config: Configuration object with 'name' attribute
        generator: Generator model to load weights into
        discriminator: Discriminator model to load weights into
        optimizer_g: Generator optimizer (optional, for training resumption)
        optimizer_d: Discriminator optimizer (optional, for training resumption)
        device: Device to map tensors to
        save_dir: Directory where checkpoints are saved
        
    Returns:
        Tuple of (start_epoch, global_step, history)
        start_epoch is the next epoch to train (checkpoint_epoch + 1)
    """
    checkpoint_path = get_checkpoint_path(config, save_dir)
    
    if not checkpoint_path.exists():
        return 0, 0, {}
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Load model states
    generator.load_state_dict(checkpoint['generator_state_dict'])
    discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
    
    # Load optimizer states if provided
    if optimizer_g is not None and 'optimizer_g_state_dict' in checkpoint:
        optimizer_g.load_state_dict(checkpoint['optimizer_g_state_dict'])
    if optimizer_d is not None and 'optimizer_d_state_dict' in checkpoint:
        optimizer_d.load_state_dict(checkpoint['optimizer_d_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    global_step = checkpoint.get('global_step', 0)
    history = checkpoint.get('history', {})
    current_T = checkpoint.get('current_T', None)
    
    return epoch, global_step, history, current_T


def checkpoint_exists(config, save_dir: str = './models') -> bool:
    """
    Check if a checkpoint exists for the given config.
    
    Args:
        config: Configuration object with 'name' attribute
        save_dir: Directory where checkpoints are saved
        
    Returns:
        True if checkpoint exists
    """
    return get_checkpoint_path(config, save_dir).exists()


def get_checkpoint_info(config, save_dir: str = './models') -> Optional[Dict[str, Any]]:
    """
    Get information about a checkpoint without loading the full state.
    
    Args:
        config: Configuration object with 'name' attribute
        save_dir: Directory where checkpoints are saved
        
    Returns:
        Dict with checkpoint info or None if not found
    """
    checkpoint_path = get_checkpoint_path(config, save_dir)
    
    if not checkpoint_path.exists():
        return None
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    return {
        'epoch': checkpoint.get('epoch', 0),
        'global_step': checkpoint.get('global_step', 0),
        'config_name': checkpoint.get('config_name', 'unknown'),
        'path': checkpoint_path
    }


def delete_checkpoint(config, save_dir: str = './models') -> bool:
    """
    Delete a checkpoint for the given config.
    
    Args:
        config: Configuration object with 'name' attribute
        save_dir: Directory where checkpoints are saved
        
    Returns:
        True if deleted, False if not found
    """
    checkpoint_path = get_checkpoint_path(config, save_dir)
    
    if checkpoint_path.exists():
        checkpoint_path.unlink()
        return True
    return False


def load_for_inference(
    config,
    generator: torch.nn.Module,
    discriminator: Optional[torch.nn.Module] = None,
    device: torch.device = None,
    save_dir: str = './models'
) -> int:
    """
    Load models for inference only (no optimizer states).
    
    Args:
        config: Configuration object
        generator: Generator model
        discriminator: Discriminator model (optional)
        device: Device to map tensors to
        save_dir: Directory where checkpoints are saved
        
    Returns:
        Epoch the checkpoint was saved at
    """
    checkpoint_path = get_checkpoint_path(config, save_dir)
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"No checkpoint found for config '{config.name}'")
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    generator.load_state_dict(checkpoint['generator_state_dict'])
    
    if discriminator is not None:
        discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
    
    return checkpoint.get('epoch', 0)
