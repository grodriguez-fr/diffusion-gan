"""
Loss functions for GAN training.
Implements Hinge Loss, BCE, and WGAN losses.
"""

import torch
import torch.nn as nn


class HingeLoss:
    """
    Hinge Loss for GANs (used in SNGAN).
    
    Critical for SNGAN stability - unlike BCE, gradients continue flowing
    even when the discriminator is very confident.
    
    Discriminator Loss: E[max(0, 1 - D(x_real))] + E[max(0, 1 + D(x_fake))]
    Generator Loss: -E[D(x_fake)]
    """
    
    @staticmethod
    def d_loss(real_outputs, fake_outputs):
        """
        Discriminator loss with Hinge Loss.
        
        Args:
            real_outputs: Discriminator outputs for real images (no sigmoid)
            fake_outputs: Discriminator outputs for fake images (no sigmoid)
            
        Returns:
            Discriminator loss value
        """
        loss_real = torch.nn.functional.relu(1.0 - real_outputs).mean()
        loss_fake = torch.nn.functional.relu(1.0 + fake_outputs).mean()
        return loss_real + loss_fake
    
    @staticmethod
    def g_loss(fake_outputs):
        """
        Generator loss with Hinge Loss.
        
        Args:
            fake_outputs: Discriminator outputs for fake images (no sigmoid)
            
        Returns:
            Generator loss value
        """
        return -fake_outputs.mean()


class BCELoss:
    """
    Binary Cross Entropy Loss for GANs (vanilla GAN).
    
    Note: Can suffer from vanishing gradients when discriminator is confident.
    Not recommended for SNGAN.
    """
    
    def __init__(self):
        self.criterion = nn.BCEWithLogitsLoss()
    
    def d_loss(self, real_outputs, fake_outputs, real_labels, fake_labels):
        """
        Discriminator loss with BCE.
        
        Args:
            real_outputs: Discriminator logits for real images
            fake_outputs: Discriminator logits for fake images
            real_labels: Labels for real images (typically ones)
            fake_labels: Labels for fake images (typically zeros)
            
        Returns:
            Discriminator loss value
        """
        loss_real = self.criterion(real_outputs, real_labels)
        loss_fake = self.criterion(fake_outputs, fake_labels)
        return loss_real + loss_fake
    
    def g_loss(self, fake_outputs, real_labels):
        """
        Generator loss with BCE.
        
        Args:
            fake_outputs: Discriminator logits for fake images
            real_labels: Target labels (typically ones - fool discriminator)
            
        Returns:
            Generator loss value
        """
        return self.criterion(fake_outputs, real_labels)


class WGANLoss:
    """
    Wasserstein GAN Loss (WGAN).
    
    Discriminator Loss: -E[D(x_real)] + E[D(x_fake)]
    Generator Loss: -E[D(x_fake)]
    
    Note: Requires weight clipping or gradient penalty for stability.
    """
    
    @staticmethod
    def d_loss(real_outputs, fake_outputs):
        """
        Discriminator loss with Wasserstein distance.
        
        Args:
            real_outputs: Discriminator outputs for real images
            fake_outputs: Discriminator outputs for fake images
            
        Returns:
            Discriminator loss value
        """
        return -real_outputs.mean() + fake_outputs.mean()
    
    @staticmethod
    def g_loss(fake_outputs):
        """
        Generator loss with Wasserstein distance.
        
        Args:
            fake_outputs: Discriminator outputs for fake images
            
        Returns:
            Generator loss value
        """
        return -fake_outputs.mean()


def get_loss_function(loss_type: str):
    """
    Get loss function by name.
    
    Args:
        loss_type: 'hinge', 'bce', or 'wgan'
        
    Returns:
        Loss function instance
        
    Raises:
        ValueError: If loss_type is not recognized
    """
    loss_map = {
        'hinge': HingeLoss(),
        'bce': BCELoss(),
        'wgan': WGANLoss(),
    }
    
    if loss_type not in loss_map:
        available = ', '.join(loss_map.keys())
        raise ValueError(f"Loss type '{loss_type}' not recognized. Available: {available}")
    
    return loss_map[loss_type]
