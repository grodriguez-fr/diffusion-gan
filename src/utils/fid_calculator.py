"""
Simple FID calculator using pytorch-fid.
"""
import torch
import os
import tempfile
import shutil
from pathlib import Path
from torchvision.utils import save_image
from pytorch_fid import fid_score
from tqdm import tqdm


def calculate_fid(generator, dataloader, num_samples=5000, z_dim=128, device='cuda', verbose=True):
    """
    Calculate FID score between real images and generated images.
    
    Args:
        generator: Generator model
        dataloader: DataLoader with real images
        num_samples: Number of samples to use (default: 5000)
        z_dim: Latent dimension (default: 128)
        device: Device to use (default: 'cuda')
        verbose: Print progress (default: True)
    
    Returns:
        float: FID score (lower is better)
    """
    # Setup
    generator = generator.to(device)
    generator.eval()
    
    # Create temp directories
    temp_dir = tempfile.mkdtemp()
    real_dir = os.path.join(temp_dir, 'real')
    fake_dir = os.path.join(temp_dir, 'fake')
    os.makedirs(real_dir, exist_ok=True)
    os.makedirs(fake_dir, exist_ok=True)
    
    try:
        # Save real images
        if verbose:
            print(f"Saving {num_samples} real images...")
        count = 0
        pbar = tqdm(total=num_samples, desc="Real images", disable=not verbose)
        for images, _ in dataloader:
            for img in images:
                if count >= num_samples:
                    break
                save_image(img, os.path.join(real_dir, f'real_{count:05d}.png'), normalize=True)
                count += 1
                pbar.update(1)
            if count >= num_samples:
                break
        pbar.close()
        
        # Generate fake images
        if verbose:
            print(f"Generating {num_samples} fake images...")
        with torch.no_grad():
            for i in tqdm(range(num_samples), desc="Fake images", disable=not verbose):
                z = torch.randn(1, z_dim, device=device)
                fake_img = generator(z)
                save_image(fake_img[0], os.path.join(fake_dir, f'fake_{i:05d}.png'), normalize=True)
        
        # Calculate FID
        if verbose:
            print("Computing FID score...")
        fid_value = fid_score.calculate_fid_given_paths(
            [real_dir, fake_dir],
            batch_size=50,
            device=str(device),
            dims=2048,
            num_workers=0
        )
        
        return fid_value
        
    finally:
        # Cleanup
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


def calculate_fid_silent(generator, dataloader, num_samples=5000, z_dim=128, device='cuda'):
    """
    Calculate FID without printing progress.
    """
    return calculate_fid(generator, dataloader, num_samples, z_dim, device, verbose=False)
