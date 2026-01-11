"""
Model testing utilities for visual inspection and FID calculation.
"""
import torch
import matplotlib.pyplot as plt
import numpy as np
from src.utils.fid_calculator import calculate_fid


def test_model(generator, dataloader, config, device='cuda', num_images=10, calculate_fid_score=True, fid_samples=5000):
    """
    Test generator: display generated images and calculate FID.
    
    Args:
        generator: Generator model
        dataloader: DataLoader with real images
        config: Config object
        device: Device to use
        num_images: Number of images to display
        calculate_fid_score: Whether to calculate FID (default: True)
        fid_samples: Number of samples for FID calculation
    
    Returns:
        dict: Results with 'fid' and 'diversity_ratio'
    """
    generator = generator.to(device)
    generator.eval()
    
    results = {}
    
    # Generate images
    with torch.no_grad():
        noise = torch.randn(num_images, config.z_dim).to(device)
        generated_images = generator(noise)
        
        # Calculate diversity
        flat_imgs = generated_images.view(num_images, -1)
        distances = []
        for i in range(num_images):
            for j in range(i+1, num_images):
                dist = (flat_imgs[i] - flat_imgs[j]).pow(2).sum().sqrt().item()
                distances.append(dist)
        
        if distances:
            avg_distance = sum(distances) / len(distances)
            num_channels = generated_images.shape[1]
            img_pixels = generated_images.shape[2] * generated_images.shape[3]
            max_possible_dist = (2.0 ** 2 * img_pixels * num_channels) ** 0.5
            diversity_ratio = avg_distance / max_possible_dist
            results['diversity_ratio'] = diversity_ratio
        
        # Convert to numpy for display
        generated_images = generated_images.detach().cpu().numpy()
    
    # Display images
    is_grayscale = generated_images.shape[1] == 1
    actual_h, actual_w = generated_images.shape[2], generated_images.shape[3]
    
    fig_width = min(25, max(15, num_images * actual_w / 20))
    fig_height = max(2, actual_h / 15)
    fig, axes = plt.subplots(1, num_images, figsize=(fig_width, fig_height))
    
    for i in range(num_images):
        if is_grayscale:
            img = generated_images[i, 0]
            img = (img + 1) / 2
            img = np.clip(img, 0, 1)
            axes[i].imshow(img, cmap='gray')
        else:
            img = generated_images[i].transpose(1, 2, 0)
            img = (img + 1) / 2
            img = np.clip(img, 0, 1)
            axes[i].imshow(img)
        
        axes[i].axis('off')
    
    plt.suptitle(f'Generated {actual_h}x{actual_w} - {config.name}', fontsize=12)
    plt.tight_layout()
    plt.show()
    
    # Print results
    print(f"Diversity ratio: {results.get('diversity_ratio', 0):.4f}")
    
    # Calculate FID if requested
    if calculate_fid_score:
        print(f"\nCalculating FID with {fid_samples} samples...")
        fid = calculate_fid(generator, dataloader, num_samples=fid_samples, z_dim=config.z_dim, device=device)
        results['fid'] = fid
        
        print(f"\nFID Score: {fid:.2f}")
        if fid < 20:
            print("Quality: Excellent ⭐")
        elif fid < 50:
            print("Quality: Good ✓")
        elif fid < 100:
            print("Quality: Average")
        else:
            print("Quality: Poor")
    
    return results
