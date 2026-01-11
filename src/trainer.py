"""
Trainer class for GAN training.
Handles training loop, logging, and model saving.
Supports both vanilla GAN and Diffusion-GAN modes.
"""

import torch
import tqdm
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os
from pathlib import Path
from src.utils.losses import get_loss_function
from src.utils.apply_diffusion import DiffusionHelper
import torchvision.utils as vutils
import numpy as np

# Import FID calculation
try:
    from src.utils.fid_calculator import calculate_fid_silent
    FID_AVAILABLE = True
except ImportError:
    FID_AVAILABLE = False

# Import model loader for checkpointing
from src.tools.model_loader import save_checkpoint, load_checkpoint, checkpoint_exists, get_checkpoint_info


class Trainer:
    """
    Trainer for GAN models with TensorBoard logging and model checkpointing.
    
    Args:
        generator: Generator model
        discriminator: Discriminator model
        config: Configuration object with training parameters
        train_loader: DataLoader for training data
        device: torch.device for computation
    """
    
    def __init__(self, generator, discriminator, config, train_loader, device, resume=True):
        self.generator = generator
        self.discriminator = discriminator
        self.config = config
        self.train_loader = train_loader
        self.device = device
        
        # Move models to device
        self.generator = self.generator.to(device)
        self.discriminator = self.discriminator.to(device)
        
        # Loss function
        self.loss_fn = get_loss_function(config.loss_type)
        
        # Optimizers
        self.optimizer_g = torch.optim.Adam(
            generator.parameters(), 
            lr=config.lr_g, 
            betas=(config.beta1, config.beta2)
        )
        self.optimizer_d = torch.optim.Adam(
            discriminator.parameters(), 
            lr=config.lr_d, 
            betas=(config.beta1, config.beta2)
        )
        
        # Training state - will be updated if resuming
        self.start_epoch = 0
        self.global_step = 0
        self.resumed = False
        self.history = {
            'd_loss': [],
            'g_loss': [],
            'd_acc': [],
            'val_metrics': [],
            'fid_scores': []
        }
        
        # Try to resume from checkpoint
        if resume and checkpoint_exists(config, config.save_dir):
            self.start_epoch, self.global_step, saved_history, saved_T = load_checkpoint(
                config=config,
                generator=self.generator,
                discriminator=self.discriminator,
                optimizer_g=self.optimizer_g,
                optimizer_d=self.optimizer_d,
                device=device,
                save_dir=config.save_dir
            )
            # Restore history
            if saved_history:
                self.history = saved_history
            # Restore T if saved
            if saved_T is not None:
                self.T = saved_T
            self.start_epoch += 1  # Start from next epoch
            self.resumed = True
        
        # TensorBoard setup - use config name for consistent run folder
        os.makedirs(config.log_dir, exist_ok=True)
        os.makedirs(config.save_dir, exist_ok=True)
        
        # Use consistent run name based on config (not timestamp) for resumption
        self.run_name = config.name
        self.writer = SummaryWriter(os.path.join(config.log_dir, self.run_name))
        
        # Create samples directory for image grids
        self.samples_dir = os.path.join(config.log_dir, self.run_name, 'samples')
        os.makedirs(self.samples_dir, exist_ok=True)
        
        # Fixed noise for consistent visualization across epochs (4x4 grid = 16 images)
        self.fixed_noise = torch.randn(16, config.z_dim).to(device)
        
        # Current adaptive T (for Diffusion-GAN)
        self.T = config.T_max
        
        # Initialize Diffusion Helper if using diffusion
        self.diffusion_helper = None
        if config.use_diffusion:
            self.diffusion_helper = DiffusionHelper(config, device)
            print(f"🌀 Diffusion-GAN mode enabled: T_max={config.T_max}, σ={config.diffusion_sigma}")
        
        # FID tracking (activé par config)
        self.compute_fid = getattr(config, 'compute_fid', True) and FID_AVAILABLE
        self.fid_frequency = getattr(config, 'fid_frequency', 10)
        self.fid_num_samples = getattr(config, 'fid_num_samples', 5000)
        
    def print_setup(self):
        """Print training setup information."""
        print(f"📊 TensorBoard: tensorboard --logdir={self.config.log_dir}")
        print(f"📁 Run: {self.run_name}")
        print(f"⚙️  Config: {self.config.name}")
        
        if self.resumed:
            print(f"🔄 Resuming from epoch {self.start_epoch} (step {self.global_step})")
        print()
        
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"Using device: {self.device}\n")
        
        print(f"🎯 Loss function: {self.config.loss_type.upper()}")
        
        print(f"\n🎯 Training configuration:")
        print(f"   Image size: {self.config.image_size}x{self.config.image_size}")
        print(f"   Batch size: {self.config.batch_size}")
        print(f"   Epochs: {self.start_epoch} → {self.config.num_epochs}")
        print(f"   n_critic: {self.config.n_critic}")
        print(f"   LR: G={self.config.lr_g}, D={self.config.lr_d}")
        if self.config.use_diffusion:
            print(f"\n🌀 Diffusion-GAN settings:")
            print(f"   T_max={self.config.T_max}, T_min={self.config.T_min}")
            print(f"   σ (diffusion_sigma)={self.config.diffusion_sigma}")
            print(f"   β_start={self.config.beta_start}, β_end={self.config.beta_end}")
            print(f"   target_d={self.config.target_d} (adjust every {self.config.adjust_frequency} iters)")
            print(f"   t_sampling={self.config.t_sampling}")
        print()
    
    def train_discriminator_step(self, images):
        """
        Single discriminator training step.
        
        In Diffusion-GAN mode:
        1. Sample timestep t from [0, T_current]
        2. Apply diffusion to both real and fake images
        3. D receives (diffused_image, t) pair
        
        Args:
            images: Batch of real images
            
        Returns:
            d_loss: Discriminator loss value
            d_correct: Number of correct predictions
            d_total: Total number of predictions
        """
        batch_size = images.size(0)
        self.optimizer_d.zero_grad()
        
        # Generate fake images (G(z) - generator never sees t!)
        noise = torch.randn(batch_size, self.config.z_dim).to(self.device)
        fake_images = self.generator(noise)
        
        if self.config.use_diffusion and self.diffusion_helper is not None:
            # === DIFFUSION-GAN MODE ===
            # Sample timestep t for this batch
            t = self.diffusion_helper.sample_timesteps(batch_size)
            
            # Apply forward diffusion to BOTH real and fake images
            real_diffused = self.diffusion_helper.diffuse(images, t)
            fake_diffused = self.diffusion_helper.diffuse(fake_images.detach(), t)
            
            # D(y, t) - discriminator sees diffused images AND timestep
            real_outputs = self.discriminator(real_diffused, t.float())
            fake_outputs = self.discriminator(fake_diffused, t.float())
        else:
            # === VANILLA GAN MODE ===
            real_outputs = self.discriminator(images)
            fake_outputs = self.discriminator(fake_images.detach())
        
        # Compute D loss based on loss type
        if self.config.loss_type == 'hinge':
            d_loss = self.loss_fn.d_loss(real_outputs, fake_outputs)
        elif self.config.loss_type == 'bce':
            real_labels = torch.ones(batch_size, 1).to(self.device)
            fake_labels = torch.zeros(batch_size, 1).to(self.device)
            d_loss = self.loss_fn.d_loss(real_outputs, fake_outputs, real_labels, fake_labels)
        else:  # wgan
            d_loss = self.loss_fn.d_loss(real_outputs, fake_outputs)
        
        d_loss.backward()
        self.optimizer_d.step()
        
        # Track discriminator accuracy
        with torch.no_grad():
            d_correct = ((real_outputs > 0).sum() + (fake_outputs <= 0).sum()).item()
            d_total = batch_size * 2
            d_accuracy = d_correct / d_total
            
            # Update adaptive T if using diffusion
            if self.config.use_diffusion and self.diffusion_helper is not None:
                self.diffusion_helper.update_adaptive_T(d_accuracy)
        
        return d_loss.item(), d_correct, d_total
    
    def train_generator_step(self, batch_size):
        """
        Single generator training step.
        
        In Diffusion-GAN mode:
        - G generates clean image: x_fake = G(z)
        - Diffusion is applied AFTER generation: y_fake = diffuse(x_fake, t)
        - Gradient flows back through diffusion process to G
        
        Args:
            batch_size: Size of the batch
            
        Returns:
            g_loss: Generator loss value
        """
        self.optimizer_g.zero_grad()
        
        # G(z) -> clean fake image (G never sees t!)
        noise = torch.randn(batch_size, self.config.z_dim).to(self.device)
        fake_images = self.generator(noise)
        
        if self.config.use_diffusion and self.diffusion_helper is not None:
            # === DIFFUSION-GAN MODE ===
            # Sample timestep (same distribution as D training)
            t = self.diffusion_helper.sample_timesteps(batch_size)
            
            # Apply diffusion to fake images (gradient flows through!)
            fake_diffused = self.diffusion_helper.diffuse(fake_images, t)
            
            # D(y_fake, t) - discriminator evaluates diffused fake
            fake_outputs = self.discriminator(fake_diffused, t.float())
        else:
            # === VANILLA GAN MODE ===
            fake_outputs = self.discriminator(fake_images)
        
        # Compute G loss based on loss type
        if self.config.loss_type == 'hinge':
            g_loss = self.loss_fn.g_loss(fake_outputs)
        elif self.config.loss_type == 'bce':
            real_labels = torch.ones(batch_size, 1).to(self.device)
            g_loss = self.loss_fn.g_loss(fake_outputs, real_labels)
        else:  # wgan
            g_loss = self.loss_fn.g_loss(fake_outputs)
        
        g_loss.backward()
        self.optimizer_g.step()
        
        return g_loss.item()
    
    def train_epoch(self, epoch):
        """
        Train for one epoch.
        
        Args:
            epoch: Current epoch number (0-indexed)
            
        Returns:
            avg_d_loss: Average discriminator loss
            avg_g_loss: Average generator loss
            avg_d_acc: Average discriminator accuracy
        """
        pbar = tqdm.tqdm(
            self.train_loader, 
            desc=f'Epoch [{epoch+1:3d}/{self.config.num_epochs}]', 
            unit='batch', 
            ncols=130, 
            leave=True
        )
        
        epoch_d_loss = 0
        epoch_g_loss = 0
        batch_count = 0
        d_correct_total = 0
        d_total_total = 0
        
        for images, _ in pbar:
            images = images.to(self.device)
            batch_size = images.size(0)
            
            # Train discriminator n_critic times
            for _ in range(self.config.n_critic):
                d_loss, d_correct, d_total = self.train_discriminator_step(images)
                d_correct_total += d_correct
                d_total_total += d_total
            
            # Train generator once
            g_loss = self.train_generator_step(batch_size)
            
            # Accumulate losses
            epoch_d_loss += d_loss
            epoch_g_loss += g_loss
            batch_count += 1
            
            # Write to TensorBoard
            self.writer.add_scalar('Loss/D', d_loss, self.global_step)
            self.writer.add_scalar('Loss/G', g_loss, self.global_step)
            self.writer.add_scalar('Accuracy/D', d_correct_total / d_total_total, self.global_step)
            
            # Log adaptive T for Diffusion-GAN
            if self.config.use_diffusion and self.diffusion_helper is not None:
                self.writer.add_scalar('Diffusion/T_current', 
                                       self.diffusion_helper.get_T_current(), 
                                       self.global_step)
            
            self.global_step += 1
            
            # Update progress bar
            avg_d = epoch_d_loss / batch_count
            avg_g = epoch_g_loss / batch_count
            d_acc = d_correct_total / d_total_total
            
            postfix = {
                'D_loss': f'{avg_d:.4f}', 
                'G_loss': f'{avg_g:.4f}', 
                'D_acc': f'{d_acc:.3f}'
            }
            
            # Add T_current for Diffusion-GAN
            if self.config.use_diffusion and self.diffusion_helper is not None:
                postfix['T'] = self.diffusion_helper.get_T_current()
            
            pbar.set_postfix(postfix)
        
        return avg_d, avg_g, d_acc
    
    def validate(self, epoch):
        """
        Validation step: Generate images and compute diversity metrics.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            dict: Validation metrics (pixel_std, mean_variance, diversity_ratio)
        """
        self.generator.eval()
        
        with torch.no_grad():
            # Generate images from fixed noise for consistency
            fake_images = self.generator(self.fixed_noise)
            
            # === 1. Compute Pixel Standard Deviation (Poor Man's FID) ===
            pixel_std = fake_images.std().item()
            pixel_mean = fake_images.mean().item()
            
            # === 2. Compute Mean Variance (Mode Collapse Detector) ===
            # Calculate mean of each image
            batch_size = fake_images.size(0)
            img_means = fake_images.view(batch_size, -1).mean(dim=1)
            mean_variance = img_means.var().item()
            
            # === 3. Compute Diversity Ratio (L2 Distance) ===
            flat_imgs = fake_images.view(batch_size, -1)
            distances = []
            # Sample 100 random pairs to avoid O(n²) complexity
            num_samples = min(100, (batch_size * (batch_size - 1)) // 2)
            for _ in range(num_samples):
                i, j = np.random.choice(batch_size, 2, replace=False)
                dist = (flat_imgs[i] - flat_imgs[j]).pow(2).sum().sqrt().item()
                distances.append(dist)
            avg_distance = np.mean(distances)
            
            # Calculate theoretical max distance for normalization
            num_channels = fake_images.size(1)
            img_size = fake_images.size(2) * fake_images.size(3)
            max_possible_dist = (2.0 ** 2 * img_size * num_channels) ** 0.5
            diversity_ratio = avg_distance / max_possible_dist
            
            # === 4. Save Image Grid ===
            # Normalize to [0, 1] for visualization
            fake_images_normalized = (fake_images + 1) / 2
            fake_images_normalized = torch.clamp(fake_images_normalized, 0, 1)
            
            # Create grid (4x4 = 16 images) at full resolution
            grid = vutils.make_grid(fake_images_normalized, nrow=4, normalize=False, pad_value=1)
            
            # Save to file
            grid_path = os.path.join(self.samples_dir, f'epoch_{epoch+1:03d}.png')
            vutils.save_image(grid, grid_path)
            
            # Save to TensorBoard
            self.writer.add_image('Generated/Image_Grid', grid, epoch)
            
            # === 5. Log Metrics to TensorBoard ===
            self.writer.add_scalar('Validation/Pixel_Std', pixel_std, epoch)
            self.writer.add_scalar('Validation/Pixel_Mean', pixel_mean, epoch)
            self.writer.add_scalar('Validation/Mean_Variance', mean_variance, epoch)
            self.writer.add_scalar('Validation/Diversity_Ratio', diversity_ratio, epoch)
            self.writer.add_scalar('Validation/Avg_L2_Distance', avg_distance, epoch)
            
            # === 6. Console Output (Basic Metrics Only) ===
            mode_collapse_warning = ""
            if mean_variance < 0.001:
                mode_collapse_warning = " ⚠️ MODE COLLAPSE!"
            elif diversity_ratio < 0.15:
                mode_collapse_warning = " ⚠️ Low Diversity"
            
            print(f"   📊 Validation: Std={pixel_std:.4f}, Diversity={diversity_ratio:.4f}{mode_collapse_warning}")
        
        self.generator.train()
        
        return {
            'pixel_std': pixel_std,
            'mean_variance': mean_variance,
            'diversity_ratio': diversity_ratio,
            'avg_distance': avg_distance
        }
    
    def train(self):
        """
        Main training loop with checkpoint resumption support.
        
        Returns:
            dict: Training history with losses and accuracies
        """
        self.print_setup()
        
        # Use restored history or start fresh
        history = self.history
        
        for epoch in range(self.start_epoch, self.config.num_epochs):
            avg_d, avg_g, avg_acc = self.train_epoch(epoch)
            history['d_loss'].append(avg_d)
            history['g_loss'].append(avg_g)
            history['d_acc'].append(avg_acc)
            
            # Run validation every 5 epochs
            if (epoch + 1) % 5 == 0 or epoch == 0:
                val_metrics = self.validate(epoch)
                history['val_metrics'].append({
                    'epoch': epoch + 1,
                    **val_metrics
                })
            
            # Compute FID score périodiquement
            if self.compute_fid and ((epoch + 1) % self.fid_frequency == 0 or epoch == 0):
                print(f"\n📏 Computing FID (epoch {epoch + 1})...")
                try:
                    fid_score = calculate_fid_silent(
                        generator=self.generator,
                        dataloader=self.train_loader,
                        num_samples=self.fid_num_samples,
                        z_dim=self.config.z_dim,
                        device=self.device
                    )
                    
                    # Log to TensorBoard
                    self.writer.add_scalar('Metrics/FID', fid_score, epoch + 1)
                    
                    # Save to history
                    history['fid_scores'].append({
                        'epoch': epoch + 1,
                        'fid': fid_score
                    })
                    
                    print(f"   FID: {fid_score:.2f}")
                    
                except Exception as e:
                    print(f"   Error: {e}")
            
            # Save checkpoint after each epoch
            save_checkpoint(
                generator=self.generator,
                discriminator=self.discriminator,
                optimizer_g=self.optimizer_g,
                optimizer_d=self.optimizer_d,
                epoch=epoch,
                global_step=self.global_step,
                config=self.config,
                history=history,
                current_T=self.T,
                save_dir=self.config.save_dir
            )
        
        print(f"\n✅ Training complete! Checkpoint saved to {self.config.save_dir}/")
        print(f"📊 View results: tensorboard --logdir={self.config.log_dir}")
        print(f"🖼️  Image grids saved to {self.samples_dir}/")
        
        self.writer.close()
        
        return history
    
    def save_models(self):
        """Save generator and discriminator models."""
        torch.save(
            self.generator.state_dict(), 
            f'{self.config.save_dir}/generator_{self.run_name}.pt'
        )
        torch.save(
            self.discriminator.state_dict(), 
            f'{self.config.save_dir}/discriminator_{self.run_name}.pt'
        )
    
    def load_models(self, generator_path, discriminator_path):
        """
        Load pre-trained models.
        
        Args:
            generator_path: Path to generator weights
            discriminator_path: Path to discriminator weights
        """
        self.generator.load_state_dict(torch.load(generator_path, map_location=self.device))
        self.discriminator.load_state_dict(torch.load(discriminator_path, map_location=self.device))
        print(f"✅ Models loaded from {generator_path} and {discriminator_path}")
