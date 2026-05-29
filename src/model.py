"""
Training Loop and Optimization Logic for Conditional WGAN-GP.

This module fuses the alternating WGAN-GP optimization with conditional 
masking and L1 border penalties to generate seamless continuous terrain.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import torchvision.utils as vutils
import torch.nn.functional as F

from config import (
    LEARNING_RATE, BETA1, BETA2, CRITIC_ITERATIONS, LAMBDA_GP, LATENT_DIM, CONDITION_WIDTH
)

def compute_gradient_penalty(critic, real_samples, fake_samples, conditions, masks):
    """
    Calculates the conditional gradient penalty for the cWGAN-GP.
    """
    batch_size = real_samples.size(0)
    current_device = real_samples.device
    
    fake_samples = fake_samples.detach()
    
    epsilon = torch.rand(batch_size, 1, 1, 1, device=current_device)
    interpolated = (epsilon * real_samples + ((1 - epsilon) * fake_samples)).requires_grad_(True)
    
    if isinstance(critic, nn.DataParallel):
        critic_interpolated = critic.module(interpolated, conditions, masks)
    else:
        critic_interpolated = critic(interpolated, conditions, masks)
    
    gradients = torch.autograd.grad(
        outputs=critic_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones_like(critic_interpolated, device=current_device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    
    gradients = gradients.view(batch_size, -1)
    gradient_norm = gradients.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1.0) ** 2)
    
    return gradient_penalty

def train_wgan(generator, critic, dataloader, device):
    """
    Executes the main WGAN-GP training loop with conditional stitching.
    """
    opt_gen = optim.Adam(generator.parameters(), lr=LEARNING_RATE, betas=(BETA1, BETA2))
    opt_critic = optim.Adam(critic.parameters(), lr=LEARNING_RATE, betas=(BETA1, BETA2))

    from config import EPOCHS 

    # --- SETUP FIXED EVALUATION SET ---
    # We pull a single batch from the dataloader to create a permanent 
    # reference point for visual checkpointing across all epochs.
    print("Preparing fixed evaluation conditions...")
    real_batch = next(iter(dataloader)).to(device)
    fixed_batch_size = min(16, real_batch.shape[0])
    
    fixed_noise = torch.randn(fixed_batch_size, LATENT_DIM, 1, 1, device=device)
    fixed_masks = torch.zeros(fixed_batch_size, 1, 256, 256, device=device)
    
    # We apply a left-side condition to the evaluation batch for consistency
    fixed_masks[:, :, :, :CONDITION_WIDTH] = 1 
    fixed_conditions = real_batch[:fixed_batch_size] * fixed_masks
    # ----------------------------------

    for epoch in range(EPOCHS):
        
        # --- VISUAL CHECKPOINTING ---
        if epoch % 10 == 0:
            generator.eval()
            with torch.no_grad():
                fake_checkpoint = generator(fixed_noise, fixed_conditions, fixed_masks).cpu()
                # Assuming channel 0 is Elevation
                elevation_only = fake_checkpoint[:, 0:1, :, :] 
                vutils.save_image(elevation_only, f"checkpoint_epoch_{epoch}.png", nrow=4, normalize=True)
            generator.train()
        # ----------------------------

        loop = tqdm(dataloader, leave=True)
        # Note: Dataloader yields just real_images; we dynamically construct masks below
        for batch_idx, real_images in enumerate(loop):
            real_images = real_images.to(device)
            batch_size = real_images.shape[0]

            # --- DYNAMIC MASK GENERATION ---
            masks = torch.zeros(batch_size, 1, 256, 256, device=device)
            side = torch.randint(0, 5, (1,)).item() 
            
            if side == 1: masks[:, :, :, :CONDITION_WIDTH] = 1     # Left
            elif side == 2: masks[:, :, :, -CONDITION_WIDTH:] = 1  # Right
            elif side == 3: masks[:, :, :CONDITION_WIDTH, :] = 1   # Top
            elif side == 4: masks[:, :, -CONDITION_WIDTH:, :] = 1  # Bottom
            
            conditions = real_images * masks
            # -------------------------------

            # --- CRITIC UPDATE ---
            for _ in range(CRITIC_ITERATIONS):
                noise = torch.randn(batch_size, LATENT_DIM, 1, 1, device=device)
                
                fake_images = generator(noise, conditions, masks)

                critic_real = critic(real_images, conditions, masks).reshape(-1)
                critic_fake = critic(fake_images.detach(), conditions, masks).reshape(-1)

                gp = compute_gradient_penalty(critic, real_images, fake_images, conditions, masks)

                loss_critic = -(torch.mean(critic_real) - torch.mean(critic_fake)) + LAMBDA_GP * gp

                critic.zero_grad()
                loss_critic.backward()
                opt_critic.step()

            # --- GENERATOR UPDATE ---
            output = critic(fake_images, conditions, masks).reshape(-1)
            
            # L1 Penalty: Forces the generator to perfectly copy the boundary condition pixels
            l1_penalty = F.l1_loss(fake_images * masks, conditions)
            
            # Combine the adversarial loss with a heavy weight (10.0) on the L1 boundary loss
            loss_gen = -torch.mean(output) + (10.0 * l1_penalty)

            generator.zero_grad()
            loss_gen.backward()
            opt_gen.step()

            # --- LOGGING ---
            if batch_idx % 50 == 0:
                loop.set_description(f"Epoch [{epoch}/{EPOCHS}]")
                loop.set_postfix(
                    Loss_Critic=loss_critic.item(), 
                    Loss_Gen=loss_gen.item(),
                    L1_Border=l1_penalty.item()
                )
                
    return generator, critic