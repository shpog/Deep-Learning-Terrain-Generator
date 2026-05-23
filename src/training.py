"""
Training Loop and Optimization Logic for WGAN-GP.

This module contains the core algorithms required to train the Terrain Generator.
It handles the alternating optimization between the Critic and Generator, and 
computes the mathematically strict Gradient Penalty required for Wasserstein GANs.
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
    Calculates the gradient penalty for the WGAN-GP to enforce 1-Lipschitz continuity.

    The penalty ensures the gradient of the Critic's output with respect to the 
    input image has a norm (magnitude) of roughly 1.0. This stabilizes training 
    and prevents mode collapse.

    Args:
        critic (torch.nn.Module): The initialized Critic network.
        real_samples (torch.Tensor): A batch of real terrain maps.
        fake_samples (torch.Tensor): A batch of generated fake terrain maps.

    Returns:
        torch.Tensor: A scalar tensor representing the computed gradient penalty.
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
    Executes the main WGAN-GP training loop.

    The loop strictly follows the WGAN rule of updating the Critic multiple 
    times (defined by CRITIC_ITERATIONS) for every single Generator update. 

    Args:
        generator (torch.nn.Module): The initialized TerrainGenerator network.
        critic (torch.nn.Module): The initialized TerrainCritic network.
        dataloader (torch.utils.data.DataLoader): The DataLoader providing real terrain patches.
        device (torch.device): The primary computation device (e.g., 'cuda' or 'cpu').

    Returns:
        tuple: A tuple containing the fully trained networks:
            - torch.nn.Module: The updated Generator.
            - torch.nn.Module: The updated Critic.
    """
    opt_gen = optim.Adam(generator.parameters(), lr=LEARNING_RATE, betas=(BETA1, BETA2))
    opt_critic = optim.Adam(critic.parameters(), lr=LEARNING_RATE, betas=(BETA1, BETA2))

    generator.train()
    critic.train()

    from config import EPOCHS 

    for epoch in range(EPOCHS):
        if epoch % 10 == 0:
            generator.eval()
            with torch.no_grad():
                fixed_noise = torch.randn(16, LATENT_DIM, 1, 1, device=device)
                from config import TILE_SIZE, CONDITION_WIDTH, IMG_CHANNELS
                dummy_condition = torch.full((16, IMG_CHANNELS, TILE_SIZE, TILE_SIZE), 0.5, device=device)
                dummy_mask = torch.zeros((16, 1, TILE_SIZE, TILE_SIZE), device=device)
                dummy_mask[:, :, :, :CONDITION_WIDTH] = 1.0
                dummy_condition = dummy_condition * dummy_mask
                fake_checkpoint = generator(fixed_noise, dummy_condition, dummy_mask).cpu()
                elevation_only = fake_checkpoint[:, 0:1, :, :] 
                import torchvision.utils as vutils
                vutils.save_image(elevation_only, f"checkpoint_epoch_{epoch}.png", nrow=4, normalize=True)
            generator.train()
        loop = tqdm(dataloader, leave=True)
        for batch_idx, (conditions, masks, real_images) in enumerate(loop):
            real_images = real_images.to(device)
            conditions = conditions.to(device)
            masks = masks.to(device) # Send masks to GPU
            batch_size = real_images.shape[0]

            for _ in range(CRITIC_ITERATIONS):
                noise = torch.randn(batch_size, LATENT_DIM, 1, 1, device=device)
                
                # 2. Pass masks to Generator and Critic
                fake_images = generator(noise, conditions, masks)

                critic_real = critic(real_images, conditions, masks).reshape(-1)
                critic_fake = critic(fake_images.detach(), conditions, masks).reshape(-1)

                # Update compute_gradient_penalty signature to accept masks if you haven't already!
                gp = compute_gradient_penalty(critic, real_images, fake_images, conditions, masks)

                loss_critic = -(torch.mean(critic_real) - torch.mean(critic_fake)) + LAMBDA_GP * gp

                critic.zero_grad()
                loss_critic.backward()
                opt_critic.step()

            output = critic(fake_images, conditions, masks).reshape(-1)
            
            l1_penalty = F.l1_loss(fake_images * masks, conditions)
            
            loss_gen = -torch.mean(output) + (10.0 * l1_penalty)

            generator.zero_grad()
            loss_gen.backward()
            opt_gen.step()

            if batch_idx % 50 == 0:
                loop.set_description(f"Epoch [{epoch}/{EPOCHS}]")
                loop.set_postfix(
                    Loss_Critic=loss_critic.item(), 
                    Loss_Gen=loss_gen.item()
                )
                
    return generator, critic