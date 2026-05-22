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

from config import (
    LEARNING_RATE, BETA1, BETA2, CRITIC_ITERATIONS, LAMBDA_GP, LATENT_DIM
)

def compute_gradient_penalty(critic, real_samples, fake_samples):
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
        critic_interpolated = critic.module(interpolated)
    else:
        critic_interpolated = critic(interpolated)
    
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
        loop = tqdm(dataloader, leave=True)
        for batch_idx, real_images in enumerate(loop):
            real_images = real_images.to(device)
            batch_size = real_images.shape[0]

            for _ in range(CRITIC_ITERATIONS):
                noise = torch.randn(batch_size, LATENT_DIM, 1, 1, device=device)
                fake_images = generator(noise)

                critic_real = critic(real_images).reshape(-1)
                critic_fake = critic(fake_images.detach()).reshape(-1)

                gp = compute_gradient_penalty(critic, real_images, fake_images)

                loss_critic = (
                    -(torch.mean(critic_real) - torch.mean(critic_fake)) 
                    + LAMBDA_GP * gp
                )

                critic.zero_grad()
                loss_critic.backward()
                opt_critic.step()

            output = critic(fake_images).reshape(-1)
            loss_gen = -torch.mean(output)

            generator.zero_grad()
            loss_gen.backward()
            opt_gen.step()

            if batch_idx % 50 == 0:
                loop.set_description(f"Epoch [{epoch}/{EPOCHS}]")
                loop.set_postfix(
                    Loss_Critic=loss_critic.item(), 
                    Loss_Gen=loss_gen.item()
                )
            if epoch % 10 == 0:
                generator.eval()
                with torch.no_grad():
                    fixed_noise = torch.randn(16, LATENT_DIM, 1, 1, device=device)
                    fake_checkpoint = generator(fixed_noise).cpu()
                    elevation_only = fake_checkpoint[:, 0:1, :, :] 
                    vutils.save_image(elevation_only, f"checkpoint_epoch_{epoch}.png", nrow=4, normalize=True)
                generator.train()
                
    return generator, critic