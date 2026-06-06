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

from dataset import get_dataloader
from config import (
    LR_GEN, LR_CRITIC, BETA1, BETA2, CRITIC_ITERATIONS, LAMBDA_GP, LATENT_DIM, CONDITION_WIDTH, BATCH_SIZE
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

def train_wgan(generator, critic, device):
    """
    Executes the main WGAN-GP training loop with conditional stitching.
    """
    opt_gen = optim.Adam(generator.parameters(), lr=LR_GEN, betas=(BETA1, BETA2))
    opt_critic = optim.Adam(critic.parameters(), lr=LR_CRITIC, betas=(BETA1, BETA2))

    from config import EPOCHS

    init_dataloader = get_dataloader(batch_size=BATCH_SIZE, num_workers=0, current_epoch=0)
    real_batch = next(iter(init_dataloader)).to(device)
    del init_dataloader

    fixed_batch_size = min(16, real_batch.shape[0])
    
    fixed_noise = torch.randn(fixed_batch_size, LATENT_DIM, 1, 1, device=device)
    fixed_masks = torch.zeros(fixed_batch_size, 1, 256, 256, device=device)
    
    fixed_masks[:, :, :, :CONDITION_WIDTH] = 1 
    fixed_conditions = real_batch[:fixed_batch_size] * fixed_masks

    for epoch in range(EPOCHS):
        
        dataloader = get_dataloader(batch_size=BATCH_SIZE, num_workers=4, current_epoch=epoch)

        # --- VISUAL CHECKPOINTING ---
        if epoch % 10 == 0:
            generator.eval()
            with torch.no_grad():
                fake_checkpoint = generator(fixed_noise, fixed_conditions, fixed_masks).cpu()
                # Assuming channel 0 is Elevation
                elevation_only = fake_checkpoint[:, 0:1, :, :] 
                vutils.save_image(elevation_only, f"checkpoint_epoch_{epoch}.png", nrow=4, normalize=True, value_range=(-1, 1))
            generator.train()

        loop = tqdm(dataloader, leave=True)
        for batch_idx, real_images in enumerate(loop):
            real_images = real_images.to(device)
            batch_size = real_images.shape[0]

            masks = torch.zeros(batch_size, 1, 256, 256, device=device)
            
            # Losujemy liczbę krawędzi do zamaskowania: od 0 do 4
            num_edges = torch.randint(0, 5, (1,)).item()
            
            if num_edges > 0:
                chosen_edges = torch.randperm(4)[:num_edges]
                
                for edge in chosen_edges:
                    if edge == 0:   masks[:, :, :, :CONDITION_WIDTH] = 1
                    elif edge == 1: masks[:, :, :, -CONDITION_WIDTH:] = 1
                    elif edge == 2: masks[:, :, :CONDITION_WIDTH, :] = 1
                    elif edge == 3: masks[:, :, -CONDITION_WIDTH:, :] = 1
            
            conditions = real_images * masks

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

            output = critic(fake_images, conditions, masks).reshape(-1)
            
            mask_pixels = (masks == 1).expand_as(fake_images)
            if mask_pixels.any():
                l1_penalty = F.l1_loss(fake_images[mask_pixels], conditions[mask_pixels])
            else:
                l1_penalty = torch.tensor(0.0, device=device)
            
            loss_gen = -torch.mean(output)# + (0.5 * l1_penalty)

            generator.zero_grad()
            loss_gen.backward()
            opt_gen.step()

            if batch_idx % 50 == 0:
                loop.set_description(f"Epoch [{epoch}/{EPOCHS}]")
                loop.set_postfix(
                    Loss_Critic=loss_critic.item(), 
                    Loss_Gen=loss_gen.item(),
                    L1_Border=l1_penalty.item()
                )
                
    return generator, critic