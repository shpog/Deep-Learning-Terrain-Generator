import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from config import (
    LEARNING_RATE, BETA1, BETA2, CRITIC_ITERATIONS, LAMBDA_GP, LATENT_DIM
)

def compute_gradient_penalty(critic, real_samples, fake_samples, device):
    """
    Calculates the gradient penalty for WGAN-GP.
    
    1. Generates random interpolations between real and fake images.
    2. Passes interpolations through the Critic.
    3. Calculates gradients of the Critic's output with respect to the interpolations.
    4. Penalizes gradients that deviate from a norm (magnitude) of 1.0.
    """
    batch_size = real_samples.size(0)
    
    current_device = real_samples.device

    epsilon = torch.rand(batch_size, 1, 1, 1, device=current_device)
    
    interpolated = (epsilon * real_samples + ((1 - epsilon) * fake_samples)).requires_grad_(True)
    
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
    The main WGAN-GP training loop.
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
                critic_fake = critic(fake_images).reshape(-1)

                gp = compute_gradient_penalty(critic, real_images, fake_images)

                loss_critic = (
                    -(torch.mean(critic_real) - torch.mean(critic_fake)) 
                    + LAMBDA_GP * gp
                )

                critic.zero_grad()
                loss_critic.backward(retain_graph=True)
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
                
    return generator, critic