import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import torchvision.utils as vutils
import os

from config import (
    BATCH_SIZE,
    EPOCHS,
    LATENT_DIM,
    IMG_CHANNELS,
    MODELS_DIR,
    LR_GEN,
    LR_CRITIC,
    BETA1,
    BETA2,
    CRITIC_ITERATIONS,
    LAMBDA_GP,
)
from dataset import get_macro_dataloader
from model import MacroGenerator, MacroCritic, initialize_weights


def compute_gradient_penalty_unconditional(critic, real_samples, fake_samples, device):
    """
    Calculates the gradient penalty for the unconditional Wasserstein GAN (WGAN-GP).

    This enforces the 1-Lipschitz constraint by evaluating the critic's gradients
    on random interpolations between real and generated macro terrain samples,
    without any external conditioning.

    :param critic: The unconditional macro critic (discriminator) network.
    :type critic: torch.nn.Module
    :param real_samples: A batch of real low-resolution macro terrain tensors.
    :type real_samples: torch.Tensor
    :param fake_samples: A batch of generated low-resolution macro terrain tensors.
    :type fake_samples: torch.Tensor
    :param device: The PyTorch device used for computation.
    :type device: torch.device
    :return: The calculated unconditional gradient penalty.
    :rtype: torch.Tensor
    """
    batch_size = real_samples.size(0)

    fake_samples = fake_samples.detach()

    epsilon = torch.rand(batch_size, 1, 1, 1, device=device)

    interpolated = (
        epsilon * real_samples + ((1 - epsilon) * fake_samples)
    ).requires_grad_(True)

    if isinstance(critic, nn.DataParallel):
        critic_interpolated = critic.module(interpolated)
    else:
        critic_interpolated = critic(interpolated)

    gradients = torch.autograd.grad(
        outputs=critic_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones_like(critic_interpolated, device=device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradients = gradients.view(batch_size, -1)
    gradient_norm = gradients.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1.0) ** 2)

    return gradient_penalty


def main():
    """
    The primary execution script for training the unconditional Macro-level Terrain Generator.

    Initializes the macro generator and critic networks, configures the optimizers,
    and runs the WGAN-GP training loop over the specified number of epochs.
    It supports multi-GPU training via DataParallel, generates periodic visual
    checkpoints, and safely saves the model weights upon completion.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_count = torch.cuda.device_count()

    print(f"--- Trening Modelu Makro (Kontynenty) ---")
    print(f"Urządzenie: {device} | Liczba GPU: {gpu_count}")

    generator = MacroGenerator(latent_dim=LATENT_DIM, img_channels=IMG_CHANNELS)
    critic = MacroCritic(img_channels=IMG_CHANNELS)

    initialize_weights(generator)
    initialize_weights(critic)

    if gpu_count > 1:
        generator = nn.DataParallel(generator)
        critic = nn.DataParallel(critic)

    generator = generator.to(device)
    critic = critic.to(device)

    opt_gen = optim.Adam(generator.parameters(), lr=LR_GEN, betas=(BETA1, BETA2))
    opt_critic = optim.Adam(critic.parameters(), lr=LR_CRITIC, betas=(BETA1, BETA2))

    dataloader = get_macro_dataloader(batch_size=BATCH_SIZE, num_workers=4)

    fixed_noise = torch.randn(16, LATENT_DIM, 1, 1, device=device)

    print("\nRozpoczynamy pętlę treningową...")

    for epoch in range(EPOCHS):

        if epoch % 5 == 0:
            generator.eval()
            with torch.no_grad():
                fake_checkpoint = generator(fixed_noise).cpu()
                elevation_only = fake_checkpoint[:, 0:1, :, :]
                vutils.save_image(
                    elevation_only,
                    f"macro_checkpoint_epoch_{epoch}.png",
                    nrow=4,
                    normalize=True,
                    value_range=(-1, 1),
                )
            generator.train()

        loop = tqdm(dataloader, leave=True)
        for batch_idx, real_images in enumerate(loop):
            real_images = real_images.to(device)
            current_batch_size = real_images.shape[0]

            for _ in range(CRITIC_ITERATIONS):
                noise = torch.randn(current_batch_size, LATENT_DIM, 1, 1, device=device)
                fake_images = generator(noise)

                critic_real = critic(real_images).reshape(-1)
                critic_fake = critic(fake_images.detach()).reshape(-1)

                gp = compute_gradient_penalty_unconditional(
                    critic, real_images, fake_images, device
                )

                loss_critic = (
                    -(torch.mean(critic_real) - torch.mean(critic_fake))
                    + LAMBDA_GP * gp
                )

                critic.zero_grad()
                loss_critic.backward()
                opt_critic.step()

            noise_gen = torch.randn(current_batch_size, LATENT_DIM, 1, 1, device=device)
            fake_images_gen = generator(noise_gen)

            output = critic(fake_images_gen).reshape(-1)

            loss_gen = -torch.mean(output)

            generator.zero_grad()
            loss_gen.backward()
            opt_gen.step()

            if batch_idx % 20 == 0:
                loop.set_description(f"Makro Epoka [{epoch}/{EPOCHS}]")
                loop.set_postfix(Loss_C=loss_critic.item(), Loss_G=loss_gen.item())

    print("\nTrening Makro zakończony! Zapisywanie wag...")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    gen_weights = (
        generator.module.state_dict() if gpu_count > 1 else generator.state_dict()
    )
    crit_weights = critic.module.state_dict() if gpu_count > 1 else critic.state_dict()

    torch.save(gen_weights, MODELS_DIR / "macro_generator.pth")
    torch.save(crit_weights, MODELS_DIR / "macro_critic.pth")
    print(f"Wagi Makro zapisane bezpiecznie w: {MODELS_DIR}")


if __name__ == "__main__":
    main()
