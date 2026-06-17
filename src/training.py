import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import torchvision.utils as vutils
import torch.nn.functional as F

from dataset import get_dataloader
from config import (
    LR_GEN,
    LR_CRITIC,
    BETA1,
    BETA2,
    CRITIC_ITERATIONS,
    LAMBDA_GP,
    LATENT_DIM,
    CONDITION_WIDTH,
    BATCH_SIZE,
)


def compute_gradient_loss(fake, real, mask):
    """
    Computes the L1 loss between the spatial gradients of the generated and real images.

    This function calculates the differences between adjacent pixels along both the
    x (horizontal) and y (vertical) axes to evaluate texture and structural fidelity.
    The loss is only calculated where the provided mask is valid.

    :param fake: The generated high-resolution terrain tensor.
    :type fake: torch.Tensor
    :param real: The ground truth high-resolution terrain tensor.
    :type real: torch.Tensor
    :param mask: A binary mask indicating valid regions for gradient calculation.
    :type mask: torch.Tensor
    :return: The sum of the L1 losses for the x and y spatial gradients.
    :rtype: torch.Tensor
    """
    dx_fake = fake[:, :, :, 1:] - fake[:, :, :, :-1]
    dx_real = real[:, :, :, 1:] - real[:, :, :, :-1]
    mask_dx = mask[:, :, :, 1:] * mask[:, :, :, :-1]

    dy_fake = fake[:, :, 1:, :] - fake[:, :, :-1, :]
    dy_real = real[:, :, 1:, :] - real[:, :, :-1, :]
    mask_dy = mask[:, :, 1:, :] * mask[:, :, :-1, :]

    mask_dx = (mask_dx == 1).expand_as(dx_fake)
    mask_dy = (mask_dy == 1).expand_as(dy_fake)

    loss_dx = (
        F.l1_loss(dx_fake[mask_dx], dx_real[mask_dx])
        if mask_dx.any()
        else torch.tensor(0.0, device=fake.device)
    )
    loss_dy = (
        F.l1_loss(dy_fake[mask_dy], dy_real[mask_dy])
        if mask_dy.any()
        else torch.tensor(0.0, device=fake.device)
    )

    return loss_dx + loss_dy


def compute_gradient_penalty(critic, real_samples, fake_samples, macro_guides):
    """
    Calculates the gradient penalty for the conditional Wasserstein GAN (WGAN-GP).

    This enforces the Lipschitz constraint by computing the norm of the critic's
    gradients with respect to random interpolations between real and generated samples,
    conditioned on the macro guides.

    :param critic: The conditional critic (discriminator) network.
    :type critic: torch.nn.Module
    :param real_samples: A batch of real high-resolution terrain tensors.
    :type real_samples: torch.Tensor
    :param fake_samples: A batch of generated high-resolution terrain tensors.
    :type fake_samples: torch.Tensor
    :param macro_guides: The low-resolution macro guides conditioning the critic.
    :type macro_guides: torch.Tensor
    :return: The computed gradient penalty to be added to the critic's loss.
    :rtype: torch.Tensor
    """
    batch_size = real_samples.size(0)
    current_device = real_samples.device
    fake_samples = fake_samples.detach()
    epsilon = torch.rand(batch_size, 1, 1, 1, device=current_device)
    interpolated = (
        epsilon * real_samples + ((1 - epsilon) * fake_samples)
    ).requires_grad_(True)

    if isinstance(critic, nn.DataParallel):
        critic_interpolated = critic.module(interpolated, macro_guides)
    else:
        critic_interpolated = critic(interpolated, macro_guides)

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
    return torch.mean((gradient_norm - 1.0) ** 2)


def train_wgan(generator, critic, device):
    """
    Executes the WGAN-GP training loop for the conditional high-resolution terrain generator.

    This function orchestrates the adversarial training process, updating the critic
    multiple times per generator update. It calculates critic loss (with gradient penalty)
    and generator loss (supplemented by an L1 penalty), while periodically saving
    checkpoint images of the generated elevation data.

    :param generator: The conditional terrain generator network.
    :type generator: torch.nn.Module
    :param critic: The conditional terrain critic network.
    :type critic: torch.nn.Module
    :param device: The PyTorch device (CPU or CUDA) to perform training on.
    :type device: torch.device
    :return: A tuple containing the fully trained generator and critic networks.
    :rtype: tuple(torch.nn.Module, torch.nn.Module)
    """
    opt_gen = optim.Adam(generator.parameters(), lr=LR_GEN, betas=(BETA1, BETA2))
    opt_critic = optim.Adam(critic.parameters(), lr=LR_CRITIC, betas=(BETA1, BETA2))

    from config import EPOCHS

    init_dataloader = get_dataloader(
        batch_size=BATCH_SIZE, num_workers=0, current_epoch=0
    )
    real_batch, macro_batch = next(iter(init_dataloader))
    real_batch = real_batch.to(device)
    macro_batch = macro_batch.to(device)
    del init_dataloader

    fixed_batch_size = min(16, real_batch.shape[0])
    fixed_noise = torch.randn(fixed_batch_size, LATENT_DIM, 1, 1, device=device)

    fixed_macro_guides = macro_batch[:fixed_batch_size]

    for epoch in range(EPOCHS):
        dataloader = get_dataloader(
            batch_size=BATCH_SIZE, num_workers=4, current_epoch=epoch
        )

        if epoch % 10 == 0:
            generator.eval()
            with torch.no_grad():
                fake_checkpoint = generator(fixed_noise, fixed_macro_guides).cpu()
                elevation_only = fake_checkpoint[:, 0:1, :, :]
                vutils.save_image(
                    elevation_only,
                    f"checkpoint_epoch_{epoch}.png",
                    nrow=4,
                    normalize=True,
                    value_range=(-1, 1),
                )
            generator.train()

        loop = tqdm(dataloader, leave=True)
        for batch_idx, (real_images, macro_guides) in enumerate(loop):
            real_images = real_images.to(device)
            macro_guides = macro_guides.to(device)
            batch_size = real_images.shape[0]

            for _ in range(CRITIC_ITERATIONS):
                noise = torch.randn(batch_size, LATENT_DIM, 1, 1, device=device)

                fake_images = generator(noise, macro_guides)

                critic_real = critic(real_images, macro_guides).reshape(-1)
                critic_fake = critic(fake_images.detach(), macro_guides).reshape(-1)

                gp = compute_gradient_penalty(
                    critic, real_images, fake_images, macro_guides
                )

                loss_critic = (
                    -(torch.mean(critic_real) - torch.mean(critic_fake))
                    + LAMBDA_GP * gp
                )

                critic.zero_grad()
                loss_critic.backward()
                opt_critic.step()

            output = critic(fake_images, macro_guides).reshape(-1)

            l1_penalty = F.l1_loss(fake_images, real_images)
            LAMBDA_L1 = 2.0
            loss_gen = -torch.mean(output) + (LAMBDA_L1 * l1_penalty)

            generator.zero_grad()
            loss_gen.backward()
            opt_gen.step()

            if batch_idx % 50 == 0:
                loop.set_description(f"Epoch [{epoch}/{EPOCHS}]")
                loop.set_postfix(
                    Loss_C=loss_critic.item(),
                    Loss_G=loss_gen.item(),
                    L1=l1_penalty.item(),
                )

    return generator, critic
