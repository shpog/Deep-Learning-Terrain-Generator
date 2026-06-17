import torch
import torch.nn as nn
import os
from pathlib import Path

from dataset import get_dataloader
from model import TerrainGenerator, TerrainCritic, initialize_weights
from training import train_wgan

from config import BATCH_SIZE, EPOCHS, LATENT_DIM, IMG_CHANNELS, MODELS_DIR


def main():
    """
    The primary execution script for training the Conditional High-Resolution Terrain Generator.

    This function acts as the main entry point for the training pipeline. It detects
    available hardware (applying PyTorch's DataParallel if multiple GPUs are found),
    initializes the conditional TerrainGenerator and TerrainCritic networks, applies
    standard weight initialization, and triggers the WGAN-GP training loop. Finally,
    it ensures that the trained model weights are safely written to the models directory.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_count = torch.cuda.device_count()

    print(f"--- Terrain Generator AI ---")
    print(f"Using device: {device} with {gpu_count} GPU(s)")

    print("Initializing Generator and Critic...")
    generator = TerrainGenerator(latent_dim=LATENT_DIM, img_channels=IMG_CHANNELS)
    critic = TerrainCritic(img_channels=IMG_CHANNELS)

    initialize_weights(generator)
    initialize_weights(critic)

    if gpu_count > 1:
        print(f"Activating DataParallel across {gpu_count} GPUs!")
        generator = nn.DataParallel(generator)
        critic = nn.DataParallel(critic)

    generator = generator.to(device)
    critic = critic.to(device)

    initialize_weights(generator)
    initialize_weights(critic)

    print(f"\nStarting WGAN-GP Training for {EPOCHS} Epochs...")
    generator, critic = train_wgan(generator=generator, critic=critic, device=device)

    print("\nTraining complete! Saving model weights...")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    gen_weights = (
        generator.module.state_dict() if gpu_count > 1 else generator.state_dict()
    )
    crit_weights = critic.module.state_dict() if gpu_count > 1 else critic.state_dict()

    torch.save(gen_weights, MODELS_DIR / "generator.pth")
    torch.save(crit_weights, MODELS_DIR / "critic.pth")

    print(f"Models safely saved to: {MODELS_DIR}")


if __name__ == "__main__":
    main()
