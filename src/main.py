"""
Execution Entry Point for Terrain Generation Training.

This script ties together the dataset loaders, the neural network models, 
and the WGAN-GP training loop. It handles hardware detection (including 
multi-GPU cluster scaling via DataParallel) and saves the final weights.
"""

import torch
import torch.nn as nn
import os
from pathlib import Path

from dataset import get_dataloader
from model import TerrainGenerator, TerrainCritic, initialize_weights
from training import train_wgan

from config import (
    BATCH_SIZE, EPOCHS, LATENT_DIM, IMG_CHANNELS, MODELS_DIR
)

def main():
    """
    Initializes hardware, models, and begins the training loop.

    Workflow:
        1. Detects available GPUs and initializes CUDA if available.
        2. Loads the memory-mapped TerrainDataset via a multiprocessing DataLoader.
        3. Initializes the Generator and Critic networks.
        4. Wraps models in torch.nn.DataParallel if >1 GPU is detected.
        5. Executes the WGAN-GP training loop for the specified number of epochs.
        6. Safely unwraps and saves the final model weights to the disk.
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
    generator, critic = train_wgan(
        generator=generator, 
        critic=critic,
        device=device
    )
    
    print("\nTraining complete! Saving model weights...")
    
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    gen_weights = generator.module.state_dict() if gpu_count > 1 else generator.state_dict()
    crit_weights = critic.module.state_dict() if gpu_count > 1 else critic.state_dict()
    
    torch.save(gen_weights, MODELS_DIR / "generator.pth")
    torch.save(crit_weights, MODELS_DIR / "critic.pth")
    
    print(f"Models safely saved to: {MODELS_DIR}")

if __name__ == "__main__":
    main()