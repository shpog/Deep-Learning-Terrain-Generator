import torch
import os
from pathlib import Path

from dataset import get_dataloader
from model import TerrainGenerator, TerrainCritic, initialize_weights
from training import train_wgan

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Terrain Generator AI ---")
    print(f"Using device: {device}")

    BATCH_SIZE = 32 
    EPOCHS = 100
    LATENT_DIM = 128
    
    print("\nInitializing DataLoader...")
    dataloader = get_dataloader(batch_size=BATCH_SIZE, num_workers=0)
    
    print("Initializing Generator and Critic...")
    generator = TerrainGenerator(latent_dim=LATENT_DIM, img_channels=3).to(device)
    critic = TerrainCritic(img_channels=3).to(device)
    
    initialize_weights(generator)
    initialize_weights(critic)
    
    print("\nStarting WGAN-GP Training...")
    generator, critic = train_wgan(
        generator=generator, 
        critic=critic, 
        dataloader=dataloader, 
        device=device, 
        epochs=EPOCHS, 
        latent_dim=LATENT_DIM
    )
    
    print("\nTraining complete! Saving model weights...")
    
    project_root = Path(__file__).resolve().parent.parent
    models_dir = project_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    torch.save(generator.state_dict(), models_dir / "generator.pth")
    torch.save(critic.state_dict(), models_dir / "critic.pth")
    
    print(f"Models safely saved to: {models_dir}")

if __name__ == "__main__":
    main()