import torch
import pytest

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from model import TerrainGenerator, TerrainCritic

def test_generator_output_shape():
    batch_size = 4
    latent_dim = 128
    channels = 3
    
    noise = torch.randn(batch_size, latent_dim, 1, 1)
    
    # Conditional GAN: The Generator also needs a low-res guide!
    macro_guide = torch.randn(batch_size, channels, 256, 256)
    
    generator = TerrainGenerator(latent_dim=latent_dim, img_channels=channels)
    
    # Pass both noise and guide
    output = generator(noise, macro_guide)
    
    assert output.shape == (batch_size, channels, 256, 256), f"Bad Generator output shape: {output.shape}"

def test_critic_output_shape():
    batch_size = 4
    channels = 3
    
    fake_images = torch.randn(batch_size, channels, 256, 256)
    
    # Conditional GAN: The Critic concatenates the image with the macro guide!
    macro_guide = torch.randn(batch_size, channels, 256, 256)
    
    critic = TerrainCritic(img_channels=channels)
    
    # Pass both the evaluated image and its structural condition
    output = critic(fake_images, macro_guide)
    
    assert output.shape == (batch_size, 1, 1, 1), f"Bad Critic output shape: {output.shape}"