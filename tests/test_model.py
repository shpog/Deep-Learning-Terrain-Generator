import torch
import pytest

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from model import TerrainGenerator, TerrainCritic

def test_generator_output_shape():
    """Ensures the Generator scales the latent vector up to exactly [Batch, 3, 256, 256]"""
    batch_size = 4
    latent_dim = 128
    channels = 3
    
    noise = torch.randn(batch_size, latent_dim, 1, 1)
    
    generator = TerrainGenerator(latent_dim=latent_dim, img_channels=channels)
    
    output = generator(noise)
    
    assert output.shape == (batch_size, channels, 256, 256), f"Bad Generator output shape: {output.shape}"

def test_critic_output_shape():
    """Ensures the Critic scales a 256x256 image down to a 1x1 scalar score"""
    batch_size = 4
    channels = 3
    
    fake_images = torch.randn(batch_size, channels, 256, 256)
    
    critic = TerrainCritic(img_channels=channels)
    
    output = critic(fake_images)
    
    assert output.shape == (batch_size, 1, 1, 1), f"Bad Critic output shape: {output.shape}"