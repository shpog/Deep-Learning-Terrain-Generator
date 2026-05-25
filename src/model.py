"""
Neural Network Architectures for Terrain Generation.

This module defines the PyTorch modules for the Wasserstein GAN with 
Gradient Penalty (WGAN-GP). It includes the Generator (which upsamples 
noise into terrain maps) and the Critic (which scores the realism of maps).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class TerrainGenerator(nn.Module):
    """
    Bottleneck Outpainting Generator.
    Compresses the edge condition into a spatial bottleneck, merges it with 
    the latent noise, and uses Dropout to enforce generative variance.
    """
    def __init__(self, latent_dim=128, img_channels=3):
        super(TerrainGenerator, self).__init__()
        self.latent_dim = latent_dim
        
        # ENCODER: Downsamples to 4x4. No skip connections.
        self.encoder = nn.Sequential(
            self._conv_block(img_channels + 1, 32),   # 256 -> 128
            self._conv_block(32, 64),                 # 128 -> 64
            self._conv_block(64, 128),                # 64 -> 32
            self._conv_block(128, 256),               # 32 -> 16
            self._conv_block(256, 512),               # 16 -> 8
            self._conv_block(512, 512)                # 8 -> 4
        )
        
        # NOISE MAPPING: Project 1D latent vector to 3D spatial tensor
        self.noise_fc = nn.Linear(latent_dim, 512 * 4 * 4)
        
        # DECODER: Upsamples from 4x4 back to 256x256
        self.decoder = nn.Sequential(
            # Bottleneck input: 512 (encoded edge) + 512 (noise) = 1024
            self._up_block(1024, 512, use_dropout=True),  # 4 -> 8
            self._up_block(512, 256, use_dropout=True),   # 8 -> 16
            self._up_block(256, 128, use_dropout=True),   # 16 -> 32
            self._up_block(128, 64),                      # 32 -> 64
            self._up_block(64, 32),                       # 64 -> 128
            
            # Final Layer: 128 -> 256
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.ReflectionPad2d(1),
            nn.Conv2d(32, img_channels, kernel_size=3, stride=1, padding=0),
            nn.Sigmoid() 
        )

    def _conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def _up_block(self, in_channels, out_channels, use_dropout=False):
        layers = [
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        if use_dropout:
            layers.append(nn.Dropout2d(0.3))
            
        return nn.Sequential(*layers)

    def forward(self, noise, condition, mask):
        x = torch.cat([condition, mask], dim=1)
        encoded_edge = self.encoder(x)
        
        b_size = noise.size(0)
        noise_flat = noise.view(b_size, self.latent_dim)
        noise_spatial = self.noise_fc(noise_flat).view(b_size, 512, 4, 4)
        
        bottleneck = torch.cat([encoded_edge, noise_spatial], dim=1)
        
        return self.decoder(bottleneck)


class TerrainCritic(nn.Module):
    """
    The Critic (Discriminator) network for conditional WGAN-GP.

    Evaluates the realism of the generated tile and its alignment with the edge condition.

    Attributes:
        model (torch.nn.Sequential): The sequential block of neural network layers.
    """
    def __init__(self, img_channels=3):
        """
        Initializes the TerrainCritic.

        Args:
            img_channels (int, optional): The number of input channels (e.g., 3). 
                Defaults to 3.
        """
        super(TerrainCritic, self).__init__()

        in_channels = (img_channels * 2) + 1
        
        self.model = nn.Sequential(
            # Input: [Batch, 3, 256, 256] -> Output: [Batch, 16, 128, 128]
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, 16, kernel_size=4, stride=2, padding=0),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Input: 128x128 -> Output: 64x64
            self._block(16, 32, 4, 2, 0),
            
            # Input: 64x64 -> Output: 32x32
            self._block(32, 64, 4, 2, 0),
            
            # Input: 32x32 -> Output: 16x16
            self._block(64, 128, 4, 2, 0),
            
            # Input: 16x16 -> Output: 8x8
            self._block(128, 256, 4, 2, 0),
            
            # Input: 8x8 -> Output: 4x4
            self._block(256, 512, 4, 2, 0),
            
            # Final Layer: 4x4 -> 1x1 scalar
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0) 
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        """
        Creates a standard Critic building block safely using InstanceNorm.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int): Size of the convolving kernel.
            stride (int): Stride of the convolution.
            padding (int): Zero-padding added to both sides of the input.

        Returns:
            torch.nn.Sequential: A block of Conv2d, InstanceNorm2d, and LeakyReLU.
        """
        return nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True), # NO BatchNorm here!
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x, condition, mask):
        """
        Args:
            x: The terrain patch [Batch, 3, 256, 256]
            condition: The edge hint [Batch, 3, 256, 32]
        """
        combined = torch.cat([x, condition, mask], dim=1)
        return self.model(combined)


def initialize_weights(model):
    """
    Initializes the network weights according to a normal distribution.

    Args:
        model (torch.nn.Module): The PyTorch neural network module to initialize.
    """
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0)