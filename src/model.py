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
    Conditional Generator network for the WGAN-GP.

    Takes a latent noise vector and an edge condition, and progressively upsamples them through 
    transposed convolutional layers to output a 3-channel image 
    (Elevation, Precipitation, Temperature) normalized between [0, 1].

    Attributes:
        latent_dim (int): The size of the input noise vector.
        model (torch.nn.Sequential): The sequential block of neural network layers.
    """
    def __init__(self, latent_dim=128, img_channels=3):
        """
        Initializes the TerrainGenerator.

        Args:
            latent_dim (int, optional): The dimensionality of the input noise vector. 
                Defaults to 128.
            img_channels (int, optional): The number of output channels (e.g., 3). 
                Defaults to 3.
        """
        super(TerrainGenerator, self).__init__()
        self.latent_dim = latent_dim
        
        # ENCODER: Input is now [Batch, 4, 256, 256] (Masked Condition + Mask)
        self.encoder = nn.Sequential(
            # 256 -> 128
            nn.Conv2d(img_channels + 1, 32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # 128 -> 64
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            # 64 -> 32
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # 32 -> 16
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # 16 -> 8
            nn.Conv2d(256, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # 8 -> 4
            nn.Conv2d(256, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Flatten to 1D and project linearly to match the latent_dim size
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, latent_dim)
        )

        combined_dim = latent_dim * 2
        
        self.generator = nn.Sequential(
            nn.ConvTranspose2d(combined_dim, 512, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            
            # Input: 4x4 -> Output: 8x8
            self._block(512, 256, 3, 1, 1),
            
            # Input: 8x8 -> Output: 16x16
            self._block(256, 128, 3, 1, 1),
            
            # Input: 16x16 -> Output: 32x32
            self._block(128, 64, 3, 1, 1),
            
            # Input: 32x32 -> Output: 64x64
            self._block(64, 32, 3, 1, 1),
            
            # Input: 64x64 -> Output: 128x128
            self._block(32, 16, 3, 1, 1),
            
            # Final Layer: 128x128 -> 256x256 (3 channels)
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(16, img_channels, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid() 
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        """
        Creates a standard Generator building block.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int): Size of the convolving kernel.
            stride (int): Stride of the convolution.
            padding (int): Zero-padding added to both sides of the input.

        Returns:
            torch.nn.Sequential: A block of ConvTranspose2d, BatchNorm2d, and ReLU.
        """
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, noise, condition, mask):
        """
        Executes the forward pass of the Conditional Generator.

        Args:
            noise (torch.Tensor): A noise tensor of shape [Batch, latent_dim, 1, 1].
            condition (torch.Tensor): Edge image of shape [Batch, img_channels, 256, 32].

        Returns:
            torch.Tensor: Generated terrain image of shape [Batch, img_channels, 256, 256].
        """
        encoder_input = torch.cat([condition, mask], dim=1)
        encoded_edge = self.encoder(encoder_input)
        encoded_edge = encoded_edge.view(encoded_edge.size(0), self.latent_dim, 1, 1)
        
        combined_input = torch.cat([noise, encoded_edge], dim=1)
        return self.generator(combined_input)


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
            nn.Conv2d(in_channels, 16, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Input: 128x128 -> Output: 64x64
            self._block(16, 32, 4, 2, 1),
            
            # Input: 64x64 -> Output: 32x32
            self._block(32, 64, 4, 2, 1),
            
            # Input: 32x32 -> Output: 16x16
            self._block(64, 128, 4, 2, 1),
            
            # Input: 16x16 -> Output: 8x8
            self._block(128, 256, 4, 2, 1),
            
            # Input: 8x8 -> Output: 4x4
            self._block(256, 512, 4, 2, 1),
            
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